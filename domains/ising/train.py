"""R3GAN training for Ising update rules.

Supports REGIS, the non-local control (--g-arch control), and the
pair-conditional GAN (--g-arch control --pair-cond). The paired variant
conditions its discriminator on the starting state and obtains the real
endpoint by evolving that state with the Glauber engine.

One generator step corresponds to one Glauber sweep. Each iteration samples
consecutive marginals, starts from real crops or the replay pool, and rolls
through their time interval. Gradients cover the final grad_window updates.
Domain length L(t) is used only for evaluation.

RECORD contains the paper's configuration; --fixed-crops and --n-train select
the data-efficiency experiments. Commands are in docs/reproduce.md.
Smoke test: python -m tests.test_smoke_train
"""
from __future__ import annotations

import argparse
import collections
import sys
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel

from workspace import WORK, resolve  # gitignored artifact dirs, or $REGIS_WORK

from domains.ising.glauber import sweep_
from models.train.disc import Critic
from models.train.logger import Logger
from models.train.r3gan import relativistic, zero_centered_gp
from models.nca import NCA
from models.control import NonLocalControl


# Paper training configuration, following tab:ising_recipe.
# Command-line overrides are saved in config.json and checkpoints.
RECORD = dict(
    lr=2e-4, beta1=0.0, beta2=0.9,        # optimiser: AdamW, constant learning rate
    batch_size=8, crop=128,               # batch / crop: random periodic 128x128 windows
    total_steps=30_000,
    gamma=1.0,                            # adversarial loss: weight of the zero-centred gradient penalties
    grad_window=45,                       # gradient window: backpropagate through the last 45 steps of a roll
    ema_decay=0.999, ema_warmup=1000,     # decay = min(ema_decay, (1+n)/(ema_warmup+n))
    channels=1,                           # the spin alone, no hidden channels
    mlp_width=128,                        # width of REGIS's per-cell MLP
    conv_base=38, conv_dilations="1,2,5,9",   # the non-local control: width, one 3x3 conv per dilation
    grad_clip=1.0,                        # gradient-norm clip, G and D
    pool_frac=0.3, pool_cap=192,          # replay pool: start probability at marginal s >= 1, size per marginal
    d_base=32, d_down=4,                  # critic width and stride-2 blocks
    log_every=20, ckpt_every=2500,        # scalar log and checkpoint intervals; the last step always saves
)
# the tiny run of tests/test_smoke_train.py
SMOKE = dict(total_steps=6, batch_size=4, crop=16, grad_window=3, mlp_width=32, conv_base=8, d_base=16,
             ema_warmup=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g-arch", choices=["regis", "control"], default="regis",
                   help="REGIS or the non-local control (a stack of dilated convolutions)")
    p.add_argument("--perception-ksize", type=int, default=3,
                   help="REGIS's perception kernel (3, 5, 7: the k3/k5/k7 rows)")
    p.add_argument("--pair-cond", action="store_true",
                   help="the critic sees (start, endpoint); the real endpoint is the engine "
                        "rolled the same number of sweeps from the same start. The paper's "
                        "pair-conditional GAN is --g-arch control --pair-cond")
    p.add_argument("--fixed-crops", action="store_true",
                   help="every real crop is its field's (0,0) window, no random origin (data ladder)")
    p.add_argument("--n-train", type=int, default=0,
                   help="train on the first N fields per marginal (0: all; data ladder)")
    p.add_argument("--bf16-prefix", action="store_true",
                   help="bf16 autocast on the no-grad part of each roll (data ladder)")
    p.add_argument("--data", default=str(WORK / "data" / "ising_marginals.npz"),
                   help="marginals file; a relative path resolves under the artifact tree")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="ising_dev")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile-step", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the generator step")
    p.add_argument("--smoke-test", action="store_true", help="the tiny SMOKE run of tests/test_smoke_train.py")
    # the ablation switches; every run in the paper uses the defaults
    p.add_argument("--update-clip", type=float, default=10.0, help="clip c on each update")
    p.add_argument("--hinge-weight", type=float, default=1e-4,
                   help="weight lambda of the update hinge L_upd, the mean of relu(|f| - 1)^2 "
                        "over each raw branch")
    p.add_argument("--fire-mode", default="checker2",
                   help="checker2: a checkerboard sweep, parity A then B, one step per sweep; "
                        "bernoulli: an iid mask at --fire-rate")
    p.add_argument("--fire-rate", type=float, default=1.0, help="firing probability under bernoulli")
    p.add_argument("--z2-sym", action=argparse.BooleanOptionalAction, default=True,
                   help="Z2-antisymmetrised update (no phase bias)")
    p.add_argument("--parity-even", action=argparse.BooleanOptionalAction, default=True,
                   help="spatial kernels symmetric under 180-degree rotation (no net advection)")
    p.add_argument("--noise-channels", type=int, default=2,
                   help="per-cell N(0, 1) noise channels, fresh every half-step")
    args = argparse.Namespace(**RECORD, **vars(p.parse_args()))
    if args.smoke_test:
        vars(args).update(SMOKE)
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    path = resolve(args.data)
    if not path.exists():
        sys.exit(f"no dataset at {path}\n  generate it first (docs/reproduce.md), "
                 "or point REGIS_WORK at the tree that holds data/")
    data = np.load(path)
    marginal_ts = [0] + data["stage_ts"].tolist()    # t_0 = 0 (iid), then the observed marginals
    n_marginals = len(marginal_ts) - 1                # marginals 1..n_marginals
    train = {s: data[f"train_s{s}"] for s in range(1, n_marginals + 1)}
    if args.n_train:
        train = {k: v[: args.n_train] for k, v in train.items()}
        print(f"DATA BUDGET: {args.n_train} snapshots/marginal", flush=True)
    CS = args.crop
    LAT = int(data["lat"])

    def crops(arr: np.ndarray, n: int) -> torch.Tensor:
        """(n, 1, CS, CS) float 0/1 crops of random fields: periodic at random
        origins, or the (0,0) window under --fixed-crops."""
        idx = rng.integers(0, arr.shape[0], n)
        if args.fixed_crops:
            out = arr[idx, :CS, :CS].astype(np.float32)
            return torch.from_numpy(out)[:, None].to(device)
        oy, ox = rng.integers(0, LAT, n), rng.integers(0, LAT, n)
        out = np.empty((n, CS, CS), np.float32)
        for i in range(n):
            r = np.roll(arr[idx[i]], (-int(oy[i]), -int(ox[i])), (0, 1))
            out[i] = r[:CS, :CS]
        return torch.from_numpy(out)[:, None].to(device)

    G = (NonLocalControl if args.g_arch == "control" else NCA).from_hparams(vars(args)).to(device)

    def _ema_avg(avg_p, p, n):
        decay = min(args.ema_decay, (1 + n) / (args.ema_warmup + n))
        return decay * avg_p + (1 - decay) * p
    # built before the compile, so the EMA copy keeps its own, uncompiled step
    ema = AveragedModel(G, avg_fn=_ema_avg).to(device)

    # Compile the repeated update to reduce GPU launch overhead.
    if args.compile_step:
        G.step = torch.compile(G.step)
        print("compiled G.step")

    d_in = (2 if args.pair_cond else 1) + n_marginals   # [start,] x, one time plane per marginal
    D = Critic(in_channels=d_in, base=args.d_base, n_down=args.d_down).to(device)
    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"params: G={n_g/1e3:.1f}K D={n_d/1e3:.1f}K | crop {CS} marginals t={marginal_ts} "
          f"device={device}", flush=True)

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.01)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.01)

    out_root = WORK / "runs" / args.run_name
    config = dict(vars(args))
    config.update(G_params=n_g, D_params=n_d, stage_ts=marginal_ts)
    logger = Logger(run_dir=out_root, config=config)
    print(f"run_dir: {out_root}", flush=True)

    def fresh_noise() -> torch.Tensor:
        return torch.randn(args.batch_size, args.noise_channels, CS, CS, device=device)

    def update(st: torch.Tensor) -> torch.Tensor:
        """one generator update, fresh noise for each half-step"""
        return G.step(st, noise=fresh_noise(), noise2=fresh_noise())

    def roll(state: torch.Tensor, n_updates: int):
        """z_{t_a} -> z_{t_b}, with the gradient through the last grad_window
        updates. Returns the endpoint and the mean relu(|f| - 1)^2 over the
        gradient window (L_upd before its weight lambda)."""
        ngrad = min(args.grad_window, n_updates)
        G.start_hinge(device)        # the prefix accumulates too; discarded below
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16_prefix):
            for _ in range(n_updates - ngrad):
                state = update(state)
        state = state.float()        # fp32 into the gradient window
        G.start_hinge(device)        # L_upd counts the gradient window only
        for _ in range(ngrad):
            state = update(state)
        return state, G.hinge_sum / G.hinge_calls

    # replay pool: rolled states that start later segments
    pool = {s: collections.deque(maxlen=args.pool_cap) for s in range(1, n_marginals)}

    eng_gen = torch.Generator(device=device).manual_seed(args.seed + 12345)

    @torch.no_grad()
    def engine_succ(start: torch.Tensor, n_steps: int) -> torch.Tensor:
        """The real successor of the same start, (B,1,CS,CS) 0/1 -> (B,1,CS,CS)
        0/1: the Glauber engine, n_steps sweeps, periodic on the crop."""
        sp = torch.where(start[:, 0] > 0.5, 1.0, -1.0)
        for _ in range(n_steps):
            sweep_(sp, 1.0, eng_gen)
        return (sp > 0).float()[:, None]

    def critic_input(x: torch.Tensor, s_next: int, start=None) -> torch.Tensor:
        """The input of D(x, t_b): [start,] x, and the target time t_b as one
        one-hot plane per marginal. R1 and R2 differentiate D with respect to
        all of it."""
        t_planes = torch.zeros(x.shape[0], n_marginals, CS, CS, device=device)
        t_planes[:, s_next - 1] = 1.0
        return torch.cat(([start, x] if args.pair_cond else [x]) + [t_planes], 1)

    t0 = time.time()
    for step in range(args.total_steps):
        s = int(rng.integers(0, n_marginals))            # the marginal pair (s -> s+1)
        s_next = s + 1
        t_a, t_b = marginal_ts[s], marginal_ts[s_next]

        # 1. trajectory generation t_a -> t_b (delta = 1: the critic reads x_{t_b})
        use_pool = (s >= 1 and args.pool_frac > 0
                    and len(pool[s]) >= args.batch_size
                    and rng.random() < args.pool_frac)
        if use_pool:
            sel = rng.integers(0, len(pool[s]), args.batch_size)
            init = torch.stack([pool[s][i] for i in sel])
        elif s == 0:        # iid spins at t_0 = 0
            init = (torch.rand(args.batch_size, 1, CS, CS, device=device) < 0.5).float()
        else:
            init = crops(train[s], args.batch_size)

        start = None
        if args.pair_cond:
            # Binarise the shared start for the generator, engine, and paired critic.
            init = start = (init > 0.5).float()
        state, upd_hinge = roll(init, t_b - t_a)
        x_gen = G.readout(state)

        # 2. distribution comparison: x_obs ~ p_{t_b}, independent of the start
        #    (pair GAN: the engine's successor of the same start)
        x_obs = (engine_succ(start, t_b - t_a) if args.pair_cond
                 else crops(train[s_next], args.batch_size))

        # 3. adversarial optimisation: L_D, then L_NCA
        gen_in = critic_input(x_gen.detach(), s_next, start).requires_grad_(True)
        obs_in = critic_input(x_obs, s_next, start).requires_grad_(True)
        d_obs, d_gen = D(obs_in), D(gen_in)
        r1 = zero_centered_gp(obs_in, d_obs).mean()
        r2 = zero_centered_gp(gen_in, d_gen).mean()
        adv_d = relativistic(d_obs, d_gen)
        loss_d = adv_d + 0.5 * args.gamma * (r1 + r2)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        d_gn = clip_grad_norm_(D.parameters(), args.grad_clip)
        opt_d.step()

        with torch.no_grad():
            d_obs_new = D(critic_input(x_obs, s_next, start))      # D has stepped: score x_obs again
        g_adv = relativistic(D(critic_input(x_gen, s_next, start)), d_obs_new)
        loss_nca = g_adv + args.hinge_weight * upd_hinge
        opt_g.zero_grad(set_to_none=True)
        loss_nca.backward()
        g_gn = clip_grad_norm_(G.parameters(), args.grad_clip)
        opt_g.step()
        ema.update_parameters(G)

        # rolled states become future starts at marginal s_next
        if s_next < n_marginals:
            pool[s_next].extend(state.detach())

        if step % args.log_every == 0:
            logger.scalars({
                "loss/D": loss_d.item(), "loss/G": loss_nca.item(),
                "loss/D_adv": adv_d.item(), "loss/G_adv": g_adv.item(),
                "penalty/gp_real": r1.item(), "penalty/gp_fake": r2.item(),
                "logits/D_real": d_obs.mean().item(), "logits/D_fake": d_gen.mean().item(),
                "grad/D": float(d_gn), "grad/G": float(g_gn),
                "penalty/update_hinge": float(upd_hinge.detach()),
                "perf/steps_per_s": (step + 1) / (time.time() - t0),
            }, step=step)

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.total_steps:
            ckpt = {"step": step + 1, "G": G.state_dict(), "D": D.state_dict(),
                    "ema_G": ema.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "args": {"hparams": vars(args)}}
            (out_root / "ckpts").mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out_root / "ckpts" / f"ckpt_{step + 1:06d}.pt")

    logger.close(status="completed")
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
