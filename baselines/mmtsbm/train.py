"""Train MMtSBM with independent-coupling warm-up and iterative Markovian fitting.

After warm-up, each drift is fitted to endpoint pairs sampled with the
opposite direction's EMA drift. Ising uses forward-only warm-up, circular
padding, symmetry averaging, and a central loss window. MNIST trains both
directions through warm-up and two IMF iterations, hearts through warm-up and
three. Commands and settings are in docs/reproduce.md."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from workspace import WORK, resolve   # gitignored artifact dirs, or $REGIS_WORK
from baselines.mmtsbm.bridge import bridge_batch, simulate
from baselines.mmtsbm.net import build_drift

OPPOSITE = {"forward": "backward", "backward": "forward"}
# samples per batch when the IMF coupling (outer iterations >= 1) is simulated;
# the batches draw their noise in turn, so another value changes the coupling
CHUNK = 2048


def load_marginals(data_dir: Path, n_marginals: int, device, prefix="ising_t"):
    """Marginal i -> the (N, C, H, W) float32 tensor stored at <prefix><i>.pt
    (ising_t: Ising windows, +-1 spins; class_t: the MNIST digit loop, grey
    levels; hearts_t: the 21 heart channels, z-scored)."""
    xs = []
    for i in range(n_marginals):
        f = data_dir / f"{prefix}{i}.pt"
        if not f.is_file():
            raise SystemExit(f"missing marginal {i} at {f}\n"
                             f"  (Ising: python -m domains.ising.data windows; "
                             f"MNIST: python -m domains.mnist.data digitloop; "
                             f"hearts: python -m domains.hearts.data_bridges)")
        xs.append(torch.load(f).to(device))
    return xs


@torch.no_grad()
def couple(marginals, bridges, emas, sigma, direction, outer, n_sim):
    """Construct (z0, z1) pairs, with z0 at the earlier marginal.

    Use independent samples during warm-up and the opposite EMA drift afterward.
    Both endpoints are sampled with replacement to max(n_left, n_right) pairs."""
    pairs = []
    for bi, (a, b) in enumerate(bridges):
        left, right = marginals[bi], marginals[bi + 1]
        n = max(len(left), len(right))
        z0 = left[torch.randint(len(left), (n,), device=left.device)]
        z1 = right[torch.randint(len(right), (n,), device=right.device)]
        if outer > 0:
            prev = OPPOSITE[direction]
            start = z1 if direction == "forward" else z0
            end = torch.cat([simulate(emas[prev], start[c:c + CHUNK], a, b, sigma, prev, n_sim)
                             for c in range(0, n, CHUNK)])
            z0, z1 = (end, z1) if direction == "forward" else (z0, end)
        pairs.append((z0, z1))
    return pairs


def train_one_direction(drift, ema, opt, pairs, bridges, direction, n_steps, args,
                        outer):
    """Bridge-match `direction` against a fixed coupling. Returns mean loss."""
    z0 = torch.cat([p[0] for p in pairs])
    z1 = torch.cat([p[1] for p in pairs])
    bi = torch.cat([torch.full((len(p[0]),), i, device=z0.device)
                    for i, p in enumerate(pairs)])
    t_a = torch.tensor([b[0] for b in bridges], device=z0.device)[bi]
    t_b = torch.tensor([b[1] for b in bridges], device=z0.device)[bi]
    t_max = float(bridges[-1][1])

    perm, ptr, losses = None, 0, torch.zeros(n_steps, device=z0.device)
    t0 = time.time()
    for step in range(n_steps):
        if perm is None or ptr + args.batch > len(perm):
            perm, ptr = torch.randperm(len(z0), device=z0.device), 0
        idx = perm[ptr:ptr + args.batch]
        ptr += args.batch

        z_s, s, fwd, bwd = bridge_batch(z0[idx], z1[idx], args.sigma, args.eps)
        target = fwd if direction == "forward" else bwd
        t = t_a[idx].view(-1, 1) + s.view(-1, 1) * (t_b[idx] - t_a[idx]).view(-1, 1)
        # weight sigma * sqrt(absolute time travelled from the direction's
        # first marginal): forward sqrt(t), backward sqrt(t_max - t)
        w = args.sigma * (t if direction == "forward" else t_max - t).sqrt()
        w = w.view(-1, *([1] * (z_s.ndim - 1)))
        pred, tgt = w * drift(z_s, t), w * target
        if args.halo:                     # overlap-tile: the window's outer ring is never a target
            h = args.halo
            pred, tgt = pred[..., h:-h, h:-h], tgt[..., h:-h, h:-h]
        loss = F.mse_loss(pred, tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drift.parameters(), args.grad_clip)
        opt.step()
        if step == n_steps - 1 and outer == 0:
            # Reset EMA after warm-up, before IMF. Warm-up-only runs (Ising)
            # therefore save the final weights as "ema", regardless of --ema-decay.
            ema.module.load_state_dict(drift.state_dict())
        ema.update_parameters(drift)
        losses[step] = loss.detach()
        if step % 500 == 0:
            print(f"  {direction} outer {outer} step {step}: "
                  f"loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return float(losses.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--data", default=str(WORK / "data" / "ising_windows"),
                   help="directory of <prefix><i>.pt marginals")
    p.add_argument("--prefix", default="ising_t",
                   help="marginal filename prefix: ising_t (Ising anchors), "
                        "class_t (MNIST digit loop) or hearts_t (heart stages)")
    p.add_argument("--marginals", type=int, default=6)
    p.add_argument("--sigma", type=float, default=2.4,
                   help="bridge noise. The decisive knob: too small freezes "
                        "the transport, too large phase-collapses the longest "
                        "bridge")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=50000)
    p.add_argument("--inner-steps", type=int, default=12500)
    p.add_argument("--outer-iters", type=int, default=3)
    p.add_argument("--sim-steps", type=int, default=600,
                   help="SDE steps across the whole chain for the IMF coupling "
                        "refresh, split evenly over the bridges")
    p.add_argument("--blocks", default="32,64,128")
    p.add_argument("--layers-per-block", type=int, default=2)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--symmetrise", default="", choices=["", "z2", "rot180", "z2rot180"])
    p.add_argument("--circular", action="store_true",
                   help="circular padding in every convolution (border-blind "
                        "drift; see net.circularise). Runs without it paint a "
                        "domain wall along the periodic wrap when rolled on a "
                        "full field. Recorded "
                        "in config.json; loaders rebuild with the same flag.")
    p.add_argument("--out-dir", default=str(WORK / "runs"))
    p.add_argument("--halo", type=int, default=0,
                   help="score the bridge loss only on the central (side - 2*halo)^2 "
                        "box; the ring is context only. 0 scores the whole window.")
    p.add_argument("--only-direction", choices=["forward", "backward"], default="",
                   help="warm-up only: train just this direction's outer-0 drift and "
                        "save its ckpt_0000_<direction>.pt. Outer 0 couples the "
                        "marginals independently, so the two warm-up directions share "
                        "nothing and can run as two parallel jobs whose checkpoints "
                        "are put in one run directory afterwards. Requires "
                        "--outer-iters 1.")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.warmup_steps, args.inner_steps, args.outer_iters = 30, 20, (1 if args.only_direction else 2)
        args.sim_steps, args.batch = 20, 8
    if args.only_direction and args.outer_iters != 1:
        raise SystemExit("--only-direction splits the independent warm-up; it needs --outer-iters 1")
    args.data = str(resolve(args.data))    # config.json records it

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    times = list(range(args.marginals))
    bridges = list(zip(times[:-1], times[1:]))
    n_sim = round(args.sim_steps / len(bridges))
    marginals = load_marginals(Path(args.data), args.marginals, device, args.prefix)
    side = marginals[0].shape[-1]
    blocks = tuple(int(x) for x in args.blocks.split(","))
    drifts, emas, opts = {}, {}, {}
    for d in ("forward", "backward"):
        drifts[d] = build_drift(side, blocks, args.layers_per_block,
                                channels=marginals[0].shape[1],
                                symmetrise=args.symmetrise,
                                circular=args.circular).to(device)
        emas[d] = torch.optim.swa_utils.AveragedModel(
            drifts[d], avg_fn=lambda a, q, n: args.ema_decay * a
            + (1 - args.ema_decay) * q).eval()
        opts[d] = torch.optim.AdamW(drifts[d].parameters(), lr=args.lr,
                                    betas=(0.9, 0.999), weight_decay=0.01)
    n_par = sum(q.numel() for q in drifts["forward"].parameters())
    print(f"drift params: {n_par / 1e6:.2f}M x2  device={device}  "
          f"times {times}  bridges {bridges}", flush=True)

    out_root = Path(args.out_dir) / args.run_name
    (out_root / "ckpts").mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps(
        dict(vars(args), g_arch="mmtsbm", times=times, bridges=bridges,
             lat=side, drift_params=n_par,
             steps_per_bridge=[n_sim] * len(bridges)),
        indent=1, default=str))
    t_start = time.time()

    history = []
    for outer in range(args.outer_iters):
        for direction in ("backward", "forward"):
            if args.only_direction and direction != args.only_direction:
                continue
            n_steps = args.warmup_steps if outer == 0 else args.inner_steps
            pairs = couple(marginals, bridges, emas, args.sigma, direction,
                           outer, n_sim)
            loss = train_one_direction(
                drifts[direction], emas[direction], opts[direction], pairs,
                bridges, direction, n_steps, args, outer)
            del pairs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            torch.save({"outer_iter": outer, "direction": direction,
                        "ema": emas[direction].module.state_dict(),
                        "net": drifts[direction].state_dict(),
                        "args": dict(vars(args), times=times, lat=side)},
                       out_root / "ckpts" / f"ckpt_{outer:04d}_{direction}.pt")
            history.append({"outer": outer, "direction": direction,
                            "steps": n_steps, "mean_loss": loss,
                            "elapsed_s": time.time() - t_start})
            (out_root / "history.json").write_text(json.dumps(history, indent=1))
            print(f"[outer {outer}] {direction}: mean loss {loss:.4f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    print(f"done in {time.time() - t_start:.0f}s -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
