"""Train REGIS or its non-local control on the digit cycle 0 -> ... -> 9 -> 0.

A replay pool stores states, class labels, ages, and persistent noise vectors.
Each iteration advances a sampled state toward the next class. Only the
discriminator receives the target class; the generator receives the state
and persistent noise. Pool entries older than max_age are replaced with
real digits and fresh noise.

Readouts are sampled from the final readout_window updates. Discriminator
and generator steps use separate stochastic rollouts; the generator readout
returns to the pool. Training uses R3GAN losses and a hidden-state norm
penalty. Evaluation uses the EMA generator.

RECORD contains the paper's configuration. Select --g-arch regis or control.
Commands are in docs/reproduce.md. Outputs are written to
<workspace>/runs/<run-name>/{config.json, events.jsonl, ckpts/ckpt_NNNNNN.pt}.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from workspace import WORK
from models.train.logger import Logger
from domains.mnist.rule import CycleNCA
from domains.mnist.control import CycleControl
from models.train.r3gan import Discriminator, relativistic
from domains.mnist.fast_gp import discriminator_loss
from domains.mnist.classifier import SIDE
from domains.mnist.data import digits_by_class

# Paper training configuration, saved in config.json and checkpoints.
# Architecture settings are defined in rule.py and control.py.
RECORD = dict(
    channel_n=17,                     # 16 hidden channels + the visible one (last)
    hidden_dim=256,                   # regis: width of the per-cell MLP
    control_base=48,                  # control: channel width
    control_dilations=(1, 2, 5, 9),
    z_dim=16,                         # the persistent noise ξ, fixed along a trajectory; it encodes the style
    fire_rate=0.5,
    cycle_n=10,
    max_nca_steps=30,                 # rollout length; the readout is sampled within its final window
    readout_window=5,                 # the paper's δ: the readout is after one of the last 5 updates
    d_channels=32, d_stages=3, d_blocks=2,   # D: 3 downsampling stages + the final one = the paper's four stages
    batch_size=128, lr_g=2e-4, lr_d=2e-4, beta1=0.0, beta2=0.99,
    gamma=1.0,                        # gradient-penalty weight
    grad_clip=1.0,
    norm_weight=1.0,                  # hidden-norm hinge weight
    r_max=1.0,                        # hidden norm above which the hinge acts
    ema_decay=0.999, ema_warmup=10_000,   # decay = min(ema_decay, (1+n)/(ema_warmup+n))
    pool_size=2048,
    max_age=5,                        # pool age, in transitions: an entry older than this is replaced
    save_every=25_000, log_every=100,
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def sample_digits(digits, labels):
    """One random training digit of each class in `labels`."""
    return torch.stack([digits[c][torch.randint(0, len(digits[c]), (1,)).item()] for c in labels])


class CyclePool:
    """Pool entries: state, current class, age, z (the persistent noise ξ). An
    entry that has made more than `max_age` transitions is replaced by a fresh
    real digit."""

    def __init__(self, size, digits, C, z_dim, cycle_n, max_age):
        dev = digits[0].device
        self.digits, self.cycle_n, self.max_age = digits, cycle_n, max_age
        self.states = torch.zeros((size, C, SIDE, SIDE), device=dev)
        self.labels = torch.zeros(size, dtype=torch.long, device=dev)
        self.age = torch.full((size,), max_age + 1, dtype=torch.long, device=dev)    # all expired: refill() fills it
        self.z = torch.zeros((size, z_dim), device=dev)
        self.refill()

    def sample(self, n):
        i = torch.randperm(len(self.states), device=self.states.device)[:n]
        return i, self.states[i], self.labels[i], self.z[i]

    def put_back(self, i, states, labels):
        """The readout states return with their new class and age + 1; entries
        past max_age are replaced."""
        self.states[i] = states.detach()
        self.labels[i] = labels
        self.age[i] += 1
        self.refill()

    def refill(self):
        mask = self.age > self.max_age
        n = mask.sum().item()
        if n == 0:
            return
        # Assign classes in order when exactly cycle_n entries expire. Preserve
        # this branch to keep the paper runs' NumPy random sequence.
        labels = np.arange(self.cycle_n) if n == self.cycle_n else np.random.choice(self.cycle_n, n)
        img = sample_digits(self.digits, labels)
        self.states[mask] = 0
        self.states[mask, -1:] = img
        self.labels[mask] = torch.as_tensor(labels, device=self.labels.device)
        self.age[mask] = 0
        self.z[mask] = torch.randn(n, self.z.shape[1], device=self.z.device)


def rollout(G, state, z, n):
    """The states after each of n updates from `state`."""
    states = []
    for _ in range(n):
        state = G.step(state, z)
        states.append(state)
    return states


def save_checkpoint(run_dir, step, G, D, args, ema_G):
    a = dict(vars(args), hparams=G.hparams)          # the model's own record of how it was built
    out = run_dir / "ckpts"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "G": G.state_dict(), "D": D.state_dict(), "args": a, "ema_G": ema_G.state_dict()},
               out / f"ckpt_{step:06d}.pt")
    print(f"checkpoint at step {step}", flush=True)


def train(G, D, args, device, run_dir):
    digits = digits_by_class(True, device)
    pool = CyclePool(args.pool_size, digits, args.channel_n, args.z_dim, args.cycle_n, args.max_age)

    def _ema_avg(avg, model_p, num):
        decay = min(args.ema_decay, (1 + num) / (args.ema_warmup + num))
        return avg + (1 - decay) * (model_p - avg)
    ema_G = torch.optim.swa_utils.AveragedModel(G, avg_fn=_ema_avg)
    G.step = torch.compile(G.step)

    fused = torch.cuda.is_available()
    G_opt = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2), fused=fused)
    D_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2), fused=fused)

    updates = args.max_nca_steps          # one transition
    batch_arange = torch.arange(args.batch_size, device=device)
    logger = Logger(run_dir=run_dir, config=vars(args))

    for step in range(args.iterations):
        batch_indices, pool_states, pool_labels, batch_z = pool.sample(args.batch_size)
        target_labels = (pool_labels + 1) % args.cycle_n
        real_img = sample_digits(digits, target_labels.tolist())
        # the readout time of each trajectory: after one of the last readout_window updates
        readout_idx = torch.randint(0, args.readout_window, (args.batch_size,), device=device)

        # Discriminator update from a gradient-free rollout.
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            states = rollout(G, pool_states, batch_z, updates)
        fake_batch = torch.stack(states[-args.readout_window:])[readout_idx, batch_arange]
        D.requires_grad_(True)
        D_opt.zero_grad(set_to_none=True)
        # fast_gp computes the R1/R2 parameter gradients using forward-mode AD.
        d_loss, adv_d, gp_real, gp_fake = discriminator_loss(D, real_img, G.readout(fake_batch), target_labels, args.gamma)
        d_loss.backward()
        nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
        D_opt.step()

        # Generator update: rerun the same starts and z with fresh firing masks.
        # Keep intermediate states for the hidden-state penalty.
        D.requires_grad_(False)
        G_opt.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16):
            states = rollout(G, pool_states, batch_z, updates)
        fake_batch = torch.stack(states[-args.readout_window:])[readout_idx, batch_arange]
        d_real = D(real_img, target_labels)
        d_fake = D(G.readout(fake_batch), target_labels)
        g_adv = relativistic(d_fake, d_real)
        # Penalise hidden-channel norms above r_max, averaged over cells and updates.
        hidden_hinge = torch.stack([torch.relu(s[:, :-1].norm(dim=1) - args.r_max).pow(2) for s in states]).mean()
        g_loss = g_adv + args.norm_weight * hidden_hinge
        g_loss.backward()
        nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
        G_opt.step()
        ema_G.update_parameters(G)

        # Return the readout state to the pool with its new class.
        pool.put_back(batch_indices, fake_batch, target_labels)

        if step % args.log_every == 0:
            d_reg = 0.5 * args.gamma * (gp_real.mean() + gp_fake.mean())
            logger.scalars({"loss/D_adv": adv_d.item(), "loss/D_reg": float(d_reg), "loss/G_adv": g_adv.item(),
                            "loss/hidden_hinge": hidden_hinge.item(),
                            "penalty/gp_real": gp_real.mean().item(), "penalty/gp_fake": gp_fake.mean().item()}, step=step)
        if step % (10 * args.log_every) == 0:
            print(f"step {step:>6d}  D {adv_d.item():.3f}  G {g_adv.item():.3f}  hinge {hidden_hinge.item():.4f}", flush=True)
        if (step + 1) % args.save_every == 0:
            save_checkpoint(run_dir, step + 1, G, D, args, ema_G)
    if (step + 1) % args.save_every != 0:
        save_checkpoint(run_dir, step + 1, G, D, args, ema_G)
    logger.close(status="completed")


def parse_args():
    p = argparse.ArgumentParser(description="Train REGIS or its non-local control on cycling MNIST.")
    p.add_argument("--g-arch", choices=["regis", "control"], default="regis",
                   help="regis: the local rule (3x3 perception). control: four dilated 3x3 convolutions, "
                        "receptive field 35 px per update, everything else identical")
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iterations", type=int, default=100_000, help="training iterations (shorten for a smoke run)")
    return argparse.Namespace(**RECORD, **vars(p.parse_args()))


def build_generator(args):
    if args.g_arch == "control":
        return CycleControl(channel_n=args.channel_n, base=args.control_base, z_dim=args.z_dim,
                            fire_rate=args.fire_rate, flat_dilations=args.control_dilations)
    return CycleNCA(channel_n=args.channel_n, hidden_dim=args.hidden_dim, z_dim=args.z_dim, fire_rate=args.fire_rate)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = build_generator(args).to(device)
    widths = [args.d_channels] * (args.d_stages + 1)
    D = Discriminator(widths=widths, cardinalities=[4] * len(widths), blocks_per_stage=[args.d_blocks] * len(widths),
                      expansion=2, num_classes=args.cycle_n, embed_dim=32, in_channels=1).to(device)
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"{args.run_name} on {device}: {args.g_arch}, G {n(G):,} parameters, D {n(D):,}; "
          f"{args.cycle_n}-digit cycle, {args.max_nca_steps} updates per transition", flush=True)
    train(G, D, args, device, WORK / "runs" / args.run_name)


if __name__ == "__main__":
    main()
