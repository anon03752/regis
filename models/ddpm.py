"""State-conditioned DDPM baseline trained on paired Ising snapshots.

The model predicts a state at t + delta from the state at t and log(delta).
It receives no absolute Ising time, so sampled jumps can be chained beyond
the training window. The v-prediction target is a*eps - s*x0; denoiser inputs
are [x_noisy, sqrt(alpha_bar), log-delta, start state], with scalar inputs
broadcast as planes and spin fields represented on the [-1, 1] scale.

The custom U-Net uses circular padding and spatial conditioning planes,
allowing training on 192x192 windows and sampling on 512x512 fields.

    python -m models.ddpm --run-name ddpm_pairs15000_seed0 --seed 0 --compile
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 1000            # diffusion steps
DELTA_MAX = 1000    # the longest trained jump (data.pairs draws delta < 1000);
                    # the log-delta plane is ln(delta) / ln(DELTA_MAX)
LR, EMA_DECAY = 1e-4, 0.999


def _c3(cin, cout):
    return nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular")


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.c2 = _c3(cin, cout), _c3(cout, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.c1(x))
        h = F.silu(self.c2(h))
        return h + self.skip(x)


class DenoiserUNet(nn.Module):
    """Four-level residual U-Net, widths base x (1, 2, 3, 4): encoder e, d, f,
    g, bottleneck b, decoder ug, uf, u. Average-pool down, periodic bilinear
    up, zero-initialised head."""

    def __init__(self, base=48):
        super().__init__()
        self.e1 = Block(4, base)
        self.e2 = Block(base, base)
        self.d1 = Block(base, base * 2)
        self.d2 = Block(base * 2, base * 2)
        self.f1 = Block(base * 2, base * 3)
        self.f2 = Block(base * 3, base * 3)
        self.g1 = Block(base * 3, base * 4)
        self.g2 = Block(base * 4, base * 4)
        self.b1 = Block(base * 4, base * 4)
        self.ug1 = Block(base * 7, base * 3)
        self.ug2 = Block(base * 3, base * 3)
        self.uf1 = Block(base * 5, base * 2)
        self.uf2 = Block(base * 2, base * 2)
        self.u1 = Block(base * 3, base)
        self.u2 = Block(base, base)
        self.head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _up(self, x):
        # bilinear on the torus: border pixels blend with the wrapped
        # neighbour instead of a clamped copy of the edge
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        return x[..., 2:-2, 2:-2]

    def forward(self, x):
        e = self.e2(self.e1(x))
        d = self.d2(self.d1(F.avg_pool2d(e, 2)))
        f = self.f2(self.f1(F.avg_pool2d(d, 2)))
        g = self.b1(self.g2(self.g1(F.avg_pool2d(f, 2))))
        ug = self.ug2(self.ug1(torch.cat([self._up(g), f], 1)))
        uf = self.uf2(self.uf1(torch.cat([self._up(ug), d], 1)))
        u = self.u2(self.u1(torch.cat([self._up(uf), e], 1)))
        return self.head(u)


def cosine_alpha_bar():
    """Nichol-Dhariwal cosine schedule: alpha_bar, shape (T,)."""
    t = torch.linspace(0, 1, T + 1)
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 1.0)[1:]


@torch.no_grad()
def sample(net, start, delta, gen, steps=250):
    """One jump of `delta` sweeps from `start` ((B, 1, H, W) in {-1, +1}):
    ancestral sampling on `steps` evenly respaced diffusion steps. Returns the
    continuous sample in [-1, 1]."""
    dev = start.device
    ab = cosine_alpha_bar().to(dev)
    ts = np.unique(np.linspace(T - 1, 0, min(steps, T)).astype(int))[::-1]
    logd = torch.full_like(start, float(np.log(delta) / np.log(DELTA_MAX)))
    x = torch.randn(start.shape, generator=gen, device=dev)
    for i, t in enumerate(ts):
        a, s = ab[t].sqrt(), (1 - ab[t]).sqrt()
        v = net(torch.cat([x, a.expand_as(x), logd, start], 1))
        # x0 via eps (not a*x - s*v): the float32 arithmetic the paper caches were sampled with
        eps = s * x + a * v
        x0 = ((x - s * eps) / a).clamp(-1, 1)
        if i + 1 == len(ts):
            return x0
        ab_t, ab_n = ab[t], ab[ts[i + 1]]
        beta = (1 - ab_t / ab_n).clamp(1e-8, 0.999)
        mu = (ab_n.sqrt() * beta * x0 + (ab_t / ab_n).sqrt() * (1 - ab_n) * x) / (1 - ab_t)
        var = beta * (1 - ab_n) / (1 - ab_t)
        x = mu + var.sqrt() * torch.randn(x.shape, generator=gen, device=dev)


def main():
    from workspace import WORK, resolve

    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--data", default=str(WORK / "data" / "ddpm_pairs_15000.npz"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--crop", type=int, default=128, help="side of the region the loss sees")
    p.add_argument("--halo", type=int, default=32,
                   help="context ring around the crop: the net sees (crop + 2 halo)^2, "
                        "so the window's own edges are never a training target")
    p.add_argument("--base", type=int, default=48, help="U-Net width")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(2000 + args.seed)
    W, LO, HI = args.crop + 2 * args.halo, args.halo, args.halo + args.crop

    # every pair contributes its fixed top-left window: no random crops, no augmentation
    d = np.load(resolve(args.data))
    starts, targets = d["starts"][:, :W, :W], d["targets"][:, :W, :W]
    logd = np.log(d["delta"].astype(np.float64)) / np.log(float(DELTA_MAX))

    net = DenoiserUNet(args.base).to(device)
    n_par = sum(q.numel() for q in net.parameters())
    print(f"{len(starts)} pairs, denoiser {n_par / 1e6:.2f}M params, {device}", flush=True)
    # compile wraps the forward only, so optimizer, EMA and checkpoint keys
    # stay those of the raw module
    fnet = torch.compile(net) if args.compile else net
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    ema = torch.optim.swa_utils.AveragedModel(
        net, avg_fn=lambda a, q, n: EMA_DECAY * a + (1 - EMA_DECAY) * q)
    ab = cosine_alpha_bar().to(device)

    out = WORK / "runs" / args.run_name
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        dict(vars(args), lr=LR, ema_decay=EMA_DECAY, out_dir=str(out.parent),
             denoiser_params=n_par, n_pairs=len(starts)), indent=1))
    t0 = time.time()
    for step in range(args.total_steps):
        pi = rng.integers(0, len(starts), args.batch)
        x = torch.from_numpy(targets[pi]).to(device)[:, None].float() * 2.0 - 1.0
        c = torch.from_numpy(starts[pi]).to(device)[:, None].float() * 2.0 - 1.0
        t = torch.randint(0, T, (args.batch,), device=device)
        a = ab[t].sqrt().view(-1, 1, 1, 1)
        s = (1 - ab[t]).sqrt().view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        x_t = a * x + s * eps
        dplane = torch.as_tensor(logd[pi], dtype=torch.float32,
                                 device=device).view(-1, 1, 1, 1).expand(-1, 1, W, W)
        v = fnet(torch.cat([x_t, a.expand(-1, 1, W, W), dplane, c], 1))
        loss = F.mse_loss(v[..., LO:HI, LO:HI], (a * eps - s * x)[..., LO:HI, LO:HI])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update_parameters(net)
        if step % 1000 == 0:
            print(f"step {step}: loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"ema": ema.state_dict(), "args": {"hparams": vars(args)}},
               out / "ckpts" / f"ckpt_{args.total_steps:06d}.pt")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
