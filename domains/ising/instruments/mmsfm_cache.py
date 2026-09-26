"""Cache MMSFM rollouts using SRK integration from shared initial fields.

Uses 160 steps per transition, float64 states, compiled heads, bf16 SDPA,
and TF32 matmuls. Intermediate readouts use the nearest solver step, with
solver time mapped linearly between observation times. Brownian-path seeds
depend on the training seed, chunk, batch offset, and transition.

This script enables deterministic algorithms. The original paper caches
used autotuning, so their numerical trajectories may differ. See
rollouts.py for the cache format and docs/reproduce.md for commands."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from baselines.mmsfm.net import Drift, build_heads, configure_execution, sdpa_attention, upstream
from domains.ising.data import TIMES
from domains.ising.instruments.rollouts import between_marginals, chunks, quench, save
from workspace import WORK

BATCH = 2     # fields integrated together; the Brownian paths depend on it
STEPS = 160   # SRK steps per transition


class TransitionSDE:
    """One transition for torchsde in [0, 1] occupation coordinates: the heads take the spins
    x*2-1, so their drifts and the noise sigma are halved"""
    noise_type, sde_type = "additive", "ito"

    def __init__(self, drift, phase, shape, sigma):
        self.drift, self.phase, self.shape, self.sigma = drift, phase, shape, sigma
        self.memo = {}   # torchsde's SRK evaluates f again on earlier stages' states

    def f(self, t, y):
        """flow + score heads at the transition's phase + t"""
        # Retain y to prevent Python from reusing its id while the cache entry exists.
        key = (float(t), id(y))
        if key not in self.memo:
            self.memo[key] = y, self.drift(t, y.reshape(self.shape), self.phase).flatten(1)
            if len(self.memo) > 8:
                self.memo.pop(next(iter(self.memo)))
        return self.memo[key][1]

    def g_prod(self, t, y, v):
        return self.sigma / 2 * v


def load_heads(run, dev, tr):
    """Load and compile the final flow and score heads; return them and the run config."""
    cfg = json.loads((run / "config.json").read_text())
    heads = build_heads(tr, tuple(cfg["dims"]), cfg["channels"], cfg["attention_resolutions"], dev)
    w = torch.load(sorted(run.glob("step_*.pt"))[-1], map_location="cpu", weights_only=True)
    for m, key in zip(heads, ["model_state_dict", "score_model_state_dict"]):
        m.load_state_dict(w[key], strict=True)
        m.eval().requires_grad_(False)
    return [torch.compile(m, dynamic=False) for m in heads], cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory under runs/ (holds config.json and step_*.pt; the last is read)")
    ap.add_argument("--label", default=None, help="cache stem (default: the run name)")
    ap.add_argument("--ns", type=int, default=512, help="number of fields")
    ap.add_argument("--lat", type=int, default=512)
    ap.add_argument("--raw", action="store_true", help="also store the un-thresholded states, float16")
    args = ap.parse_args()

    tr = upstream()
    import torchsde
    from torchcfm.models.unet import unet

    configure_execution()
    unet.QKVAttentionLegacy.forward = sdpa_attention(torch.bfloat16)
    # cuDNN and inductor otherwise pick kernels by timing, and the picks change the rounding
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # required by deterministic mode
    torch.use_deterministic_algorithms(True)
    dev = "cuda"
    run = WORK / "runs" / args.run
    heads, cfg = load_heads(run, dev, tr)
    drift = Drift(*heads)
    marginal_times = [0] + TIMES   # t = 0 is the iid quench
    transition_len = 1.0 / (len(marginal_times) - 1)
    dt = transition_len / STEPS

    inside, ks = between_marginals(marginal_times, STEPS)

    t0 = time.time()
    bits, raw = [[] for _ in ks], [[] for _ in ks]
    with torch.inference_mode():
        for j, n in chunks(args.ns):
            x0 = quench(j, n, args.lat, dev).to(torch.float64)[:, None]
            for lo in range(0, n, BATCH):
                x, got = x0[lo:lo + BATCH], {}
                for transition, at in enumerate(inside):
                    steps = sorted(at)
                    ts = torch.tensor([0.0] + [i * dt for i in steps] + [transition_len], dtype=x.dtype, device=dev)
                    bm = torchsde.BrownianInterval(
                        t0=0., t1=transition_len, size=x.flatten(1).shape, dtype=x.dtype, device=dev,
                        entropy=20269500 + 100000 * cfg["seed"] + 1000 * lo + transition + 10_000_000 * j,
                        levy_area_approximation="space-time")
                    phase = torch.full((len(x),), transition * transition_len, dtype=x.dtype, device=dev)
                    sde = TransitionSDE(drift, phase, x.shape, cfg["sigma"])
                    out = torchsde.sdeint(sde, x.flatten(1), ts, bm=bm, method="srk", dt=dt, adaptive=False)
                    for i, state in zip(steps, out[1:-1]):
                        got[at[i]] = state.reshape_as(x)[:, 0].cpu()
                    x = out[-1].reshape_as(x)
                    assert torch.isfinite(x).all(), f"non-finite state after transition {transition}"
                    got[marginal_times[transition + 1]] = x[:, 0].cpu()
                for f, k in enumerate(ks):
                    bits[f].append(np.packbits((got[k] > 0.5).numpy()))
                    if args.raw:
                        raw[f].append(got[k].half().numpy())
                print(f"chunk {j}, fields {lo}..{lo + len(x) - 1} [{time.time() - t0:.0f}s]", flush=True)
    save(args.label or args.run, args.lat, args.ns, ks, bits, raw if args.raw else None,
         run=args.run, anchors=marginal_times, steps=STEPS)


if __name__ == "__main__":
    main()
