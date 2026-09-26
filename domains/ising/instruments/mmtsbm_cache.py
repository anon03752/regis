"""Cache MMtSBM rollouts from shared initial fields.

Each bridge starts at the previous endpoint and uses the forward EMA drift.
For warm-up-only Ising runs, these are the final training weights.
Intermediate readouts are solver states at the nearest bridge step, with
bridge time mapped linearly between observation times. Chunk j uses sampler
seed j. See rollouts.py for the cache format."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from baselines.mmtsbm.bridge import simulate
from baselines.mmtsbm.net import build_drift
from domains.ising.data import TIMES
from domains.ising.instruments.rollouts import between_marginals, chunks, quench, save
from workspace import WORK

BATCH = 32         # fields integrated together; the sampler noise depends on it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name under runs/")
    ap.add_argument("--label", default=None, help="cache stem (default: the run name)")
    ap.add_argument("--ns", type=int, default=512, help="number of fields")
    ap.add_argument("--lat", type=int, default=512)
    ap.add_argument("--steps-per-bridge", type=int, default=120, help="Euler-Maruyama steps")
    ap.add_argument("--raw", action="store_true",
                    help="also store the un-thresholded states, float16 in [0, 1]")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = WORK / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text())
    marginal_times = [0] + TIMES   # t = 0 is the iid quench
    bridges = cfg["bridges"]
    drift = build_drift(cfg["lat"], tuple(int(x) for x in cfg["blocks"].split(",")),
                        cfg["layers_per_block"], symmetrise=cfg["symmetrise"],
                        circular=cfg["circular"])
    ck = torch.load(sorted(run.glob("ckpts/ckpt_*_forward.pt"))[-1], map_location="cpu",
                    weights_only=False)
    drift.load_state_dict(ck["ema"])
    drift = drift.to(dev).eval()

    inside, ks = between_marginals(marginal_times, args.steps_per_bridge)

    t0 = time.time()
    bits, raw = [[] for _ in ks], [[] for _ in ks]
    for j, n in chunks(args.ns):
        x0 = torch.where(quench(j, n, args.lat, dev), 1.0, -1.0)[:, None]
        torch.manual_seed(j)
        for b0 in range(0, n, BATCH):
            z, got = x0[b0:b0 + BATCH], {}
            for (a, b), at, k_end in zip(bridges, inside, marginal_times[1:]):
                def keep(i, s, at=at):
                    if i in at:
                        got[at[i]] = s[:, 0].cpu()
                z = simulate(drift, z, a, b, cfg["sigma"], "forward", args.steps_per_bridge, on_step=keep)
                got[k_end] = z[:, 0].cpu()
            for f, k in enumerate(ks):
                bits[f].append(np.packbits((got[k] > 0).numpy()))
                if args.raw:
                    raw[f].append(((got[k] + 1) / 2).clamp(0, 1).half().numpy())
        print(f"chunk {j}: {n} fields to t = {marginal_times[-1]} [{time.time() - t0:.0f}s]", flush=True)
    save(args.label or args.run, args.lat, args.ns, ks, bits, raw if args.raw else None,
         run=args.run, anchors=marginal_times, steps_per_bridge=[args.steps_per_bridge] * len(bridges))


if __name__ == "__main__":
    main()
