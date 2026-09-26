"""Cache DDPM rollouts from shared initial fields.

The Glauber engine first evolves each field to t = 3. Sampled DDPM jumps
then follow the requested grid, with jumps in the trained range 10..1000.
Chunk j uses engine seed 8801 + 1000*j. See rollouts.py for the cache format
and docs/reproduce.md for the separate chains used by the filmstrip."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from domains.ising.data import JUMP_MIN
from domains.ising.glauber import sweep_
from domains.ising.instruments.rollouts import chunks, quench, save
from models.ddpm import DELTA_MAX, DenoiserUNet, sample
from workspace import WORK

START = 3
# every table read-out from 150 on (150, 300, 700, 1000, 2000, 4000) plus
# the curve points, each hop inside the trained jump range 10..1000; the
# first hop lands on 13, so the DDPM has no t = 10 or 17 cell
DDPM_TIMES = [13, 24, 42, 55, 74, 100, 150, 175, 231, 300, 408, 550, 700, 961,
              1000, 1278, 1700, 2000, 2261, 3007, 4000]
BATCH = 8          # fields sampled together; each batch has its own seeded generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name under runs/")
    ap.add_argument("--label", default=None, help="cache stem (default: the run name)")
    ap.add_argument("--ns", type=int, default=512, help="number of fields")
    ap.add_argument("--lat", type=int, default=512)
    ap.add_argument("--grid", default=None, help="comma list of read-out times (default: DDPM_TIMES)")
    ap.add_argument("--sample-steps", type=int, default=250)
    ap.add_argument("--raw", action="store_true",
                    help="also store the un-thresholded samples, float16 in [0, 1]")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = [int(k) for k in args.grid.split(",")] if args.grid else DDPM_TIMES
    hops = list(zip([START] + grid[:-1], grid))
    for k0, k1 in hops:
        assert JUMP_MIN <= k1 - k0 <= DELTA_MAX, f"hop {k0} -> {k1} outside the trained jump range"

    ck = torch.load(sorted((WORK / "runs" / args.run / "ckpts").glob("*.pt"))[-1],
                    map_location=dev, weights_only=False)
    net = DenoiserUNet(ck["args"]["hparams"]["base"]).to(dev).eval()
    net.load_state_dict({k.removeprefix("module."): v for k, v in ck["ema"].items() if k.startswith("module.")})

    t0 = time.time()
    bits, raw = [[] for _ in hops], [[] for _ in hops]
    for j, n in chunks(args.ns):
        state = torch.where(quench(j, n, args.lat, dev), 1.0, -1.0)
        gen = torch.Generator(device=dev).manual_seed(8801 + 1000 * j)
        for _ in range(START):
            sweep_(state, 1.0, gen)
        state = state[:, None]
        for h, (k0, k1) in enumerate(hops):
            seed = 9000 + 17 * h + 100003 * j
            x = torch.cat([sample(net, state[b:b + BATCH], k1 - k0,
                                  torch.Generator(device=dev).manual_seed(seed + b), args.sample_steps)
                           for b in range(0, n, BATCH)])
            state = torch.where(x > 0, 1.0, -1.0)
            bits[h].append(np.packbits((x[:, 0] > 0).cpu().numpy()))
            if args.raw:
                raw[h].append(((x[:, 0] + 1) / 2).clamp(0, 1).half().cpu().numpy())
        print(f"chunk {j}: {n} fields to t = {grid[-1]} [{time.time() - t0:.0f}s]", flush=True)
    save(args.label or args.run, args.lat, args.ns, grid, bits, raw if args.raw else None,
         run=args.run, start=START, sample_steps=args.sample_steps)


if __name__ == "__main__":
    main()
