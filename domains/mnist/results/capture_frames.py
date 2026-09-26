"""Roll a trained REGIS or control run for 1000 digit transitions and keep the frames.

5000 trajectories, 500 from each start digit, each started from a real test
digit with its own persistent noise ξ (`z` in the code) ~ N(0, I), fixed along
the trajectory. Frames are kept unclamped so evaluation can detect values
outside [0, 1]. Digit labels are the frozen classifier's argmax.

The capture file is the shared format read by the tables and figures; the baselines'
captures carry the same keys:

    keep      (H,)               transitions at which frames are kept
    start     (N,)               start digit of each trajectory
    frames_h  (H, N, 32, 32)     every trajectory at every kept transition
    pred      (H, N)             the classifier's digit
    vmax      (H,)               max |pixel| over the population (see fd_tables.DIVERGED)
    grid      (H, 10, 4, 32, 32) four frames per start digit, for the sample figure
                                 (the baselines' captures: the first four classified as each digit)

  python -m domains.mnist.results.capture_frames --run mnist_regis_seed0
"""
import argparse

import numpy as np
import torch

from domains.mnist.classifier import load_classifier
from domains.mnist.data import digits_by_class
from domains.mnist.instruments.measure import CHUNK, load_ema_generator, real_starts
from domains.mnist.results import runs

KEEP = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 80, 100, 140, 200, 300, 500, 700, 1000]
NGRID = 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, choices=runs.REGIS + runs.CONTROL)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    clf = load_classifier(dev)
    val = digits_by_class(False, dev)
    G, cfg = load_ema_generator(runs.checkpoint(a.run), dev)
    updates, C, H = cfg["max_nca_steps"], cfg["channel_n"], len(KEEP)
    start, x0 = real_starts(val, C, dev)
    NT = len(start)

    pred = np.zeros((H, NT), np.int16)
    vmax = np.zeros(H)
    grid = np.zeros((H, 10, NGRID, 32, 32), np.float16)
    frames_h = np.zeros((H, NT, 32, 32), np.float16)
    for c0 in range(0, NT, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        x = x0[sl]
        torch.manual_seed(1000 + c0)
        z = torch.randn(x.shape[0], cfg["z_dim"], device=dev)
        with torch.no_grad():
            for transition in range(1, max(KEEP) + 1):
                for _ in range(updates):
                    x = G.step(x, z=z)
                if transition not in KEEP:
                    continue
                vis = x[:, -1:].float()
                k = KEEP.index(transition)
                pred[k, sl] = clf(vis).argmax(1).cpu().numpy()
                vmax[k] = max(vmax[k], float(vis.abs().max()))
                frames_h[k, sl] = vis[:, 0].cpu().numpy()
                for si in range(10):
                    idx = np.where(start[sl] == si)[0][:NGRID]
                    if len(idx):
                        grid[k, si, :len(idx)] = vis[idx, 0].cpu().numpy()
        print(f"{a.run}: trajectories {c0}..{c0 + x.shape[0] - 1} done, max|pixel| at 1000 so far {vmax[-1]:.2f}", flush=True)

    out = runs.capture(a.run)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keep=np.array(KEEP), start=start, pred=pred, grid=grid, vmax=vmax, frames_h=frames_h)
    print(f"-> {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
