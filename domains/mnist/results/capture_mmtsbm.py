"""Capture 1000 digit transitions from the final MMtSBM forward EMA drift.

Uses 500 distinct test images per starting digit and the configured
Euler-Maruyama step count. Each trajectory starts at its digit's bridge;
the sampler seed is the training seed. Sampling uses torch.compile, TF32,
and cuDNN autotuning.

The file format is defined in capture_frames.py. Here, classification uses
frames clipped to [0, 1], and grid stores the first four trajectories
classified as each digit. Stored frames remain unclamped."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from workspace import WORK
from baselines.mmtsbm.net import build_drift
from domains.mnist.classifier import load_classifier
from domains.mnist.results import runs
from domains.mnist.results.capture_frames import KEEP

PER, NGRID = 500, 4
CHUNK = 2500        # trajectories integrated together; the sampler noise depends on it


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, choices=runs.MMTSBM)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    # Match the TF32 and autotuning settings used for the paper captures.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = WORK / "runs" / a.run
    cfg = json.loads((run / "config.json").read_text())
    drift = build_drift(cfg["lat"], tuple(int(x) for x in cfg["blocks"].split(",")), cfg["layers_per_block"])
    drift.load_state_dict(torch.load(sorted(run.glob("ckpts/ckpt_*_forward.pt"))[-1], map_location="cpu",
                                     weights_only=False)["ema"])
    drift = torch.compile(drift.to(dev).eval())
    clf = load_classifier(dev)
    n_sub, sigma, H = cfg["steps_per_bridge"][0], cfg["sigma"], len(KEEP)
    ds = 1.0 / n_sub

    ev = [torch.load(Path(cfg["data"]) / f"eval_t{i}.pt") for i in range(11)]
    pools = [torch.cat([ev[0], ev[10]])] + ev[1:10]
    rng = np.random.default_rng(cfg["seed"])
    x0 = torch.cat([q[rng.choice(len(q), PER, replace=False)] for q in pools])
    start = np.repeat(np.arange(10), PER)
    torch.manual_seed(cfg["seed"])

    pred = np.zeros((H, len(start)), np.int16)
    vmax = np.zeros(H, np.float32)
    frames_h = np.zeros((H, len(start), 32, 32), np.float16)
    for c0 in range(0, len(start), CHUNK):
        sl = slice(c0, c0 + CHUNK)
        x, b = x0[sl].to(dev), torch.from_numpy(start[sl]).to(dev)
        for transition in range(1, max(KEEP) + 1):
            for i in range(n_sub):
                t = (b.float() + i * ds).view(-1, 1)
                x = x + drift(x, t) * ds + sigma * ds ** 0.5 * torch.randn_like(x)
            b = (b + 1) % 10
            if transition not in KEEP:
                continue
            k, vis = KEEP.index(transition), (x + 1) / 2
            frames_h[k, sl] = vis[:, 0].half().cpu().numpy()
            pred[k, sl] = clf(vis.clamp(0, 1)).argmax(1).cpu().numpy()
            vmax[k] = max(vmax[k], float(vis.abs().max()))
        print(f"{a.run}: trajectories {c0}..{c0 + len(x) - 1} done, max|pixel| at 1000 so far {vmax[-1]:.2f}", flush=True)

    grid = np.zeros((H, 10, NGRID, 32, 32), np.float16)
    for k in range(H):
        for d in range(10):
            rows = np.where(pred[k] == d)[0][:NGRID]
            grid[k, d, :len(rows)] = frames_h[k, rows]
    out = runs.capture(a.run)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keep=np.array(KEEP), start=start, pred=pred, grid=grid, vmax=vmax, frames_h=frames_h)
    print(f"-> {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
