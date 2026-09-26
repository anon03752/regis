"""Compute MNIST Frechet distances and bootstrap intervals.

Distances use 128-dimensional classifier features from 5000 generated and
5000 real test images. Per-seed intervals bootstrap the generated samples
with the real reference held fixed. The real-vs-real floor uses 400 draws
of two disjoint 5000-image samples. fd_tables.py aggregates seeds for the
main table. Writes eval/mnist/fd_ci.json."""
import argparse
import json

import numpy as np
import torch

from workspace import WORK
from domains.mnist.classifier import load_classifier
from domains.mnist.data import digits_by_class
from domains.mnist.results import runs

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = 5000
HORIZONS = [1, 5, 10, 100, 1000]
BOOT, CHUNK, FLOOR_DRAWS = 1000, 25, 400      # bootstrap resamples, resamples per GPU batch, floor draws
# Keep the run order fixed because all bootstraps share one RNG.
ORDER = [s for pair in zip(runs.REGIS, runs.CONTROL) for s in pair] + runs.MMTSBM + runs.MMSFM


class Ref:
    def __init__(self, feat):
        f = torch.as_tensor(np.asarray(feat, np.float64), device=DEV)
        self.mu, cov = f.mean(0), torch.cov(f.T)
        w, Q = torch.linalg.eigh(cov)
        self.sqrt = Q @ torch.diag(w.clamp_min(0).sqrt()) @ Q.T
        self.tr = torch.trace(cov)

    def fid(self, feat):
        f = torch.as_tensor(np.asarray(feat, np.float64), device=DEV)
        mu, cov = f.mean(0), torch.cov(f.T)
        d = mu - self.mu
        w = torch.linalg.eigvalsh(self.sqrt @ cov @ self.sqrt)
        return float(d @ d + torch.trace(cov) + self.tr - 2 * w.clamp_min(0).sqrt().sum())


def feature_fn():
    """images (n, 32, 32), unclamped -> features (n, 128)"""
    fe = load_classifier(DEV).features

    @torch.no_grad()
    def f(img):
        return np.concatenate([fe(torch.as_tensor(img[i:i + 512], dtype=torch.float32, device=DEV)[:, None]).cpu().numpy()
                               for i in range(0, len(img), 512)])
    return f


def real_features(f):
    """Features of the 10,000 test digits in a fixed shuffled order, and the
    indices of the 5000 that serve as the reference of every table cell."""
    val = digits_by_class(False, "cpu")
    real = torch.cat([val[c] for c in range(10)])[:, 0].numpy()
    real = real[np.random.default_rng(0).permutation(len(real))]
    return f(real), np.random.default_rng(200).permutation(len(real))[:N]


def boot_fids(feat, ref, rng):
    """Distance of BOOT bootstrap resamples of `feat` to a fixed reference."""
    f = torch.as_tensor(np.asarray(feat, np.float64), device=DEV)
    n, out = len(f), []
    for i in range(0, BOOT, CHUNK):
        b = min(CHUNK, BOOT - i)
        g = f[torch.as_tensor(rng.integers(0, n, size=(b, n)), device=DEV)]      # (b, n, d)
        mu = g.mean(1)
        c = g - mu[:, None]
        cov = c.transpose(1, 2) @ c / (n - 1)
        d = mu - ref.mu
        w = torch.linalg.eigvalsh(ref.sqrt @ cov @ ref.sqrt)
        out.append(((d * d).sum(1) + cov.diagonal(dim1=1, dim2=2).sum(1) + ref.tr
                    - 2 * w.clamp_min(0).sqrt().sum(1)).cpu().numpy())
    return np.concatenate(out)


def floor_draws(RF):
    """FD between two disjoint sets of N real test digits, FLOOR_DRAWS times."""
    out = []
    for r in range(FLOOR_DRAWS):
        h = np.random.default_rng(100 + r).permutation(len(RF))
        out.append(Ref(RF[h[N:2 * N]]).fid(RF[h[:N]]))
    return np.array(out)


def main():
    out_path = WORK / "eval" / "mnist" / "fd_ci.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    f = feature_fn()
    RF, qref = real_features(f)
    ref = Ref(RF[qref])
    floor = floor_draws(RF)
    out = {"n": N, "boot": BOOT, "horizons": HORIZONS,
           "floor": {"mean": float(floor.mean()), "sd": float(floor.std(ddof=1)), "lo": float(np.percentile(floor, 2.5)),
                     "hi": float(np.percentile(floor, 97.5)), "draws": FLOOR_DRAWS}, "cells": {}}
    print(f"floor {floor.mean():.2f}  95% [{np.percentile(floor, 2.5):.2f}, {np.percentile(floor, 97.5):.2f}]", flush=True)

    rng = np.random.default_rng(7)
    for stem in ORDER:
        path = runs.capture(stem)
        if not path.exists():
            print(f"  {stem}: no capture at {path}, skipped", flush=True)
            continue
        cells = out["cells"][stem] = {}
        with np.load(path) as d:
            keep = [int(x) for x in d["keep"]]
            FH = d["frames_h"]
            for h in HORIZONS:
                ft = f(FH[keep.index(h)])
                bs = boot_fids(ft, ref, rng)
                cells[str(h)] = {"fid": ref.fid(ft), "lo": float(np.percentile(bs, 2.5)),
                                 "hi": float(np.percentile(bs, 97.5)), "sd": float(bs.std(ddof=1))}
        print(f"  {stem:32s}" + " ".join(f"{cells[str(h)]['fid']:9.1f}" for h in HORIZONS), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
