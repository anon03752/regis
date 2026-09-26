"""Capture 1000 digit transitions from the final MMSFM checkpoint.

Uses the shared starts from capture_frames.py and 160 SRA1 steps per
transition, with float64 states and unclamped classifier inputs. The clock
for a trajectory starting at digit c is (c + transition - 1)/10 + t.

Trajectory i belongs to shard i % 6. Each shard is padded to 864 entries by
repeating its first trajectories and uses a separate Gaussian noise stream.
The heads use TF32, fp32 SDPA, and autotuning. The merged file follows
capture_frames.py, with grid holding the first four trajectories classified
as each digit, padded with zeros when fewer are available."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from workspace import WORK
from baselines.mmsfm.net import Drift, build_unets, periodic_time_embedding, sdpa_attention, upstream
from domains.mnist.classifier import load_classifier
from domains.mnist.data import digits_by_class
from domains.mnist.instruments.measure import PER, real_starts
from domains.mnist.results import runs
from domains.mnist.results.capture_frames import KEEP

NGRID = 4
SHARDS, BATCH, STEPS = 6, 864, 160    # trajectories per batch (padded); SRK steps per transition


def srk_schedule(dev):
    """(start, length) of the SRK steps over one transition, as torchsde accumulates them"""
    t, rows = 0., []
    while t < .1:
        end = min(t + .1 / STEPS, .1)
        rows.append((t, end - t))
        t = end
    return torch.tensor(rows, dtype=torch.float64, device=dev)


def one_transition(drift, x, phase, schedule, sigma, gen):
    """One transition: SRA1 (torchsde's additive-noise SRK) with Gaussian increments W and
    space-time integrals U, term by term in torchsde's order (the order fixes the rounding)"""
    z = torch.randn((len(schedule), 2, *x.shape), device=x.device, dtype=x.dtype, generator=gen)
    dts = schedule[:, 1].reshape(-1, *[1] * x.ndim)
    W = dts.sqrt() * z[:, 0]
    U = dts ** 1.5 * (.5 * z[:, 0] + z[:, 1] / (12 ** .5))    # Var U = dt^3/3, Cov(W, U) = dt^2/2
    for (t0, dt), dw, du in zip(schedule, W, U):
        rdt = 1 / dt
        f0 = drift(t0 + 0 * dt, x, phase)
        y1 = x + (1 / 3) * f0 * dt + (sigma / 2) * (1 * dw + -1 * du * rdt)
        h1 = x + (3 / 4) * f0 * dt + (sigma / 2) * ((3 / 2) * du * rdt)
        f1 = drift(t0 + (3 / 4) * dt, h1, phase)
        x = y1 + (2 / 3) * f1 * dt + (sigma / 2) * (0 * dw + 1 * du * rdt)
    return x


def part_path(run, shard):
    return WORK / "data" / "mnist" / "captures" / "parts" / f"{run}_shard{shard}.npz"


@torch.inference_mode()
def capture_shard(run, shard):
    dev = "cuda"
    tr = upstream()
    from torchcfm.models.unet import nn as unet_nn, unet
    import torch._inductor.config as inductor
    torch.set_num_threads(4)
    unet.timestep_embedding = periodic_time_embedding
    # the execution of the paper captures: no activation-checkpoint wrappers, fp32 SDPA,
    # F.silu, TF32, and kernels chosen by timing
    unet.AttentionBlock.forward = lambda self, x: self._forward(x)
    unet.ResBlock.forward = lambda self, x, emb: self._forward(x, emb)
    unet.QKVAttentionLegacy.forward = sdpa_attention(torch.float32)
    unet_nn.SiLU.forward = lambda self, x: F.silu(x)
    inductor.freezing = True
    if hasattr(inductor, "force_same_precision"):      # torch 2.7, the paper captures; gone by 2.13
        inductor.force_same_precision = False
    if hasattr(inductor, "precompilation_timeout_seconds"):   # later torch stops autotuning after 5 min
        inductor.precompilation_timeout_seconds = 60 * 60      # 2.7's limit
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    d = WORK / "runs" / run
    cfg = json.loads((d / "config.json").read_text())
    heads = build_unets(tr, tuple(cfg["dims"]), cfg["channels"], cfg["attention_resolutions"], dev)
    w = torch.load(sorted(d.glob("step_*.pt"))[-1], map_location="cpu", weights_only=True)   # the last checkpoint
    for m, key in zip(heads, ["model_state_dict", "score_model_state_dict"]):
        m.load_state_dict(w[key])
        m.eval().requires_grad_(False).to(memory_format=torch.channels_last)
    drift = torch.compile(Drift(*heads), fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
    clf = load_classifier(dev)

    start, x0 = real_starts(digits_by_class(False, "cpu"), 1, "cpu")
    idx = np.arange(shard, len(start), SHARDS)
    n, pad = len(idx), np.arange(BATCH) % len(idx)
    x = x0[idx][pad].to(dev, torch.float64)
    digit = torch.from_numpy(start[idx][pad]).to(dev)
    schedule, gen = srk_schedule(dev), torch.Generator(device=dev).manual_seed(20269300 + shard)

    H = len(KEEP)
    frames, pred = np.zeros((H, n, 32, 32), np.float32), np.zeros((H, n), np.int16)
    vmax = np.zeros(H)
    for transition in range(1, KEEP[-1] + 1):
        x = one_transition(drift, x, ((digit + transition - 1) % 10).double() / 10, schedule, cfg["sigma"], gen)
        assert torch.isfinite(x).all(), f"non-finite state after transition {transition}"
        if transition not in KEEP:
            continue
        k = KEEP.index(transition)
        vis = x[:n].float()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        pred[k] = clf(vis).argmax(1).cpu().numpy()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        frames[k] = vis[:, 0].cpu().numpy()
        vmax[k] = float(x[:n].abs().max())
        print(f"{run} shard {shard}: transition {transition}, max|pixel| {vmax[k]:.3f}", flush=True)
    out = part_path(run, shard)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, frames=frames, pred=pred, vmax=vmax)
    print(f"-> {out}")


def merge(run):
    start = np.repeat(np.arange(10), PER)
    H, N = len(KEEP), len(start)
    frames_h, pred = np.zeros((H, N, 32, 32), np.float32), np.zeros((H, N), np.int16)
    vmax = np.zeros(H)
    for s in range(SHARDS):
        with np.load(part_path(run, s)) as z:
            frames_h[:, s::SHARDS], pred[:, s::SHARDS] = z["frames"], z["pred"]
            vmax = np.maximum(vmax, z["vmax"])
    grid = np.zeros((H, 10, NGRID, 32, 32), np.float32)
    for k in range(H):
        for dg in range(10):
            i = np.where(pred[k] == dg)[0][:NGRID]
            grid[k, dg, :len(i)] = frames_h[k, i]
    out = runs.capture(run)
    np.savez_compressed(out, keep=np.array(KEEP), start=start, frames_h=frames_h, pred=pred, vmax=vmax, grid=grid)
    print(f"-> {out}  ({out.stat().st_size / 1e6:.0f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, choices=runs.MMSFM, help="runs/<run> holds config.json and the checkpoints")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--shard", type=int, choices=range(SHARDS))
    g.add_argument("--merge", action="store_true", help="join the six shards into the capture")
    a = p.parse_args()
    if a.merge:
        merge(a.run)
    else:
        capture_shard(a.run, a.shard)


if __name__ == "__main__":
    main()
