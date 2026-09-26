"""MMSFM's profiles, cascade and knock-out screen, in the layouts of the REGIS caches.

    python -m domains.hearts.instruments.mmsfm --run hearts_mmsfm_seed0

The protocol of the evaluation of record of these checkpoints. Twelve rollouts,
four noise replicas of each of the three held-out uninjured sections (one heart,
so not twelve animals), run once around the loop: eight transitions of 1/8 with
160 nominal SRA1 steps each. A knock-out zeroes one channel at the start and
holds its state, drift and noise at zero at every internal stage, with the same
noise as the unblocked run in the other channels; nothing redistributes its mass.

Readouts are in cell units (four times the sampler's state) over each sample's
occupied support, the bins whose 21 channels sum above 0.15; the cohort
reference is the held-out sections of each stage. The screen's rho(c) is the
effect at 28 dpa, each rollout's difference from the unblocked run over its
rerun with fresh noise, averaged over the twelve, with the channels of the REGIS
screen. Needs a CUDA GPU: the noise of
record is drawn there.

Writes <workspace>/eval/hearts/{marginals,cascade,knockout}_<stem>.json.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from workspace import WORK
from baselines.mmsfm.net import (Drift, build_unets, one_transition, periodic_time_embedding, sdpa_attention,
                                 srk_schedule, upstream)
from baselines.mmsfm.train import VALIDATION_HEARTS, heart_of
from domains.hearts import cohort as C
from domains.hearts.results import runs

REPLICAS, STEPS = 4, 160
SEED, RERUN_SEED = 921800, 921801
OCCUPIED = 0.15
READ = C.CELL_TYPES + [C.DAMAGE_CHANNEL]     # the 21 channels read out; DamageRef is generated, not read
SCORED = [c for c in C.CELL_TYPES if c != C.NOT_A_CELL_TYPE]
AT_28DPA = C.N_STAGES - 1


def profile(native):
    """(N, 21, H, W) -> (N, 21): each sample's mean over its occupied support"""
    occ = native.sum(1) > OCCUPIED
    return np.stack([native[k][:, occ[k]].mean(1) for k in range(len(native))])


def load_heads(run, dev):
    tr = upstream()
    from torchcfm.models.unet import nn as unet_nn, unet
    unet.timestep_embedding = periodic_time_embedding
    # the execution of the evaluation of record: no activation-checkpoint wrappers, fp32 SDPA,
    # F.silu, TF32
    unet.AttentionBlock.forward = lambda self, x: self._forward(x)
    unet.ResBlock.forward = lambda self, x, emb: self._forward(x, emb)
    unet.QKVAttentionLegacy.forward = sdpa_attention(torch.float32)
    unet_nn.SiLU.forward = lambda self, x: F.silu(x)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    d = WORK / "runs" / run
    cfg = json.loads((d / "config.json").read_text())
    heads = build_unets(tr, tuple(cfg["dims"]), cfg["channels"], cfg["attention_resolutions"], dev)
    w = torch.load(sorted(d.glob("step_*.pt"))[-1], map_location="cpu", weights_only=True)
    for m, key in zip(heads, ["model_state_dict", "score_model_state_dict"]):
        m.load_state_dict(w[key])
        m.eval().requires_grad_(False).to(memory_format=torch.channels_last)
    return Drift(*heads), cfg["sigma"]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, choices=runs.MMSFM)
    a = p.parse_args()
    dev = "cuda"
    drift, sigma = load_heads(a.run, dev)

    z = np.load(C.PATH, allow_pickle=False)
    stage = np.array([C.TP_ORDER.index(str(t)) for t in z["timepoint"]])
    held = np.array([heart_of(str(s)) in VALIDATION_HEARTS for s in z["section_id"]])
    native = 2.0 * (z["composition"].astype(np.float64) + 1.0)
    real = [profile(native[held & (stage == k)][:, :len(READ)]) for k in range(C.N_STAGES)]
    starts = (z["composition"][held & (stage == 0)] + 1) / 2          # the sampler's native/4
    x0 = torch.from_numpy(np.repeat(starts, REPLICAS, axis=0)).to(dev, torch.float64)

    duration = 1 / C.N_STAGES
    schedule = srk_schedule(dev, duration, STEPS)

    def rollout(seed, blocks=()):
        """The unblocked ensemble and one copy per knock-out, in one pass on the same noise.
        -> (per-stage profiles (8, copies, N, 21), 28 dpa states (copies, N, 21, H, W)), cell units"""
        copies = 1 + len(blocks)
        keep = torch.ones(copies, 1, x0.shape[1], 1, 1, device=dev, dtype=x0.dtype)
        for i, c in enumerate(blocks):
            keep[1 + i, :, c] = 0
        keep = keep.expand(-1, len(x0), -1, -1, -1).reshape(-1, x0.shape[1], 1, 1)
        x = x0.repeat(copies, 1, 1, 1) * keep
        gen = torch.Generator(device=dev).manual_seed(seed)
        prof = []
        for leg in range(C.N_STAGES):
            state = 4 * x[:, :len(READ)].float().cpu().numpy()
            prof.append(profile(state).reshape(copies, len(x0), -1))
            if leg == AT_28DPA:
                break
            phase = torch.full((len(x),), leg * duration, device=dev, dtype=torch.float64)
            x = one_transition(drift, x, phase, schedule, sigma, gen, keep if blocks else None, copies)
            assert torch.isfinite(x).all(), f"non-finite state after transition {leg + 1}"
        return np.stack(prof), state.reshape(copies, len(x0), *state.shape[1:])

    ko = [READ.index(name) for name in SCORED]
    print(f"unblocked and {len(ko)} knock-outs, then the rerun ...", flush=True)
    prof, ends = rollout(SEED, ko)
    _rerun, (x_rerun,) = rollout(RERUN_SEED)
    ctrl, x_ctrl = prof[:, 0], ends[0]
    blocks, rows = [], []
    for i, (name, c) in enumerate(zip(SCORED, ko)):
        blocks.append({"ko": c, "curve": prof[:, 1 + i].mean(1).tolist()})
        # as the REGIS screen: the knock-out's difference over the other 20 channels, the
        # rerun's over all 21
        others = [j for j in range(len(READ)) if j != c]
        num = np.abs(ends[1 + i][:, others] - x_ctrl[:, others]).mean((1, 2, 3))
        den = np.abs(x_rerun - x_ctrl).mean((1, 2, 3))
        q = num / den
        rows.append({"channel": name, "group": C.GROUP[name], "rho": float(q.mean()),
                     "sd": float(q.std()), "per_heart": q.tolist()})

    hours = list(C.TIMEPOINTS.values())
    meta = {"run": a.run, "n_rollouts": len(x0), "replicas": REPLICAS, "steps": STEPS,
            "seed": SEED, "rerun_seed": RERUN_SEED}
    out = WORK / "eval/hearts"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"marginals_{a.run}.json").write_text(json.dumps(
        {"run": a.run, "channels": READ, "timepoints": C.TP_ORDER, "stage_hours": hours,
         "real_mean": [r.mean(0).tolist() for r in real], "real_sd": [r.std(0).tolist() for r in real],
         "real_n": [len(r) for r in real], "model_mean": ctrl.mean(1).tolist(),
         "model_sd": ctrl.std(1).tolist(), "model_hours": hours,
         "meta": {**meta, "steps_per_leg": 1}}))
    # every knock-out outside the cascade is a control here, the regulatory stroma included
    (out / f"cascade_{a.run}.json").write_text(json.dumps(
        {"run": a.run, "channels": READ, "cascade": C.CASCADE, "ctrl": ctrl.mean(1).tolist(),
         "blocks": blocks, "controls": [c for c in SCORED if C.GROUP[c] != "A"], "meta": meta}))
    (out / f"knockout_{a.run}.json").write_text(json.dumps(
        {"run": a.run, "rows": rows, "meta": {**meta, "at": C.TP_ORDER[AT_28DPA]}}))
    print(f"wrote {out}/{{marginals,cascade,knockout}}_{a.run}.json")


if __name__ == "__main__":
    main()
