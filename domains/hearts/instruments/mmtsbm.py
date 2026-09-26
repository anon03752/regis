"""MMtSBM's cell-type profiles over the arc, chained and one-step.

    python -m domains.hearts.instruments.mmtsbm --run hearts_mmtsbm_seed0

The forward drift of the checkpoint before the fitting iterations carries the
nine uninjured start sections from stage to stage up to 28 dpa, 60 Euler steps
per bridge with the noise off. Chained is the free rollout, as for every other
model; one-step restarts each bridge from the real sections of the stage it
leaves, so no error is inherited. Profiles are measured in cell units, after inverting the z-score:
a sample's mean over the bins it occupies (channel sum above 0.15), and the
cohort's over each section's mask. Knock-outs are not run on MMtSBM.

Writes <workspace>/eval/hearts/marginals_<stem>_{chained,onestep}.json, in the
layout of marginals.py.
"""
import argparse
import json

import numpy as np
import torch

from workspace import WORK
from baselines.mmtsbm.bridge import simulate
from baselines.mmtsbm.net import build_drift
from domains.hearts import cohort as C
from domains.hearts.data_bridges import OUT as DATA
from domains.hearts.results import runs

CKPT = "ckpt_0000_forward.pt"      # the checkpoint of record: warm-up, before any fitting iteration
STEPS, EVERY = 60, 4               # Euler steps per bridge; a profile every 4th
SIGMA = 0.0
OCCUPIED = 0.15


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, choices=runs.MMTSBM)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = WORK / "runs" / a.run
    cfg = json.loads((run / "config.json").read_text())
    anchors = json.loads((DATA / "anchors.json").read_text())
    drift = build_drift(cfg["lat"], tuple(int(x) for x in cfg["blocks"].split(",")),
                        cfg["layers_per_block"], channels=len(anchors["channels"]))
    drift.load_state_dict(torch.load(run / "ckpts" / CKPT, map_location="cpu", weights_only=False)["ema"])
    drift = drift.to(dev).eval()

    scale = torch.tensor(anchors["channel_scale"])[None, :, None, None]
    centre = torch.tensor(anchors["channel_centre"])[None, :, None, None]
    X = [torch.load(DATA / f"hearts_t{i}.pt") for i in range(C.N_STAGES + 1)]
    M = [torch.load(DATA / f"hearts_mask{i}.pt") for i in range(C.N_STAGES + 1)]
    cells = lambda z: z.cpu() * scale + centre

    def profile(raw, bins):
        per = np.stack([raw[k][:, bins[k]].mean(1).numpy() for k in range(len(raw))])
        return per.mean(0), per.std(0)

    real = [profile(cells(X[i]), M[i][:, 0] > 0) for i in range(C.N_STAGES)]
    hours = list(C.TIMEPOINTS.values())
    for mode in ("chained", "onestep"):
        frames, rec = [profile(cells(X[0]), cells(X[0]).sum(1) > OCCUPIED)], [0.0]
        z = X[0].to(dev)

        def read(i, zz, leg):
            if i % EVERY == 0:
                raw = cells(zz)
                frames.append(profile(raw, raw.sum(1) > OCCUPIED))
                h0, h1 = hours[leg], hours[leg + 1]
                rec.append(h0 + (h1 - h0) * i / STEPS)

        for leg in range(C.N_STAGES - 1):
            if mode == "onestep":
                z = X[leg].to(dev)
            z = simulate(drift, z, leg, leg + 1, SIGMA, "forward", STEPS,
                         on_step=lambda i, zz, leg=leg: read(i, zz, leg))
        out = WORK / f"eval/hearts/marginals_{a.run}_{mode}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"run": f"{a.run}_{mode}", "channels": anchors["channels"], "timepoints": C.TP_ORDER,
             "stage_hours": hours, "real_mean": [m.tolist() for m, _ in real],
             "real_sd": [s.tolist() for _, s in real], "real_n": [len(x) for x in X[:C.N_STAGES]],
             "model_mean": [m.tolist() for m, _ in frames], "model_sd": [s.tolist() for _, s in frames],
             "model_hours": rec,
             "meta": {"n_rollouts": len(X[0]), "steps_per_leg": STEPS // EVERY, "sigma": SIGMA,
                      "checkpoint": CKPT}}))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
