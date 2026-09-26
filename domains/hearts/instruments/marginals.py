"""Simulated against observed cell-type profiles over the arc.

What this is and is not. The cohort's diversity limits what a goodness-of-fit
score can mean here: absolute compartment sizes have coefficients of variation
of 34-88% across sections, and thin structures are missed outright depending on
where the cross-section fell -- valves are absent from every section at two of
the eight stages. So the assessment of fit is deliberately coarse: does each
cell type peak when the cohort peaks? The reported number is the median over
cell types of the correlation between the model's trajectory, resampled at the
measured stages, and the cohort mean. A strict marginal-matching score at this
level of technical variability would mostly measure which sections were cut.

The model curve is recorded at EVERY update, not only at the leg ends, so a
figure can show the continuous trajectory the rule generates between observed
times. Sub-step hours interpolate linearly inside the leg's interval, which is
the honest mapping: the rule is trained to land on the endpoints and nothing
supervises the inside.

    python -m domains.hearts.instruments.marginals --run hearts_regis_seed0

Writes <workspace>/eval/hearts/marginals_<stem>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts import rollout as R
from domains.hearts.results import runs
from domains.hearts.rule import N_VISIBLE

SEED = 0                  # the sample draw; fixed for every run of record
N_ROLLOUTS = 48
# the 20 cell types plus damage: the same 21 channels the rollout records, so
# the cache's channel list and its arrays cannot disagree
CHANNELS = C.CELL_TYPES + [C.DAMAGE_CHANNEL]


def cohort_curves(banks):
    """-> (mean, sd, n) each (N_STAGES, n_types): the cohort's within-mask mean
    composition per section, averaged over the sections at each stage."""
    mean = np.zeros((C.N_STAGES, N_VISIBLE))
    sd, n = np.zeros_like(mean), np.zeros(C.N_STAGES, dtype=int)
    for i, tp in enumerate(C.TP_ORDER):
        if tp not in banks:
            continue
        comp, mask = banks[tp]
        m = (mask[:, 0] > 0)
        per = np.stack([(comp[k, :N_VISIBLE][:, m[k]].mean(dim=1)).cpu().numpy()
                        for k in range(comp.shape[0])])
        mean[i], sd[i], n[i] = per.mean(0), per.std(0), per.shape[0]
    return mean, sd, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, choices=runs.ALL)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rule, cfg = R.load_rule(runs.checkpoint(WORK, a.run), dev)
    holdout = cfg.get("holdout_heart") or ()
    _seeds, _ids, banks, _refs = C.load_cohort(C.PATH, dev, holdout)
    real_mean, real_sd, real_n = cohort_curves(banks)

    rng = np.random.default_rng(SEED)
    state, mask = R.seed_batch(C.PATH, N_ROLLOUTS, rng, dev, holdout_hearts=holdout)
    _s, _legs, every = R.roll(rule, state, record=True, seed=5000)
    curves = R.heart_means(every, mask)                    # (T, N, n_types)
    model_mean = curves.mean(dim=1).cpu().numpy()
    model_sd = curves.std(dim=1).cpu().numpy()

    # hours at each recorded state: the wound at 0, then leg j carries the state
    # from stage j-1 to stage j over `steps_per_leg` updates
    hours = list(C.TIMEPOINTS.values())
    rec = [0.0]
    for j in range(1, C.N_STAGES):
        h0, h1 = hours[j - 1], hours[j]
        rec += [h0 + (h1 - h0) * (i + 1) / R.STEPS_PER_LEG
                for i in range(R.STEPS_PER_LEG)]

    out = WORK / f"eval/hearts/marginals_{a.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"run": a.run, "channels": CHANNELS, "timepoints": C.TP_ORDER,
         "stage_hours": hours, "real_mean": real_mean.tolist(),
         "real_sd": real_sd.tolist(), "real_n": real_n.tolist(),
         "model_mean": model_mean.tolist(), "model_sd": model_sd.tolist(),
         "model_hours": rec,
         "meta": {"n_rollouts": N_ROLLOUTS, "seed": SEED,
                  "steps_per_leg": R.STEPS_PER_LEG}}))
    print(f"wrote {out}  ({len(rec)} recorded updates over the arc)")


if __name__ == "__main__":
    main()
