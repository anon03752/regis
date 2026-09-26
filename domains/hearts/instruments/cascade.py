"""The border-zone cascade: does blocking one state stop the next?

The atlas describes border-zone myocardium as five states, each enriched in a
narrow window of the arc, and calls them a cascade. Every state is enriched at
particular stages in the training data, so ANY model that fits the marginals
reproduces the succession -- the succession is not evidence of anything. What a
knockout adds is the DEPENDENCY: if the model has learnt the cascade as a chain,
zeroing a state should suppress those downstream and leave those upstream
intact, whereas a model that has only learnt WHEN each state appears will be
unaffected.

Each state is zeroed in turn (the hard intervention -- here the question is
whether a downstream state can rise at all) and the abundance trajectories of
all five are recorded at every update. Two controls sit alongside:

  no injury     the same seeds rolled the same length, never cut. No cascade
                state should rise at all, which establishes that the response
                is triggered by injury rather than by elapsed time.
  bystanders    knockouts of types prior biology does not implicate -- atrial
                myocardium, valves, atrial blood. Zeroing any channel drives the
                state off-distribution, so without this a general derailment
                would be indistinguishable from a captured dependency.

    python -m domains.hearts.instruments.cascade --run hearts_regis_seed0

Writes <workspace>/eval/hearts/cascade_<stem>.json.
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

SEED = 0                  # the sample draw; fixed for every run of record
N_SAMPLES = 64
N_CONTROL = 3             # replicate rollouts averaged into the unblocked curve
N_BLOCK = 2               # ... into each blocked curve and the uninjured control


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, choices=runs.ALL)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rule, cfg = R.load_rule(runs.checkpoint(WORK, a.run), dev)
    holdout = cfg.get("holdout_heart") or ()
    rng = np.random.default_rng(SEED)
    wounded, mask = R.seed_batch(C.PATH, N_SAMPLES, rng, dev, holdout_hearts=holdout)
    clean, mask_c = R.seed_batch(C.PATH, N_SAMPLES, np.random.default_rng(SEED),
                                 dev, wound=False, holdout_hearts=holdout)

    blocks = [C.CELL_TYPES.index(c) for c in C.CASCADE + C.CASCADE_CONTROLS]
    print(f"{a.run}: {N_SAMPLES} hearts, {len(blocks)} blocks "
          f"({len(C.CASCADE)} cascade + {len(C.CASCADE_CONTROLS)} bystander) "
          f"+ an unblocked and an uninjured control", flush=True)

    def curve(state, mask_, ko, seed, reps):
        """(1 + T, n_visible): each visible channel's mean abundance inside the
        heart, averaged over `reps` replicate rollouts of all the samples."""
        per = [R.heart_means(R.roll(rule, state.clone(), knockout=ko, ko_mode="zero",
                                    record=True, seed=seed + r)[2], mask_)
               for r in range(reps)]
        return torch.cat(per, dim=1).mean(dim=1).cpu().numpy()

    out = {"run": a.run, "channels": C.CELL_TYPES, "cascade": C.CASCADE,
           "controls": C.CASCADE_CONTROLS,
           "ctrl": curve(wounded, mask, None, 3000, N_CONTROL).tolist(),
           "uninjured": curve(clean, mask_c, None, 3100, N_BLOCK).tolist(),
           "blocks": []}
    print("  unblocked and uninjured controls done", flush=True)
    for k in blocks:
        out["blocks"].append({"ko": k, "channel": C.CELL_TYPES[k],
                              "curve": curve(wounded, mask, k, 4000 + 10 * k, N_BLOCK).tolist()})
        print(f"  blocked {C.CELL_TYPES[k]}", flush=True)

    # the cohort's own abundances, for the dots the figures overlay
    _seeds, _ids, banks, _refs = C.load_cohort(C.PATH, "cpu", holdout)
    out["real"] = [[float(((banks[tp][0][:, c] * (banks[tp][1][:, 0] > 0))
                           .sum(dim=(1, 2)) / (banks[tp][1][:, 0] > 0)
                           .sum(dim=(1, 2)).clamp_min(1)).mean())
                    for c in range(len(C.CELL_TYPES))] for tp in C.TP_ORDER]
    out["meta"] = {"n_samples": N_SAMPLES, "n_control": N_CONTROL, "n_block": N_BLOCK,
                   "ko_mode": "zero", "seed": SEED, "steps_per_leg": R.STEPS_PER_LEG}

    path = WORK / f"eval/hearts/cascade_{a.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
