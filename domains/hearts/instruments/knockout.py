"""The perturbation screen: how far does removing one cell type move the outcome?

The model is stochastic -- fire rate 0.5 and fresh noise at every update -- so
one rollout is a draw, not an answer. Comparing two POINTS therefore measures
the model's own chaos; this compares two DISTRIBUTIONS instead. For each cell
type we simulate N settled states with that type suppressed and ask how far
their distribution has moved from the unperturbed one.

    W(A, B) = mean over (channel, pixel) of the 1-D Wasserstein-1 distance
              between the two empirical marginals at that channel and pixel

For equal sample sizes that is just mean|sort(a) - sort(b)|. The perturbed
channel is EXCLUDED from the average: suppressing a channel trivially changes
its own distribution, and the interesting question is what happened to
everything else.

The null is measured, not assumed, and paired the same way the effect is. Each
knockout shares its seeds and their order with the control, so the control is
simulated R times over those same seeds and every pair of replicates gives one
"no effect" distance. R = 4 gives 6 null values. Reported as

    rho(c) = mean W(control, knockout_c) / mean W(control, control')

so rho = 1 means a perturbation indistinguishable from rerunning the model.

Pairing is what makes this work: every condition reuses the SAME N sections and
the SAME N wounds, so seed and wound variance cancel and the remaining spread
is the dynamics. An earlier version split pooled controls into random halves
instead, which scrambled which seeds landed in which half and inflated the null
so badly that most knockouts scored negative.

    python -m domains.hearts.instruments.knockout --run hearts_regis_seed0

Writes <workspace>/eval/hearts/knockout_<stem>.json, which the tables and the
group-recovery figures read; no measurement is recomputed downstream.
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
N_SAMPLES = 100
N_CONTROL_REPS = 4


def w1_marginal(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean 1-D W1 over every coordinate. a, b: (N, D), equal N."""
    return float((a.sort(dim=0).values - b.sort(dim=0).values).abs().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, choices=runs.ALL)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rule, cfg = R.load_rule(runs.checkpoint(WORK, a.run), dev)
    rng = np.random.default_rng(SEED)
    state0, mask = R.seed_batch(C.PATH, N_SAMPLES, rng, dev,
                                holdout_hearts=cfg.get("holdout_heart") or ())
    m = (mask > 0).to(state0.dtype)
    print(f"{a.run}: {N_SAMPLES} wounded hearts, {N_CONTROL_REPS} control "
          f"replicates, {len(C.CELL_TYPES)} knockouts", flush=True)

    def settle(ko, seed):
        s, _ = R.roll(rule, state0.clone(), knockout=ko, ko_mode="no-increase", seed=seed)
        # the 21 visible channels (20 cell types + damage), restricted to the
        # heart: the critic's view, not the hidden state
        return (s[:, :N_VISIBLE] * m).contiguous()

    def flat(t, drop=None):
        keep = [c for c in range(t.shape[1]) if c != drop]
        return t[:, keep].reshape(t.shape[0], -1)

    # the control replicates differ from each other ONLY in the model's
    # stochastic realisation, which is exactly what a knockout differs by too
    ctrls = [settle(None, seed=1000 + i) for i in range(N_CONTROL_REPS)]
    null = [w1_marginal(flat(ctrls[i]), flat(ctrls[j]))
            for i in range(len(ctrls)) for j in range(i + 1, len(ctrls))]
    null_mean = float(np.mean(null))
    print(f"  null W1 {null_mean:.5f} +/- {np.std(null, ddof=1):.5f} "
          f"over {len(null)} control pairs", flush=True)

    rows = []
    for c, name in enumerate(C.CELL_TYPES):
        ko = settle(c, seed=2000 + c)
        # average over EVERY control replicate, so one unlucky control
        # realisation cannot set the effect size
        w_all = float(np.mean([w1_marginal(flat(x), flat(ko)) for x in ctrls]))
        w_excl = float(np.mean([w1_marginal(flat(x, drop=c), flat(ko, drop=c))
                                for x in ctrls]))
        rows.append({"channel": name, "idx": c, "group": C.GROUP[name],
                     "w1_all": w_all, "w1_excl": w_excl,
                     "rho": w_excl / null_mean})
        print(f"  {name:<26} rho {rows[-1]['rho']:6.2f}", flush=True)

    rows.sort(key=lambda r: -r["rho"])
    out = WORK / f"eval/hearts/knockout_{a.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"run": a.run, "rows": rows,
         "null": {"mean": null_mean, "sd": float(np.std(null, ddof=1)),
                  "values": null, "n_pairs": len(null)},
         "meta": {"n_samples": N_SAMPLES, "n_control_reps": N_CONTROL_REPS,
                  "ko_mode": "no-increase", "seed": SEED,
                  "steps_per_leg": R.STEPS_PER_LEG, "settle_steps": R.SETTLE_STEPS,
                  "channels": C.CELL_TYPES}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
