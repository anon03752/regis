"""The perturbation screen: how far does removing one cell type move the outcome?

The model is stochastic -- fire rate 0.5 and fresh noise at every update -- so
the comparison is made within each heart. The same wounded section is rolled
three times from the same start: twice unperturbed, to measure the model's own
run-to-run variability on that heart, and once with cell type c suppressed
throughout. With x_i, x'_i and x_{i,c} its three settled states, clipped at zero,

    Delta_i(u, v) = mean |u - v| over the bins inside heart i's mask

    rho(c) = mean_i  Delta_i(x_i, x_{i,c}) / Delta_i(x_i, x'_i)

where the numerator excludes channel c -- suppressing a channel trivially
changes that channel, and the question is what happened to everything else --
and the denominator is taken over all 21 visible channels. Each heart is its own
control, so rho = 1 means a perturbation indistinguishable from rerunning the
model; normalising per heart matters because the run-to-run difference varies
between hearts. States are clipped because knocked-out states can go strongly
negative, and unclipped the distance would mostly measure the model
destabilising rather than tissue changing.

The suppression is no-increase: the channel is clamped to its pre-step value,
so the type can decay but is never produced or recruited.

    python -m domains.hearts.instruments.knockout --run hearts_regis_seed0

Writes <workspace>/eval/hearts/knockout_<stem>.json.
"""
import argparse
import json

import numpy as np
import torch

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts import rollout as R
from domains.hearts.results import runs
from domains.hearts.rule import N_VISIBLE

SEED = 0                  # the sample draw; fixed for every run of record
N_SAMPLES = 100


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, choices=runs.ALL)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rule, cfg = R.load_rule(runs.checkpoint(WORK, a.run), dev)
    state0, mask = R.seed_batch(C.PATH, N_SAMPLES, np.random.default_rng(SEED), dev,
                                holdout_hearts=cfg.get("holdout_heart") or ())
    m = (mask[:, 0] > 0).cpu().numpy()
    types = [c for c in C.CELL_TYPES if c != C.NOT_A_CELL_TYPE]
    print(f"{a.run}: {N_SAMPLES} wounded hearts, {len(types)} knock-outs", flush=True)

    def settled(ko, seed):
        s, _ = R.roll(rule, state0.clone(), knockout=ko, ko_mode="no-increase", seed=seed)
        return s[:, :N_VISIBLE].clamp_min(0.0).cpu().numpy().astype(np.float64)

    def per_heart(x, y, drop=None):
        keep = [c for c in range(N_VISIBLE) if c != drop]
        return np.array([np.abs(x[i][keep][:, m[i]] - y[i][keep][:, m[i]]).mean()
                         for i in range(len(x))])

    x, x_rerun = settled(None, 1000), settled(None, 1001)
    noise = per_heart(x, x_rerun)
    print(f"  rerun difference {noise.mean():.4f} +/- {noise.std():.4f} per heart", flush=True)

    rows = []
    for name in types:
        c = C.CELL_TYPES.index(name)
        q = per_heart(x, settled(c, 2000 + c), drop=c) / noise
        rows.append({"channel": name, "group": C.GROUP[name], "rho": float(q.mean()),
                     "sd": float(q.std()), "per_heart": q.tolist()})
        print(f"  {name:<26} rho {q.mean():6.2f}", flush=True)
    rows.sort(key=lambda r: -r["rho"])

    out = WORK / f"eval/hearts/knockout_{a.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run": a.run, "rows": rows,
        "rerun": {"mean": float(noise.mean()), "sd": float(noise.std())},
        "meta": {"n_samples": N_SAMPLES, "ko_mode": "no-increase", "seed": SEED,
                 "steps_per_leg": R.STEPS_PER_LEG, "settle_steps": R.SETTLE_STEPS}}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
