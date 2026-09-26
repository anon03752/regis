"""Evaluate REGIS runs trained on different grids of observation times.

Report L/L_GT, the slope of log L against log t over available times in
[100, 1000], and the singleton fraction at t = 1000. The reference band
spans the three REGIS runs on the standard grid. PASS_L and PASS_EXP define
the thresholds used in the Markdown report. Writes
eval/marginals512.{json,md}."""
import json
from multiprocessing import Pool

import numpy as np

from domains.ising.results import runs
from domains.ising.results.grade_campaign512 import cache_task
from workspace import WORK

PASS_L, PASS_EXP = (0.95, 1.06), (0.47, 0.50)


def ints(d):
    return {int(k): v for k, v in d.items()}


if __name__ == "__main__":
    J = json.load(open(WORK / "eval/campaign512.json"))
    PC = J["per_cache"]
    L_GT = ints(J["L_GT"])
    EXP = [k for k in sorted(L_GT) if 100 <= k <= 1000]
    with Pool(8) as pool:
        R = dict(pool.map(cache_task, runs.PLACEMENT.values()))
    R.update({s: PC[s] for s in runs.REGIS})

    def summary(stem):
        L = ints(R[stem]["L"])
        # the paper caches lack some of these times (placement: 961; REGIS: 130, 231, 408, 850, 961)
        xs = [k for k in EXP if k in L]
        return dict(ratio={k: L[k] / L_GT[k] for k in L},
                    exponent=float(np.polyfit(np.log(xs), np.log([L[k] for k in xs]), 1)[0]),
                    iso1k=ints(R[stem]["iso"])[1000])
    ref = [summary(s) for s in runs.REGIS]
    band = {k: (min(r["ratio"][k] for r in ref), max(r["ratio"][k] for r in ref)) for k in (300, 1000, 4000)}
    eband = (min(r["exponent"] for r in ref), max(r["exponent"] for r in ref))
    texp = [float(np.polyfit(np.log(EXP), np.log([PC[s]["L"][str(k)] for k in EXP]), 1)[0]) for s in runs.TRUTH]
    out = {"truth_exponent": texp, "reference_band": {str(k): v for k, v in band.items()}, "reference_exponent": eband,
           "runs": [], "reference_runs": []}
    md = ["# Marginal-placement sweep: REGIS 3×3, one run per grid of marginals, 512 fields at 512²", "",
          "Reference = the three REGIS seeds of tab:fig3_ising (marginals 3, 10, 17, 300, 1000), same 512 fields, "
          "same instrument: "
          + ", ".join(f"t = {k}: {lo:.3f}–{hi:.3f}" for k, (lo, hi) in band.items())
          + f"; exponent {eband[0]:.3f}–{eband[1]:.3f}. Engine exponent {' · '.join(f'{x:.3f}' for x in texp)}. "
          f"Pass criteria: L/truth at 1k and 4k in [{PASS_L[0]}, {PASS_L[1]}], exponent in [{PASS_EXP[0]}, {PASS_EXP[1]}].", "",
          "| marginals | seed | L/truth t = 300 | t = 1000 | t = 4000 | exponent | isolated sites @1k | verdict |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for (grid, seed), stem in runs.PLACEMENT.items():
        s = summary(stem)
        r = s["ratio"]
        ok = all(PASS_L[0] <= r[k] <= PASS_L[1] for k in (1000, 4000)) and PASS_EXP[0] <= s["exponent"] <= PASS_EXP[1]
        out["runs"].append(dict(times=list(grid), seed=seed, **s, passed=ok))
        md.append(f"| {', '.join(map(str, grid))} | {seed} | {r[300]:.3f} | {r[1000]:.3f} | {r[4000]:.3f} | {s['exponent']:.3f} | "
                  f"{s['iso1k']:.1e} | {'pass' if ok else '**outside**'} |")
    for seed, r in zip(runs.SEEDS, ref):
        out["reference_runs"].append(dict(seed=seed, **r))
        md.append(f"| reference: 3, 10, 17, 300, 1000 | {seed} | {r['ratio'][300]:.3f} | {r['ratio'][1000]:.3f} | "
                  f"{r['ratio'][4000]:.3f} | {r['exponent']:.3f} | {r['iso1k']:.1e} | |")
    (WORK / "eval/marginals512.json").write_text(json.dumps(out, indent=1, default=float))
    (WORK / "eval/marginals512.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
