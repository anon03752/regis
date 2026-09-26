"""Compute domain-growth metrics from the Ising rollout caches.

For each readout, L is the reciprocal of the mean despeckled interface
density across fields. Also report the raw singleton fraction. L_GT is the
median L across three simulator runs. Writes eval/campaign512.json."""
import argparse
import json
from multiprocessing import Pool

import numpy as np

from domains.ising.instruments import rollouts
from domains.ising.instruments.despeckle import despeckle, rho_of, iso_frac
from domains.ising.results import runs
from workspace import WORK

STEMS = (runs.TRUTH + runs.REGIS + runs.REGIS_K5 + runs.REGIS_K7 + runs.CONTROL + runs.PAIRGAN + runs.DDPM
         + runs.MMSFM + runs.MMTSBM
         + [s for lad in (runs.LADDER_REGIS, runs.LADDER_DDPM) for ss in lad.values() for s in ss])


def cache_task(stem):
    """L and the isolated-site fraction at every read-out of one cache"""
    bits, ks, m = rollouts.load(runs.cache_path(stem))
    ns, lat = m["ns"], m["lat"]
    L, iso = {}, {}
    for row, k in zip(bits, ks):
        if k == 0:                   # exclude the shared initial state stored in transport caches
            continue
        f = rollouts.unpack(row, lat, ns)
        fc = [f[c:c + 64] for c in range(0, ns, 64)]    # chunks of 64 fields: memory
        L[k] = float(1 / np.mean(np.concatenate([rho_of(despeckle(x)) for x in fc])))
        iso[k] = float(np.mean(np.concatenate([iso_frac(x) for x in fc])))
    return stem, {"iso": iso, "L": L}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()

    stems = sorted(set(STEMS))
    present = [s for s in stems if runs.cache_path(s).exists()]
    print(f"{len(present)} of {len(stems)} caches present; missing: {sorted(set(stems) - set(present)) or 'none'}", flush=True)
    PC = {}
    with Pool(a.procs) as pool:
        for n, (stem, res) in enumerate(pool.imap(cache_task, present)):
            PC[stem] = res
            print(f"  {n + 1}/{len(present)} {stem}", flush=True)

    L_GT = {k: float(np.median([PC[s]["L"][k] for s in runs.TRUTH])) for k in PC[runs.TRUTH[0]]["L"]}
    (WORK / "eval").mkdir(parents=True, exist_ok=True)
    (WORK / "eval/campaign512.json").write_text(json.dumps({"L_GT": L_GT, "per_cache": PC}, indent=1, default=float))
    print(f"wrote {WORK}/eval/campaign512.json")
