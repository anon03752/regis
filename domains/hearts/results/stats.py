"""The chapter's statistical core, in one place so tables and figures agree.

Two pre-specified contrasts on the perturbation screen, the cascade retentions,
and the across-seed summary. Nothing here reads a model or a rollout -- it reads
the instrument caches only, so every number in the chapter is recomputed from
the same measurement and no cell is retyped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr

from domains.hearts import cohort as C


def load(path) -> dict:
    return json.loads(Path(path).read_text())


# ---- the two a-priori contrasts --------------------------------------------

def cliffs_delta(a, b) -> float:
    """Cliff's delta: the probability that a random a exceeds a random b, minus
    the reverse. Rescaled by (d + 1) / 2 it IS the share of cross-group pairs
    ranked the pre-specified way, which is the unit the table reports -- 50% is
    the null, 100% complete separation."""
    return float(np.mean(np.sign(np.asarray(a)[:, None] - np.asarray(b)[None, :])))


def contrasts(knockout_cache) -> dict:
    """-> {name: (share_of_pairs, one_sided_p)} for each contrast in
    cohort.CONTRASTS, on one model's screen.

    The catch-all cluster is dropped: it is not a cell type, so knocking it out
    has no biological reading and it should not be averaged into its group.
    Each test uses only that model's own 19 cell types, so no model borrows
    power from another.
    """
    rows = [r for r in load(knockout_cache)["rows"]
            if r["channel"] != C.NOT_A_CELL_TYPE]
    rho = np.array([r["rho"] for r in rows], float)
    grp = np.array([C.GROUP[r["channel"]] for r in rows])
    out = {}
    for name, implicated, rest in C.CONTRASTS:
        a, b = rho[np.isin(grp, implicated)], rho[np.isin(grp, rest)]
        out[name] = ((cliffs_delta(a, b) + 1) / 2,
                     float(mannwhitneyu(a, b, alternative="greater").pvalue))
    return out


def sign_test(shares) -> float:
    """One-sided p that every seed falls on the pre-specified side of the 50%
    null. With seven seeds the smallest attainable value is 0.5^7 = 0.008; with
    three it is 0.125, which is why the ablation families are reported for
    completeness and the between-family tests carry the comparison."""
    s = np.asarray(shares, float)
    k, n = int((s > 0.5).sum()), len(s)
    from math import comb
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


# ---- trajectory fidelity ----------------------------------------------------

def temporal_r(marginals_cache, channels=None) -> float:
    """Median over cell types of the correlation between the model trajectory,
    resampled at the measured stages, and the cohort mean.

    A channel the cohort never varies, or the model never varies, has no
    correlation to report and is skipped rather than counted as zero.
    """
    d = load(marginals_cache)
    real, model = np.array(d["real_mean"]), np.array(d["model_mean"])
    hours = np.array(d["model_hours"])
    pick = [int(np.abs(hours - h).argmin()) for h in d["stage_hours"]]
    names = channels if channels is not None else d["channels"]
    rs = []
    for c in names:
        i = d["channels"].index(c)
        if real[:, i].std() > 1e-9 and model[pick, i].std() > 1e-9:
            rs.append(pearsonr(real[:, i], model[pick, i]).statistic)
    return float(np.median(rs))


# ---- the cascade ------------------------------------------------------------

def retention(cascade_cache) -> dict:
    """-> {'downstream', 'upstream', 'control'}: the peak a border-zone state
    keeps after a knockout, as a percentage of the SAME model's unblocked
    rollout -- so each model is its own control and any amplitude bias cancels.

    If the model has learnt the cascade as a chain, blocking a state suppresses
    those downstream of it (low retention) and leaves those upstream intact
    (near 100%). The bystander knockouts should leave all five intact.
    """
    d = load(cascade_cache)
    casc = [d["channels"].index(c) for c in d["cascade"]]
    ctrl = np.array(d["ctrl"])
    curves = {b["ko"]: np.array(b["curve"]) for b in d["blocks"]}
    peak = lambda cur, js: [cur[:, j].max() / max(ctrl[:, j].max(), 1e-9) * 100
                            for j in js]
    down, up = [], []
    for pos, k in enumerate(casc):
        if k in curves:
            down += peak(curves[k], casc[pos + 1:])
            up += peak(curves[k], casc[:pos])
    bystander = [x for c in d["controls"] if d["channels"].index(c) in curves
                 for x in peak(curves[d["channels"].index(c)], casc)]
    return {"downstream": float(np.mean(down)), "upstream": float(np.mean(up)),
            "control": float(np.mean(bystander))}


def summarise(values) -> tuple[float, float]:
    """mean +/- sample sd over a family's seeds, the form every summary line in
    the tables takes."""
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
