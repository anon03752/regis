"""The chapter's three appendix tables, built from the instrument caches.

No cell is retyped and no summary line is computed by hand: the per-seed rows,
the family means and the across-seed sign tests all come out of
`domains/hearts/results/stats.py` reading `eval/hearts/*.json`.

    python -m domains.hearts.results.tables

Writes <workspace>/figures/hearts/table_{group_recovery,fidelity,grouping}.tex.
Runs that have no cache are reported as missing and skipped, so the tables can
be built from a partial set of runs and will simply carry fewer rows.
"""
import argparse

import numpy as np

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts.results import runs
from domains.hearts.results import stats as S

CONTRAST_LABEL = {"AB_CD": r"$\it{A}$+$\it{B}$ vs $\it{C}$+$\it{D}$",
                  "A_D": r"$\it{A}$ vs $\it{D}$"}
# the literature each group's membership was fixed from, before the screen ran
SUPPORT = {
    "A": "border-zone CM dedifferentiation and proliferation is the mechanism of "
         "zebrafish heart regeneration",
    "B": "col12a1a+ fibroblast ablation reduces CM proliferation; macrophage "
         "depletion impairs regeneration",
    "C": "no established regenerative function",
    "D": "present before injury; not part of the injury response",
}


def _pt(x):
    return r"$<$0.001" if x < 1e-3 else f"{x:.3f}"


def _agg(vals, dp=0, unit="", bold=False):
    v = np.asarray(vals, float)
    mu = v.mean()
    sd = v.std(ddof=1) if v.size > 1 else 0.0
    body = f"{mu:.{dp}f} \\pm {sd:.{dp}f}"
    if bold:
        body = f"\\mathbf{{{mu:.{dp}f} \\pm {sd:.{dp}f}}}"
    return f"$ {body} ${unit}"


def available(stem):
    return ((WORK / f"eval/hearts/knockout_{stem}.json").exists()
            and (WORK / f"eval/hearts/cascade_{stem}.json").exists()
            and (WORK / f"eval/hearts/marginals_{stem}.json").exists())


def group_recovery(families):
    """The two pre-specified contrasts, model by model."""
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{\textbf{The two a-priori contrasts, model by model.} Effect "
         r"size is the share of cross-group pairs in which the cell type from the "
         r"first group has the larger knock-out effect, so $50\%$ is the null and "
         r"$100\%$ complete separation. $p$ is one-sided, testing the "
         r"pre-specified direction. $\it{A}$ is the regeneration programme, "
         r"$\it{B}$ regulatory stroma, $\it{C}$ blood, $\it{D}$ constitutive "
         r"structure; $\it{A}$+$\it{B}$ are the types independently implicated in "
         r"regeneration. Each test uses that model's own 19 cell types, so no row "
         r"borrows power from another.}",
         r"\label{tab:app-heart-group-recovery}",
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Model & Seed & \multicolumn{2}{c}{" + CONTRAST_LABEL["AB_CD"] +
         r"} & \multicolumn{2}{c}{" + CONTRAST_LABEL["A_D"] + r"} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-6}", r"& & pairs & $p$ & pairs & $p$ \\",
         r"\midrule"]
    for fi, (name, stems) in enumerate(families):
        have = [s for s in stems if available(s)]
        if not have:
            continue
        res = {s: S.contrasts(WORK / f"eval/hearts/knockout_{s}.json") for s in have}
        for i, s in enumerate(have):
            lead = name if i == 0 else ""
            seed = (s[s.rindex("_") + 1:] if "seed" in s else s[len("hearts_regis_loo_"):])
            seed = seed.replace("seed", "seed ").replace("_", r"\_")
            L.append(f"{lead} & {seed} & "
                     + " & ".join(f"{res[s][k][0] * 100:.0f}\\% & {_pt(res[s][k][1])}"
                                  for k in ("AB_CD", "A_D")) + r" \\")
        if len(have) > 1:
            cells = []
            for k in ("AB_CD", "A_D"):
                sh = [res[s][k][0] for s in have]
                cells.append(_agg([v * 100 for v in sh], unit=r"\%", bold=True)
                             + " & " + f"\\textbf{{{S.sign_test(sh):.3f}}}")
            L.append(r"\quad \emph{across seeds} & & " + " & ".join(cells) + r" \\")
        if fi < len(families) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def fidelity(families):
    """Trajectory fidelity and the cascade retentions, model by model."""
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{\textbf{Trajectory fidelity and cascade dependency, model by "
         r"model.} Retention is the peak a border-zone state keeps after one "
         r"cascade state is knocked out, as a percentage of that same model's "
         r"unblocked rollout, so each model is its own control and amplitude bias "
         r"cancels. Temporal $r$ is the correlation between the model trajectory, "
         r"resampled at the measured timepoints, and the cohort mean, taken as the "
         r"median over cell types. The upstream and control columns are reported "
         r"for context and are not the discriminating quantity.}",
         r"\label{tab:app-heart-fidelity}",
         r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Model & Seed & \multicolumn{2}{c}{Median temporal $r$} & "
         r"\multicolumn{3}{c}{Retained after one knock-out (\%)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-7}",
         r"& & all cell types & BZm cascade & downstream & upstream & control \\",
         r"\midrule"]
    for fi, (name, stems) in enumerate(families):
        have = [s for s in stems if available(s)]
        if not have:
            continue
        rows = {}
        for s in have:
            mp = WORK / f"eval/hearts/marginals_{s}.json"
            rows[s] = dict(
                r_all=S.temporal_r(mp, [c for c in C.CELL_TYPES]),
                r_bz=S.temporal_r(mp, C.CASCADE),
                **S.retention(WORK / f"eval/hearts/cascade_{s}.json"))
        for i, s in enumerate(have):
            lead = name if i == 0 else ""
            seed = (s[s.rindex("_") + 1:] if "seed" in s else s[len("hearts_regis_loo_"):])
            seed = seed.replace("seed", "seed ").replace("_", r"\_")
            r = rows[s]
            L.append(f"{lead} & {seed} & {r['r_all']:.3f} & {r['r_bz']:.3f} & "
                     f"{r['downstream']:.0f} & {r['upstream']:.0f} & {r['control']:.0f}"
                     + r" \\")
        if len(have) > 1:
            g = lambda k: [rows[s][k] for s in have]
            L.append(r"\quad \emph{mean $\pm$ s.d.} & & "
                     + _agg(g("r_all"), dp=3) + " & " + _agg(g("r_bz"), dp=3) + " & "
                     + _agg(g("downstream"), bold=True) + " & " + _agg(g("upstream"))
                     + " & " + _agg(g("control")) + r" \\")
        if fi < len(families) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def grouping():
    """The a-priori grouping and what fixed it. Reads no cache: this table IS
    the pre-registration, and it must not depend on any result."""
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{\textbf{The a-priori cell-type grouping.} Membership was fixed "
         r"from published loss-of-function evidence before the screen was run, so "
         r"the group comparison is a test rather than a description.}",
         r"\label{tab:heart-group-support}",
         r"\begin{tabular}{@{}llp{0.42\textwidth}@{}}", r"\toprule",
         r"Group & Cell types & Basis \\", r"\midrule"]
    for g in "ABCD":
        members = ", ".join(k.split(":", 1)[1] for k in C.CELL_TYPES if C.GROUP[k] == g)
        L.append(f"$\\it{{{g}}}$ \\ {C.GROUP_NAME[g]} & {members} & {SUPPORT[g]}" + r" \\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    families = [(n, s) for n, s in runs.METHODS]
    missing = [s for _n, st in families for s in st if not available(s)]
    if missing:
        print(f"no cache for {len(missing)} run(s), skipped: {', '.join(missing)}")
    out = WORK / "figures/hearts"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in [("group_recovery", group_recovery(families)),
                       ("fidelity", fidelity(families)), ("grouping", grouping())]:
        (out / f"table_{name}.tex").write_text(text + "\n")
        print(f"wrote {out / f'table_{name}.tex'}")


if __name__ == "__main__":
    main()
