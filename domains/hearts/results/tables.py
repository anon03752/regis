"""The heart chapter's four appendix tables, from the instrument caches.

No cell is retyped and no summary line is computed by hand: per-run rows, family
summaries and the across-seed sign tests all come out of `results/stats.py`
reading `eval/hearts/*.json`, and the variability table from the cohort itself.
A run with no cache is named and skipped, so the tables build from a partial set
of runs and simply carry fewer rows.

    python -m domains.hearts.results.tables

Writes <workspace>/figures/hearts/ table_heart_model_fidelity.tex (the group
recovery and fidelity tables), table_heart_aux_ablation.tex,
table_heart_celltype_grouping.tex and table_heart_variabiltiy.tex -- the files
the paper includes, under the paper's spelling.
"""
import argparse

import numpy as np

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts.results import runs
from domains.hearts.results import stats as S

OUT = WORK / "figures/hearts"

GROUP_RECOVERY_CAPTION = (
    r"\textbf{A-priori group recovery.} Knock-out effects $\rho$ (\cref{sec:app-hearts-eval}) "
    r"compared between the pre-specified groups of \cref{tab:heart-group-support}: \emph{A} "
    r"regeneration programme, \emph{B} regulatory stroma, \emph{C} blood, \emph{D} constitutive "
    r"structure. Pairs: the share of cross-group pairs, $U/(n_1 n_2)$, in which the cell type from "
    r"the first group has the larger effect ($50\%$ is the null, $100\%$ complete separation); $p$ "
    r"is reported for the one-sided Mann--Whitney in the pre-specified direction. The italic line "
    r"under each family gives the mean $\pm$ s.d.\ over its seeds and a one-sided sign test across "
    r"seeds against $50\%$;.")
FIDELITY_CAPTION = (
    r"\textbf{Trajectory fidelity and cascade retention.} Temporal $r$ and cascade retention as "
    r"defined in \cref{sec:app-hearts-eval}. Temporal $r$ is the median over cell types, for all "
    r"cell types and for the five BZm states. Rows are individual training seeds; the italic line "
    r"under each family is the mean $\pm$ s.d.\ over its seeds. A directional cascade shows low "
    r"downstream retention, with upstream and control retention near $100\%$. MMtSBM is shown in "
    r"its \emph{one-step} and \emph{chained} modes (\cref{sec:app-hearts-transport}); knock-out "
    r"metrics are not reported due to the poor fit of \emph{chained} model.")
AUX_CAPTION = (
    r"\textbf{Auxiliary-loss ablations.} Each row removes one auxiliary term and retrains from "
    r"scratch with a single seed; all other terms, the architecture, the schedule and the data are "
    r"unchanged. The homeostasis and hidden-norm terms were not ablated. The first row is the REGIS "
    r"seed with the lowest downstream retention (seed 0 in \cref{tab:app-heart-fidelity}); across "
    r"the seven REGIS seeds, downstream retention ranges over $13$--$47\%$ and the \emph{A}+\emph{B} "
    r"vs \emph{C}+\emph{D} contrast over $63$--$90\%$. Metrics as in \cref{sec:app-hearts-eval}; "
    r"$\pm$ in the retention columns is over (blocked, affected) state pairs within one run")

# full names and the loss-of-function literature each implicated type rests on
GROUPING = [
    ("BZm (activ.)", "Border-zone myocardium, activated", r"\citep{Jopling2010,Kikuchi2010}"),
    ("BZm (dediff.)", "Border-zone myocardium, dedifferentiating", r"\citep{Jopling2010,Kikuchi2010}"),
    ("BZm (dediff.-prolif.)", "Border-zone myocardium, dedifferentiating and proliferating",
     r"\citep{Jopling2010,Honkoop2019}"),
    ("BZm (prolif.)", "Border-zone myocardium, proliferating", r"\citep{Jopling2010,Honkoop2019}"),
    ("BZm (re-diff.)", "Border-zone myocardium, re-differentiating", r"\citep{Jopling2010,Sallin2015}"),
    ("MC", "Macrophages", r"\citep{Lai2017,Wei2023}"),
    ("FB (reg.)", "Fibroblasts, pro-regenerative", r"\citep{Hu2022,SanchezIranzo2018}"),
    ("MFE", "Mixed immune, fibroblast and endocardial", r"\citep{Lai2017,Hu2022,Kikuchi2011}"),
    ("Epi", "Epicardium", r"\citep{Kikuchi2011,Sun2022}"),
    ("RBC", "Erythrocytes, wound", "---"),
    ("RBC (V)", "Erythrocytes, ventricular", "---"),
    ("RBC (A)", "Erythrocytes, atrial", "---"),
    ("RZm1", "Remote-zone myocardium 1", "---"),
    ("RZm2", "Remote-zone myocardium 2", "---"),
    ("Vm (comp.)", "Ventricular compact myocardium", "---"),
    ("Am", "Atrial myocardium", "---"),
    ("VAm", "Ventricular and atrial shared myocardium", "---"),
    ("SMC", "Smooth muscle, bulbus arteriosus", "---"),
    ("Valves", "Cardiac valves", "---"),
]
GROUPING_CAPTION = (
    r"\textbf{A-priori grouping of the 19 cell types.} Groups \textit{A} and \textit{B} form the "
    r"implicated side of the contrast, \textit{C} and \textit{D} the unimplicated side. The source "
    r"atlas \citep{Li2025-tf} only defines the cell types; the evidence for the assignment comes "
    r"from other sources. The assignment was fixed before any model results were seen. Absence of "
    r"evidence is not evidence of absence; the contrast therefore tests only the weaker hypothesis "
    r"that cell types with established roles in regeneration perturb the outcome detectably more "
    r"than those without. The erythrocyte populations (RBC) may partly mark vasculature, which has "
    r"been shown to also play role in the regeneration \citep{MarinJuez2016}; endothelium is not "
    r"annotated separately here, so RBC are put into the unimplicated group with that caveat.")

# compartments of the variability table; with `residual` they partition the heart
COMPARTMENTS = [
    ("Ventricle", ["RZm1", "RZm2", "Vm (comp.)", "BZm (activ.)", "BZm (dediff.)",
                   "BZm (dediff.-prolif.)", "BZm (prolif.)", "BZm (re-diff.)", "RBC (V)"]),
    ("Atrium", ["Am", "RBC (A)"]),
    ("Bulb (SMC)", ["SMC"]),
    ("Valves", ["Valves"]),
    ("Unassigned", ["Others"]),
    ("Residual", ["VAm", "RBC", "MC", "FB (reg.)", "MFE", "Epi"]),
]
VARIABILITY_CAPTION = (
    r"Sample diversity of the Stereo-seq data ({n} sections from {h} hearts). Size is the summed "
    r"channel value over the section's occupied grid bins (cell units). Chambers include the blood "
    r"in their own lumen; residual collects VAm, unqualified RBC and the reactive types (MC, FB "
    r"(reg.), MFE, Epi), so the rows sum to the global size. CV is the coefficient of variation "
    r"across sections.")


def cache(kind, stem):
    return WORK / f"eval/hearts/{kind}_{stem}.json"


def have(stem, kinds=("knockout", "cascade", "marginals")):
    return all(cache(k, stem).exists() for k in kinds)


def _p(x):
    return r"$<$0.001" if x < 1e-3 else f"{x:.3f}"


def _seed(stem):
    if "loo" in stem:
        return stem[len("hearts_regis_loo_"):].replace("_", r"\_")
    return "seed " + stem[stem.rindex("seed") + 4:].split("_")[0]


def _bold(text, on):
    return rf"\mathbf{{{text}}}" if on else text


# the evaluation of record for MMSFM takes the median over the 19 cell types, without the catch-all
TYPES = {s: [c for c in C.CELL_TYPES if c != C.NOT_A_CELL_TYPE] for s in runs.MMSFM}
# MMtSBM has profiles only, in its two modes (instruments/mmtsbm.py)
BRIDGES = [("MMtSBM, one-step", f"{runs.MMTSBM[0]}_onestep"), ("MMtSBM, chained", f"{runs.MMTSBM[0]}_chained")]


def fidelity_row(stem):
    m = cache("marginals", stem)
    r = {"r_all": S.temporal_r(m, TYPES.get(stem, C.CELL_TYPES)), "r_bz": S.temporal_r(m, C.CASCADE)}
    if not cache("cascade", stem).exists():
        return r
    ret = S.retention(cache("cascade", stem))
    return {**r, **{k: v[0] for k, v in ret.items()}, "sd": {k: v[1] for k, v in ret.items()}}


def group_recovery(families):
    res = {s: S.contrasts(cache("knockout", s)) for _n, st in families for s in st}
    fam = {n: [s for s in st] for n, st in families}
    summ = {n: [S.summarise([res[s][k][0] * 100 for s in st], ddof=0) for k in ("AB_CD", "A_D")]
            for n, st in fam.items() if len(st) > 1}
    sign = {n: [S.sign_test([res[s][k][0] for s in fam[n]]) for k in ("AB_CD", "A_D")] for n in summ}
    best = [max(summ, key=lambda n: summ[n][i][0]) if summ else None for i in (0, 1)]
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{" + GROUP_RECOVERY_CAPTION + "}", r"\label{tab:app-heart-group-recovery}",
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Model & Seed & \multicolumn{2}{c}{$\it{A}$+$\it{B}$ vs $\it{C}$+$\it{D}$} & "
         r"\multicolumn{2}{c}{$\it{A}$ vs $\it{D}$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-6}", r"& & pairs & $p$ & pairs & $p$ \\", r"\midrule"]
    for fi, (name, stems) in enumerate(fam.items()):
        for i, s in enumerate(stems):
            (ab, pab), (ad, pad) = res[s]["AB_CD"], res[s]["A_D"]
            L.append(f"{name if i == 0 else ''} & {_seed(s)} & {ab * 100:.0f}\\% & {_p(pab)} & "
                     f"{ad * 100:.0f}\\% & {_p(pad)} \\\\")
        if name in summ:
            cells = [r"\emph{$" + _bold(f"{mu:.0f} \\pm {sd:.0f}", best[i] == name) + r"$\%} & "
                     + (rf"\textbf{{{sign[name][i]:.3f}}}" if best[i] == name else f"{sign[name][i]:.3f}")
                     for i, (mu, sd) in enumerate(summ[name])]
            L.append(r"\quad \emph{across seeds} & & " + " & ".join(cells) + r" \\")
        if fi < len(fam) - 1:
            L.append(r"\addlinespace")
    return L + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def fidelity(families, bridges):
    rows = {s: fidelity_row(s) for _n, st in families for s in st}
    fam = dict(families)
    summ = {n: {k: S.summarise([rows[s][k] for s in st], ddof=1)
                for k in ("r_all", "r_bz", "downstream", "upstream", "control")}
            for n, st in fam.items() if len(st) > 1}
    best = {"r_all": max(summ, key=lambda n: summ[n]["r_all"][0]),
            "r_bz": max(summ, key=lambda n: summ[n]["r_bz"][0]),
            "downstream": min(summ, key=lambda n: summ[n]["downstream"][0])} if summ else {}
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{" + FIDELITY_CAPTION + "}", r"\label{tab:app-heart-fidelity}",
         r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Model & Seed & \multicolumn{2}{c}{Median temporal $r$} & "
         r"\multicolumn{3}{c}{Retained after one knock-out (\%)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-7}",
         r"& & all cell types & BZm cascade & downstream & upstream & control \\", r"\midrule"]
    for fi, (name, stems) in enumerate(fam.items()):
        for i, s in enumerate(stems):
            r = rows[s]
            L.append(f"{name if i == 0 else ''}{' ' if i == 0 else ''}& {_seed(s)} & {r['r_all']:.3f} & {r['r_bz']:.3f} & "
                     f"{r['downstream']:.0f} & {r['upstream']:.0f} & {r['control']:.0f} \\\\")
        if name in summ:
            fmt = {"r_all": "{:.3f} \\pm {:.3f}", "r_bz": "{:.3f} \\pm {:.3f}"}
            cells = [r"\emph{$" + _bold(fmt.get(k, "{:.0f} \\pm {:.0f}").format(*summ[name][k]),
                                        best.get(k) == name) + "$}"
                     for k in ("r_all", "r_bz", "downstream", "upstream", "control")]
            L.append(r"\quad \emph{mean $\pm$ s.d.} & & " + " & ".join(cells) + r" \\")
        if fi < len(fam) - 1 or bridges:
            L.append(r"\addlinespace")
    for label, stem in bridges:
        r = fidelity_row(stem)
        L.append(f"{label} & {_seed(stem)} & {r['r_all']:.3f} & {r['r_bz']:.3f} & --- & --- & --- \\\\")
    if bridges:
        L.append(r"\addlinespace")
    return L + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def aux_ablation(stems):
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{" + AUX_CAPTION + "}", r"\label{tab:app-heart-aux-ablations}",
         r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{Median temporal $r$} & \multicolumn{3}{c}{Cascade retention (\%)} & "
         r"\multicolumn{2}{c}{$\it{A}$+$\it{B}$ vs $\it{C}$+$\it{D}$} & $\it{A}$ vs $\it{D}$ \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(l){9-9}",
         r"Terms active & all & BZm & down & up & control & pairs & $p$ & pairs \\", r"\midrule"]
    for i, (label, s) in enumerate(stems):
        r, ct = fidelity_row(s), S.contrasts(cache("knockout", s))
        ref = i == 0                                  # the reference REGIS row
        ret = []
        for k in ("downstream", "upstream", "control"):
            cell = f"{r[k]:.0f} \\pm {r['sd'][k]:.0f}"
            ret.append("$" + _bold(cell, ref and k == "downstream") + "$")
        ab, pab = ct["AB_CD"]
        L.append(f"{label} & {r['r_all']:.3f} & {r['r_bz']:.3f} & " + " & ".join(ret) + " & "
                 + (f"$\\mathbf{{{ab * 100:.0f}}}$\\% & \\textbf{{{_p(pab)}}}" if ref
                    else f"{ab * 100:.0f}\\% & {_p(pab)}")
                 + f" & {ct['A_D'][0] * 100:.0f}\\% \\\\")
        if ref:
            L.append(r"\addlinespace")
    return L + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def grouping():
    got = {C.CELL_TYPES[[c.split(":", 1)[1] for c in C.CELL_TYPES].index(n)] for n, _f, _s in GROUPING}
    assert got == set(C.GROUP) - {C.NOT_A_CELL_TYPE}, "the grouping table must list every scored type"
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{" + GROUPING_CAPTION + "}", r"\label{tab:heart-group-support}",
         r"\begin{tabularx}{\linewidth}{@{}l", r">{\raggedright\arraybackslash}X", "c", "c",
         r">{\raggedright\arraybackslash}X@{}}", r"\toprule",
         r"Cell type & Full name & Group & Implicated & Support \\", r"\midrule"]
    prev = None
    for name, full, cite in GROUPING:
        g = next(C.GROUP[c] for c in C.GROUP if c.split(":", 1)[1] == name)
        if prev and g != prev:
            L.append(r"\addlinespace")
        L.append(f"{name} & {full} & \\textit{{{g}}} & {'Yes' if g in 'AB' else 'No'} & {cite} \\\\")
        prev = g
    return L + [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]


def variability():
    idx = {c.split(":", 1)[1]: i for i, c in enumerate(C.CELL_TYPES)}
    rows, hearts = [], set()
    for sec in C._read(C.PATH):
        m = sec["mask"][0] > 0
        cells = sec["composition"][:len(C.CELL_TYPES)][:, m]
        rows.append({"global": float(cells.sum()),
                     **{n: float(sum(cells[idx[t]].sum() for t in ts)) for n, ts in COMPARTMENTS}})
        t, _chip, fish = sec["section_id"].split("_")
        hearts.add(f"{t}_{fish}")
    cv = lambda v: 100 * np.std(v, ddof=1) / np.mean(v)
    L = [r"\begin{table}[t]", r"  \centering",
         r"  \caption{" + VARIABILITY_CAPTION.format(n=len(rows), h=len(hearts)) + "}",
         r"  \label{tab:heart-div}", r"  \begin{tabular}{lrrrrr}", r"    \toprule",
         r"    & \multicolumn{3}{c}{Absolute (cell units)} & \multicolumn{2}{c}{Relative to global} \\",
         r"    \cmidrule(lr){2-4}\cmidrule(lr){5-6}", r"    Compartment & Mean & SD & CV & Mean & CV \\",
         r"    \midrule"]
    g = np.array([r["global"] for r in rows])
    L.append(f"    Global & {g.mean():.0f} & {g.std(ddof=1):.0f} & {cv(g):.1f}\\% & -- & -- \\\\")
    L.append(r"    \midrule")
    for n, _ts in COMPARTMENTS:
        v = np.array([r[n] for r in rows]); rel = v / g
        L.append(f"    {n} & {v.mean():.0f} & {v.std(ddof=1):.0f} & {cv(v):.1f}\\% & "
                 f"{rel.mean():.3f} & {cv(rel):.1f}\\% \\\\")
    return L + [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    families = [(n, [s for s in st if have(s)]) for n, st in runs.METHODS]
    skipped = [s for _n, st in runs.METHODS for s in st if not have(s)]
    families = [(n, st) for n, st in families if st]
    mmsfm = [s for s in runs.MMSFM if have(s)]
    bridges = [(lbl, s) for lbl, s in BRIDGES if have(s, ("marginals",))]
    skipped += [s for s in runs.MMSFM if s not in mmsfm] + [s for _l, s in BRIDGES if not have(s, ("marginals",))]
    aux = [("All six (REGIS)", runs.REGIS[0])] + [(rf"$-$ {loss}", s) for loss, s in runs.AUX.items()]
    aux = [(lbl, s) for lbl, s in aux if have(s)]
    skipped += [s for s in runs.AUX.values() if not have(s)]
    if skipped:
        print(f"no cache for {len(skipped)} run(s), skipped: {', '.join(skipped)}")
    transport = [("MMSFM (cyclic)", mmsfm)] if mmsfm else []
    files = {"table_heart_model_fidelity": group_recovery(families + transport) + [""]
             + fidelity(families + [("MMSFM", mmsfm)] * bool(mmsfm), bridges),
             "table_heart_aux_ablation": aux_ablation(aux),
             "table_heart_celltype_grouping": grouping()}
    if C.PATH.exists():
        files["table_heart_variabiltiy"] = variability()
    else:
        print(f"no cohort at {C.PATH}: skipped the variability table")
    for name, lines in files.items():
        (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")
        print(f"wrote {OUT / name}.tex")


if __name__ == "__main__":
    main()
