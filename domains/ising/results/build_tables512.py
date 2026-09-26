"""Write the Ising domain-growth and locality tables from campaign512.json.

Cells show median L/L_GT and seed range. Bold values have mean absolute
deviation from 1 below 2.5% across seeds. Intermediate transport readouts
are solver states between observation times; extrapolation cells are empty."""
import json

import numpy as np

from domains.ising.results import runs
from workspace import WORK

OUT = WORK / "eval/tables512"
J = json.load(open(WORK / "eval/campaign512.json"))
PC, L_GT = J["per_cache"], J["L_GT"]
COLS = ["300", "1000", "150", "700", "2000", "4000"]
BOLD_MAD = 0.025


def ratios(stems):
    """L / L_GT per column, one per seed; None where a cache has no read-out"""
    Ls = [PC[s]["L"] for s in stems]
    return {c: [L[c] / L_GT[c] for L in Ls] if all(c in L for L in Ls) else None for c in COLS}


def cell(v):
    if v is None:
        return "--"
    med, mad = float(np.median(v)), float(np.mean([abs(1 - x) for x in v]))
    m = f"\\textbf{{{med:.2f}}}" if mad < BOLD_MAD else f"{med:.2f}"
    return f"\\isingcell{{{m}}}{{{min(v):.2f}--{max(v):.2f}}}"


def row(name, lead, vals, bold=False, indent=8):
    n = f"\\textbf{{{name}}}" if bold else name
    return " " * indent + f"{n} & {lead} & " + " & ".join(cell(vals[c]) for c in COLS) + " \\\\"


PRE = r"""\providecommand{\isingcell}[2]{\begin{tabular}[t]{@{}c@{}}#1\\[-2.5pt]{\fontsize{6}{7}\selectfont #2}\end{tabular}}
\begin{table}[htbp]
    \centering
    \footnotesize
    \setlength{\tabcolsep}{4pt}
"""
RULESEP = r"""    \setlength{\aboverulesep}{0.5ex}
    \setlength{\belowrulesep}{0.5ex}
"""
HEAD = PRE + r"""    \newenvironment{isingrows}{\begin{tabular}[c]{@{}w{l}{86pt}w{r}{43pt}*{6}{w{c}{31pt}}@{}}}{\end{tabular}}
    \newcommand{\isinggroup}[1]{$\vcenter{\hbox{\rotatebox{90}{\textup{#1}}}}$}
""" + RULESEP + r"""    \begin{tabular}{@{}c@{\hspace{8pt}}c@{}}
    \toprule
    & \begin{isingrows}
        & & \multicolumn{2}{c}{Observed $t$} & \multicolumn{2}{c}{Interpolation $t$} & \multicolumn{2}{c}{Extrapolation $t$} \\
        \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(l){7-8}
        Model & Parameters & 300 & 1{,}000 & 150 & 700 & 2{,}000 & 4{,}000 \\
    \end{isingrows} \\
    \midrule"""

# parameter counts are totals over the networks each method trains: MMSFM is a
# flow head + a score head at 20.9 M each; MMtSBM a forward drift net at 3.6 M
# (its paper runs train no backward drift); update-rule counts are the
# generator alone (the critic is discarded at evaluation)
t1 = [HEAD,
      r"    \isinggroup{Marginals only} & \begin{isingrows}",
      row("REGIS (ours)", "17\\,K", ratios(runs.REGIS), bold=True),
      row("Non-local control", "43\\,K", ratios(runs.CONTROL)),
      row("MMSFM", "41.7\\,M", ratios(runs.MMSFM)),
      row("MMtSBM", "3.6\\,M", ratios(runs.MMTSBM)),
      r"    \end{isingrows} \\", r"    \midrule",
      r"    \isinggroup{Pairs} & \begin{isingrows}",
      row("DDPM", "4.7\\,M", ratios(runs.DDPM), bold=True),
      row("Pair-conditional GAN", "43\\,K", ratios(runs.PAIRGAN)),
      r"    \end{isingrows} \\", r"    \bottomrule", r"    \end{tabular}",
      r"\caption{\textbf{Domain growth relative to the ground truth.} Mean magnetic domain length $L$ divided by that of "
      r"the ground-truth process, at $512^2$ from the same initial configurations, $512$ fields per cell. The ground truth "
      r"is the median of three independent runs of the simulator. Columns: the two largest training marginals (observed), "
      r"two times between them (interpolation) and two beyond the last marginal (extrapolation). Cells: median over three "
      r"training seeds with the seed range beneath. Bold: mean absolute deviation from $1$ across seeds below $2.5\%$. "
      r"Pairs: models with access to paired training data. --: the model has no state past its last marginal.}",
      r"    \label{tab:fig3_ising}", r"\end{table}"]

loc = [PRE + RULESEP + r"""    \begin{tabular}{@{}w{l}{84pt}w{r}{30pt}w{r}{40pt}*{6}{w{c}{29pt}}@{}}
    \toprule
    & & & \multicolumn{2}{c}{Observed $t$} & \multicolumn{2}{c}{Interpolation $t$} & \multicolumn{2}{c}{Extrapolation $t$} \\
    \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(l){8-9}
    Model & Reach & Parameters & 300 & 1{,}000 & 150 & 700 & 2{,}000 & 4{,}000 \\
    \midrule"""]
for nm, reach, par, stems, bold in (("REGIS, $3\\times3$", "3\\,px", "17.4\\,K", runs.REGIS, True),
                                    ("REGIS, $5\\times5$", "5\\,px", "17.5\\,K", runs.REGIS_K5, False),
                                    ("REGIS, $7\\times7$", "7\\,px", "17.6\\,K", runs.REGIS_K7, False),
                                    ("Non-local control", "35\\,px", "43.2\\,K", runs.CONTROL, False)):
    loc.append(row(nm, f"{reach} & {par}", ratios(stems), bold, indent=4))
loc += [r"    \bottomrule", r"    \end{tabular}",
        r"\caption{\textbf{Perception radius and domain growth.} The same measurement as Table~\ref{tab:fig3_ising} for "
        r"REGIS at three perception radii and the non-local control. The three neural cellular automata are "
        r"parameter-matched to within $0.7\%$, so the ladder varies reach, not capacity. Cells: median over three training "
        r"seeds with the seed range beneath; bold as in Table~\ref{tab:fig3_ising}. The non-local control repeats its "
        r"Table~\ref{tab:fig3_ising} row as the non-local end of the ladder.}",
        r"    \label{tab:ising_locality}", r"\end{table}"]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "fig3_ising.tex").write_text("\n".join(t1) + "\n")
(OUT / "ising_locality.tex").write_text("\n".join(loc) + "\n")
print(f"wrote {OUT}/fig3_ising.tex and {OUT}/ising_locality.tex")
