"""Write the main and per-seed MNIST Frechet-distance tables.

The main table reports seed medians and ranges. The per-seed table reports
half-widths of the 95% bootstrap intervals from fd_ci.json. A seed is
excluded from the first stored horizon with max |pixel| > 10 onward."""
import argparse
import json

import numpy as np

from workspace import WORK
from domains.mnist.results import runs

# parameters evaluated to generate a sample, and ms per trajectory-transition at
# batch 1024 on one H100. MMtSBM samples with its forward EMA network alone;
# MMSFM evaluates a drift and a score network at every solver step.
COST = {"REGIS": (r"88\,K", "0.117"), "Non-local control": (r"83\,K", "0.116"),
        "MMSFM": (r"18.6\,M", "16.2"), "MMtSBM": (r"3.6\,M", "1.07")}
TABLE_NAME = {"REGIS": "REGIS (ours)"}
DIVERGED = 10.0     # largest absolute pixel at a kept transition; the seed is dashed from then on
CELL = r"\providecommand{\fdcell}[2]{\begin{tabular}[t]{@{}c@{}}#1\\[-2.5pt]{\scriptsize #2}\end{tabular}}"
DASH = r"\textemdash"


def diverged_at(path):
    with np.load(path) as f:
        over = [int(k) for k, v in zip(f["keep"], f["vmax"]) if v > DIVERGED]
    return over[0] if over else None


def fmt(x, point=None):
    """Use one decimal below 100, otherwise a comma-separated integer.

    When point is supplied, use its magnitude to choose the interval's precision."""
    return f"{x:.1f}" if (x if point is None else point) < 100 else f"{x:,.0f}".replace(",", "{,}")


def bf(s):
    return rf"\textbf{{{s}}}"


def seeds_table(rows, H):
    """tab:horizon-seeds: a row per seed, the half-width of its bootstrap interval beside each distance."""
    L = [CELL, r"\begin{table}[htbp]", r"    \centering", r"    \small", r"    \setlength{\tabcolsep}{1.5pt}",
         r"    \begin{tabular}{@{}l r c l c c c c c@{}}", r"    \toprule",
         r"    & & & & \multicolumn{5}{c}{Digit transitions completed} \\", r"    \cmidrule(l){5-9}",
         r"    Model & Parameters & ms\,/\,transition & Seed & " + " & ".join(f"${h}$" for h in H) + r" \\", r"    \midrule"]
    for i, name in enumerate(rows):
        if i:
            L.append(r"    \addlinespace")
        b, nm, (par, ms) = bf if name == "REGIS" else str, TABLE_NAME.get(name, name), COST[name]
        for j, (seed, cut, c) in enumerate(rows[name]):
            cells = []
            for h in H:
                if cut is not None and h >= cut:
                    cells.append(DASH)
                    continue
                e = c[str(h)]
                cells.append(b(fmt(e["fid"])) + r"\,{\scriptsize$\pm$\," + fmt((e["hi"] - e["lo"]) / 2, e["fid"]) + "}")
            head = f"{b(nm)} & {par} & {ms}" if j == 0 else " & &"
            L.append(f"    {head} & {seed} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"    \end{tabular}",
          r"    \caption{\textbf{Per-seed results behind Table~\ref{tab:horizon}.} "
          r"$\pm$: half-width of a $95\%$ percentile bootstrap interval \citep{efron1993bootstrap} over the "
          r"$5{,}000$ generated samples, the real reference held fixed. "
          r"\textemdash: the seed has diverged by this horizon.}",
          r"    \label{tab:horizon-seeds}", r"    \end{table}"]
    return "\n".join(L) + "\n"


def main_table(rows, H, floor):
    """tab:horizon: per method, the median over seeds with the seed range beneath; a dagger where a
    diverged seed is left out."""
    M = [CELL, r"\begin{table}[htbp]", r"    \centering", r"    \small",
         r"    \begin{tabular}{@{}l r c c c c c@{}}", r"    \toprule",
         r"    & & \multicolumn{5}{c}{Digit transitions completed} \\", r"    \cmidrule(l){3-7}",
         r"    Model & Parameters & " + " & ".join(f"${h}$" for h in H) + r" \\", r"    \midrule"]
    for name in rows:
        b, nm, cells = bf if name == "REGIS" else str, TABLE_NAME.get(name, name), []
        for h in H:
            v = [c[str(h)]["fid"] for _seed, cut, c in rows[name] if cut is None or h < cut]
            if not v:                   # every seed has diverged
                cells.append(DASH)
                continue
            dagger = len(v) < len(rows[name])
            top = fmt(float(np.median(v))) + (r"$^\dagger$" if dagger else "")
            cells.append(rf"\fdcell{{{b(top)}}}{{{fmt(min(v))}--{fmt(max(v))}}}")
        M.append(f"    {b(nm)} & {COST[name][0]} & " + " & ".join(cells) + r" \\")
    M += [r"    \bottomrule", r"    \end{tabular}",
          r"\caption{\textbf{Image quality over repeated digit transitions.} "
          r"Fr\'echet distance in a frozen MNIST classifier's feature space (lower is better), using $5000$ "
          r"generated and $5000$ real samples at each horizon. Generated trajectories are balanced over starting "
          r"classes. Cells: median over seven seeds, with the range across seeds beneath. "
          rf"For scale, two disjoint samples of $5000$ real images give {floor['mean']:.2f} "
          rf"($95\%$ of draws between {floor['lo']:.2f} and {floor['hi']:.2f}). "
          r"$\dagger$: at least one seed diverged and is excluded from the median and the range. "
          r"Full per-seed results appear in Table~\ref{tab:horizon-seeds}.}",
          r"    \label{tab:horizon}", r"    \end{table}"]
    return "\n".join(M) + "\n"


def main():
    d = json.loads((WORK / "eval" / "mnist" / "fd_ci.json").read_text())
    fl, H = d["floor"], d["horizons"]
    out = WORK / "figures" / "mnist"
    out.mkdir(parents=True, exist_ok=True)
    rows = {name: [(runs.shown_seed(s), diverged_at(runs.capture(s)), d["cells"][s])
                   for s in stems if s in d["cells"]] for name, stems in runs.METHODS}
    # a method with no captures is left out: a row of dashes would read as divergence
    rows = {name: r for name, r in rows.items() if r}
    assert all(len(r) == len(runs.SEEDS) for r in rows.values()), "a method is missing seeds"
    (out / "table_supp.tex").write_text(seeds_table(rows, H))
    (out / "horizon.tex").write_text(main_table(rows, H, fl))
    print(f"floor {fl['mean']:.2f}  95% [{fl['lo']:.2f}, {fl['hi']:.2f}]\n-> {out}/horizon.tex, table_supp.tex")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
