"""Rank of each cell type's knock-out effect, by a-priori group, model by model.

One panel per trained model. The 19 cell types are plotted by RANK of knock-out
effect rather than by effect size, because the models differ in overall
amplitude and rank is what the contrast actually tests -- horizontal position
within a group is arbitrary jitter. Bars are group mean ranks.

Above each panel: the share of implicated-versus-unimplicated pairs ranked the
pre-specified way, where 50% is chance, and its one-sided p.

The claim this figure carries is comparative: only the local, time-free rule
ranks the implicated types above the rest. Neither ablation should, despite
fitting the trajectories at least as well.

    python -m domains.hearts.results.fig_group_recovery

Writes <workspace>/figures/hearts/fig_group_recovery.pdf.
"""
import argparse

import numpy as np

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts.results import runs, style
from domains.hearts.results import stats as S

IMPLICATED = ("A", "B")
SHORT = {"REGIS (ours)": "REGIS", "REGIS + time": "+ time",
         "REGIS non-local": "non-local", "REGIS held-out heart": "held-out"}


def panel(ax, stem, show_y):
    cache = WORK / f"eval/hearts/knockout_{stem}.json"
    rows = [r for r in S.load(cache)["rows"] if r["channel"] != C.NOT_A_CELL_TYPE]
    rho = np.array([r["rho"] for r in rows])
    grp = np.array([C.GROUP[r["channel"]] for r in rows])
    # rank, normalised to [0, 1] with 1 = largest effect. Normalised because the
    # models differ in overall amplitude and the contrast is about order; the
    # axis has no meaningful units, so it carries no ticks, only the 0.5 chance
    # line.
    n = len(rows)
    pct = 1.0 - np.argsort(np.argsort(-rho)) / (n - 1)
    ax.axhline(0.5, color=style.HAIR, lw=0.9, ls=(0, (1.5, 2.2)), zorder=1)
    for x, g in enumerate("ABCD"):
        sel = np.where(grp == g)[0]
        if not sel.size:
            continue
        ax.scatter(x + np.linspace(-0.16, 0.16, sel.size), pct[sel], s=13,
                   color=[style.CELL_COLOUR[rows[i]["channel"]] for i in sel],
                   edgecolor="none", zorder=4)
        style.mean_mark(ax, x, pct[sel].mean(), style.GROUP_COLOUR[g],
                        half=0.26, lw=2.4, zorder=2)
    # the two tested sides, white-underlaid so they stay legible over a disc
    for side, xs in ((IMPLICATED, (0, 1)), (("C", "D"), (2, 3))):
        m = pct[np.isin(grp, side)].mean()
        ax.plot([xs[0] - 0.42, xs[1] + 0.42], [m, m], color="white", lw=2.4, zorder=2.8)
        ax.plot([xs[0] - 0.42, xs[1] + 0.42], [m, m], color=style.TEXT, lw=1.1,
                ls=(0, (2.4, 1.4)), zorder=3)
    share, p = S.contrasts(cache)["AB_CD"]
    ax.set_title(f"{SHORT.get(runs.family(stem), runs.family(stem))} {_seed(stem)}\n"
                 f"{share * 100:.0f}% of pairs, p = {p:.3f}",
                 color=style.INK, fontsize=6.2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(list("ABCD"))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([])
    ax.grid(False)
    if show_y:
        ax.set_ylabel("rank of effect")
    return share


def _seed(stem):
    return (stem[stem.rindex("_") + 1:].replace("seed", "seed ")
            if "seed" in stem else stem[len("hearts_regis_loo_"):])


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    import matplotlib.pyplot as plt

    stems = [s for _n, st in runs.METHODS for s in st
             if (WORK / f"eval/hearts/knockout_{s}.json").exists()]
    assert stems, "no knockout caches; run domains.hearts.instruments.knockout first"
    ncol = min(4, len(stems))
    nrow = -(-len(stems) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(style.TEXT_WIDTH, 1.55 * nrow),
                             squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(stems):
            ax.axis("off")
            continue
        panel(ax, stems[i], show_y=(i % ncol == 0))
    fig.text(0.005, 1.01, "Rank of each cell type's knock-out effect, by a-priori group "
             "(top = largest effect; dashes: the two tested sides)", color=style.INK, fontsize=7.5, va="bottom")
    fig.subplots_adjust(hspace=0.62, wspace=0.22)
    style.save(fig, WORK / "figures/hearts", "fig_group_recovery")
    print(f"  {len(stems)} models")


if __name__ == "__main__":
    main()
