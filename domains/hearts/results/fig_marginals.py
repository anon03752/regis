"""Observed and simulated cell-type profiles through regeneration.

One panel per cell type -- small multiples rather than twenty lines sharing an
axis, because twenty categorical hues cannot be told apart in one frame no
matter how they are chosen. Points are cohort means at the eight sampling
stages with their spread; the curve is the rule's own continuous trajectory,
recorded at every update, so the between-stage behaviour is visible rather than
interpolated by the reader.

Each panel carries its own y scale. Abundances span two orders of magnitude
across types, and a shared scale would flatten every rare population into the
axis -- which is precisely where the cascade lives. The axis is therefore
labelled per panel and no cross-panel magnitude comparison is invited.

Other runs can be overlaid with --compare, which is how the chapter shows that
the time-conditioned ablation and the transport baselines fit the marginals as
well or better while losing the dependency structure. MMtSBM is drawn as its
chained rollout.

    python -m domains.hearts.results.fig_marginals --run hearts_regis_seed0 \\
        --compare hearts_time_seed0 hearts_control_nonlocal_seed0 hearts_mmsfm_seed0 \\
        hearts_mmtsbm_seed0_chained

Writes <workspace>/figures/hearts/fig_marginals_<stem>.pdf.
"""
import argparse

import numpy as np

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts.results import runs, style
from domains.hearts.results import stats as S

NCOL = 4
CHAINED = [f"{s}_chained" for s in runs.MMTSBM]
DASHES = [(0, (1.6, 1.4)), (0, (4, 1.5)), (0, (4, 1.2, 1, 1.2)), (0, (0.6, 0.9))]


def label(stem):
    if stem in runs.MMSFM:
        return "MMSFM"
    if stem in CHAINED:
        return "MMtSBM"
    return {"REGIS (ours)": "REGIS", "REGIS + time": "+ time",
            "REGIS non-local": "non-local"}.get(runs.family(stem), runs.family(stem))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=runs.REGIS[0], choices=runs.ALL)
    p.add_argument("--compare", nargs="*", default=[], choices=runs.ALL + runs.MMSFM + CHAINED)
    a = p.parse_args()
    import matplotlib.pyplot as plt

    d = S.load(WORK / f"eval/hearts/marginals_{a.run}.json")
    alts = [(label(s), S.load(WORK / f"eval/hearts/marginals_{s}.json")) for s in a.compare]
    spl = d["meta"]["steps_per_leg"]
    real, sd = np.array(d["real_mean"]), np.array(d["real_sd"])
    model = np.array(d["model_mean"])
    chans = d["channels"]

    nrow = -(-len(chans) // NCOL)
    fig, axes = plt.subplots(nrow, NCOL, figsize=(style.TEXT_WIDTH, 1.05 * nrow),
                             squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(chans):
            ax.axis("off")
            continue
        c = chans[i]
        col = style.CELL_COLOUR[c]
        t = np.arange(model.shape[0]) / spl            # leg position; the wound at 0
        ax.plot(t, model[:, i], color=col, lw=1.1, solid_capstyle="round",
                label="REGIS", zorder=3)
        for (lbl, alt), dash in zip(alts, DASHES):
            am = np.array(alt["model_mean"])[:, alt["channels"].index(c)]
            ax.plot(np.arange(am.shape[0]) / alt["meta"]["steps_per_leg"],
                    am, color=col, lw=1.0, ls=dash, zorder=2, label=lbl)
        # every stage including the uninjured one at 0, so a type present
        # before injury is not mistaken for one that is induced
        ax.errorbar(np.arange(C.N_STAGES), real[:, i], yerr=sd[:, i], fmt="o",
                    ms=2.2, color=style.INK, ecolor=style.MUTED, elinewidth=0.5,
                    capsize=0, lw=0, zorder=4, label="cohort")
        ax.set_title(style.label(c), color=style.INK, fontsize=6)
        style.stage_axis(ax, labelled=i >= len(chans) - NCOL)
        ax.tick_params(axis="y", labelsize=5.2)
        ax.margins(y=0.16)
    # legend keys in ink: the panels' own hues are too pale to show a dash pattern
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=style.INK, marker="o", ms=2.2, lw=0),
            Line2D([], [], color=style.INK, lw=1.1)]
    keys += [Line2D([], [], color=style.INK, lw=1.0, ls=dash) for _alt, dash in zip(alts, DASHES)]
    fig.legend(keys, ["cohort", "REGIS"] + [lbl for lbl, _ in alts], loc="lower center",
               ncol=len(keys), bbox_to_anchor=(0.5, -0.025), handlelength=2.2)
    fig.text(0.005, 1.0, "Observed and simulated cell-type profiles",
             color=style.INK, fontsize=7.5, va="bottom")
    fig.subplots_adjust(hspace=0.85, wspace=0.34)
    style.save(fig, WORK / "figures/hearts", f"fig_marginals_{a.run}")


if __name__ == "__main__":
    main()
