"""Plot L/L_GT at t = 4000 against the number of training examples.

An example is one image for REGIS (5F images for F fields per marginal)
or one image pair for DDPM. Points are individual seeds; the line joins
seed medians. Off-scale points appear at the axis edge, and off-scale
medians are labelled with their values. Reads eval/campaign512.json."""
import json

import numpy as np

from domains.ising.results import runs
from figstyle import plt, save, INK, STEP_1, PAIR_1
from workspace import WORK

J = json.load(open(WORK / "eval/campaign512.json"))
PC, L_GT = J["per_cache"], J["L_GT"]
T = "4000"
YLO, YHI = 0.60, 1.16

fig, ax = plt.subplots(figsize=(3.05, 2.15))
fig.subplots_adjust(left=0.175, right=0.985, top=0.965, bottom=0.215)
ax.set_xscale("log")
ax.set_xlim(36, 26000)
ax.set_ylim(YLO, YHI)
ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
for ladder, col, mk in ((runs.LADDER_REGIS, STEP_1, "o"), (runs.LADDER_DDPM, PAIR_1, "s")):
    xs = sorted(ladder)
    per_seed = [[PC[s]["L"][T] / L_GT[T] for s in ladder[x]] for x in xs]
    med = [np.median(vs) for vs in per_seed]
    ax.plot(xs, med, color=col, lw=1.5, zorder=4)   # runs off the top where the median is off scale
    for x, vs in zip(xs, per_seed):
        for v in vs:
            if YLO <= v <= YHI:
                ax.plot(x, v, mk, color=col, ms=3.4, mew=0.35, mec="white", alpha=0.95, zorder=5)
            else:                                        # off scale: a faint marker on the edge
                ax.plot(x, YHI if v > YHI else YLO, mk, color=col, ms=3.0, mew=0.35, mec="white",
                        clip_on=False, zorder=5, alpha=0.6)
    for x, m in zip(xs, med):
        if not (YLO <= m <= YHI):
            ax.annotate(f"{m:.2f}", (x, YHI), xytext=(0, 2.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.0, color=col, annotation_clip=False)
ax.text(20000, 1.045, "REGIS", color=STEP_1, fontsize=6.8, fontweight="semibold", ha="right", va="bottom")
ax.text(20000, 0.945, "DDPM", color=PAIR_1, fontsize=6.8, fontweight="semibold", ha="right", va="top")
ax.text(38, 1.0, "ground truth", color=INK, fontsize=6.2, ha="left", va="bottom")
ax.set_xticks([50, 150, 500, 1500, 5000, 15000])
ax.set_xticklabels(["50", "150", "500", "1.5k", "5k", "15k"])
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
ax.minorticks_off()
ax.set_xlabel("training examples (images or pairs)")
ax.set_ylabel("$L/L_{\\mathrm{GT}}$ at $t = 4000$")
save(fig, WORK / "figures", "fig3_ising_ladder")
