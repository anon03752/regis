"""Plot spin-flip probabilities over time, with one panel per model.

Each curve is a neighbour class, pooled over three training seeds. Absolute
probabilities are shown on a log axis, with the analytic heat-bath rates as
dotted references. The ground-truth panel checks the measurement against
those rates. Per-seed comparisons are in fig_rule_values.py.

Reads <work>/eval/flip_table.json.
Usage: python -m domains.ising.results.fig_rule_time
Writes <work>/figures/fig_ising_rules_time.{png,pdf}.
"""
import json

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from domains.ising.instruments.flip_probe import pooled
from domains.ising.results.runs import HORIZON
from figstyle import plt, save, ACCENT, INK, TEXT, MUTED, BEYOND
from workspace import WORK

d = json.loads((WORK / "eval/flip_table.json").read_text())
KS = d["ks"]
R = d["results"]
CLS = [(1, 4), (1, 2), (1, 0), (1, -2), (1, -4)]          # 4/4 .. 0/4 aligned neighbours
ramp = LinearSegmentedColormap.from_list("accent", ["#14264A", ACCENT, "#7391C6"])
CCOL = {c: ramp(i / (len(CLS) - 1)) for i, c in enumerate(CLS)}
NSEED = 3
FAMS = [("Ground truth", ["engine"]),
        ("REGIS", [f"regis_k3 s{i}" for i in range(NSEED)]),
        ("REGIS, 7×7", [f"regis_k7 s{i}" for i in range(NSEED)]),
        ("Non-local control", [f"control_nonlocal s{i}" for i in range(NSEED)]),
        ("Pair-conditional GAN", [f"pairgan s{i}" for i in range(NSEED)])]

YLO, YHI = 5e-7, 1.9
YT = [1e-6, 1e-4, 1e-2, 1]
YTL = ["$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "1"]
XT = [10, 100, 1000, 4000]
XTL = ["10", "100", "1k", "4k"]
# 1/4 (0.982) sits on 0/4 (0.9997): draw it dashed on top
CSTY = {4: dict(lw=1.35), 2: dict(lw=1.35), 0: dict(lw=1.35),
        -2: dict(lw=1.0, ls=(0, (2.4, 2.0)), zorder=5), -4: dict(lw=2.0)}
# colour bar, one swatch per class, with guides from the ground-truth panel's heat-bath levels
KEYLAB = {4: "4/4", 2: "3/4", 0: "2/4", -2: "1/4", -4: "0/4"}


def prob(members, t, c):
    """pooled P(flip | class, t): both spin signs of the class, all seeds of the family"""
    n, f = pooled((R[m]["per_t"][str(t)] for m in members), *c)
    return f / n if n else np.nan


NP = len(FAMS)
L, Rt, TOP, BOT = 0.093, 0.972, 0.800, 0.235   # Rt leaves the last, longest title room past its panel
# the gap after the ground-truth panel is wide enough for the guides, the bar and its labels; the others
# are ordinary panel gaps. Panel width follows, so the five panels stay equal.
GAP_S, GAP_B, LEAD, BARW = 0.022, 0.100, 0.040, 0.018
WP = (Rt - L - GAP_B - (NP - 2) * GAP_S) / NP
XS = []
_x = L
for _i in range(NP):
    XS.append(_x)
    _x += WP + (GAP_B if _i == 0 else GAP_S)
BAR_X = XS[0] + WP + LEAD
fig = plt.figure(figsize=(5.5, 1.75))
axes = [fig.add_axes([x, BOT, WP, TOP - BOT]) for x in XS]
for ax, (fam, members) in zip(axes, FAMS):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(False)
    ax.axvspan(HORIZON, KS[-1], color=BEYOND, zorder=0)   # Beyond the last training marginal.
    for c in CLS:                             # Analytic heat-bath probabilities.
        ax.axhline(d["formula"][f"{c[0]},{c[1]}"], color=CCOL[c], lw=0.7, ls=(0, (1.6, 1.6)), zorder=2)
    for c in CLS:
        ys = np.array([prob(members, t, c) for t in KS])
        ax.plot(KS, np.where(ys > 0, ys, np.nan), "-", color=CCOL[c], **{"zorder": 4, **CSTY[c[1]]})
        for t, y in zip(KS, ys):              # Mark zero probabilities at the lower edge of the log axis.
            if y == 0:
                ax.plot(t, YLO * 1.35, "v", color=CCOL[c], ms=3.0, mec=TEXT, mew=0.35, zorder=6)
    ax.set_title(fam, loc="center", pad=4.5, fontsize=6.6)    # the long names are wider than a panel at 7 pt
    ax.set_xlim(KS[0], KS[-1])
    ax.set_ylim(YLO, YHI)
    ax.set_xticks(XT)
    ax.set_xticklabels(XTL, fontsize=6.4)
    ax.set_yticks(YT)
    ax.set_yticklabels(YTL if ax is axes[0] else [], fontsize=6.4)
    ax.tick_params(which="minor", length=0)
axes[0].set_ylabel("Flip probability", fontsize=7.2)
# Discrete colour key for the five neighbour classes.
cax = fig.add_axes([BAR_X, BOT, BARW, TOP - BOT])
cax.imshow(np.arange(len(CLS)).reshape(-1, 1), aspect="auto", origin="lower", interpolation="none",
           cmap=ListedColormap([CCOL[c] for c in CLS]))
cax.set_xticks([])
cax.set_yticks([])
for s in cax.spines.values():
    s.set_visible(False)
_lo, _hi = np.log10(YLO), np.log10(YHI)
for i, c in enumerate(CLS):                      # CLS runs 4/4 .. 0/4, so row i is class CLS[i]
    yf = (i + 0.5) / len(CLS)
    cax.text(1.42, yf, KEYLAB[c[1]], transform=cax.transAxes, fontsize=5.6, color=INK,
             va="center", ha="left", clip_on=False)
    # guide from the heat-bath level of each class to its swatch
    y0 = BOT + (np.log10(d["formula"][f"{c[0]},{c[1]}"]) - _lo) / (_hi - _lo) * (TOP - BOT)
    y1 = BOT + yf * (TOP - BOT)
    fig.add_artist(plt.Line2D([XS[0] + WP + 0.004, BAR_X], [y0, y1], transform=fig.transFigure,
                              color=CCOL[c], lw=0.5, alpha=0.85, solid_capstyle="butt", zorder=1))
fig.text(BAR_X + 0.5 * BARW + 0.012, TOP + 0.030, "Aligned\nneighbours", fontsize=5.8, color=MUTED,
         ha="center", va="bottom", linespacing=1.2)
fig.text(0.5 * (L + Rt), 0.038, "Simulation time $t$", ha="center", va="bottom", fontsize=7.4, color=TEXT)
save(fig, WORK / "figures", "fig_ising_rules_time")
