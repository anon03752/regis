"""Plot Ising domain growth and the perception-radius comparison.

Curves use campaign512.json. Dots above each panel mark training marginals;
arcs mark training pairs. The ground-truth band spans three simulator runs.
Transport readouts between marginals use a linear mapping from solver time
to simulation time, as defined in rollouts.between_marginals."""
import json
import sys

import numpy as np

from domains.ising.results import runs
from domains.ising.results.runs import HORIZON, MARGINALS
from figstyle import (plt, save, ACCENT, INK, TEXT, MUTED, FAINT, RULE_LINE, BEYOND, A_BAND,
                      STEP_1, STEP_2, TRANS_1, TRANS_2, PAIR_1, PAIR_2)
from workspace import WORK

PC = json.load(open(WORK / "eval/campaign512.json"))["per_cache"]
KMAX = 4000
YLO, YHI, YDATA = 3.0, 900.0, 300.0      # log axis: data live in [YLO, YDATA]; the training-data strip sits above


def curves(stems):
    """(read-out times, L of shape (seeds, times)) on the first stem's read-out times"""
    ks = sorted(int(k) for k in PC[stems[0]]["L"])
    return np.array(ks, float), np.array([[PC[s]["L"][str(k)] for k in ks] for s in stems], float)


# pairs, for the models trained on pairs: the pairs the strip draws as arcs (the DDPM's are examples) and
# its text; the others are trained on the marginals
METHODS = {
    "regis":   dict(label="REGIS", col=STEP_1, curves=curves(runs.REGIS)),
    "control": dict(label="Non-local control", col=STEP_2, curves=curves(runs.CONTROL)),
    "mmsfm":   dict(label="MMSFM", col=TRANS_1, curves=curves(runs.MMSFM), transport=True),
    "mmtsbm":  dict(label="MMtSBM", col=TRANS_2, curves=curves(runs.MMTSBM), transport=True),
    "ddpm":    dict(label="DDPM", col=PAIR_1, curves=curves(runs.DDPM),
                    pairs=([(3, 18), (9, 120), (40, 500), (250, 1000)],
                           "pairs ($t$, $t$+$\\delta t$), $t$+$\\delta t$ ≤ 1000")),
    # the pair from the quench to t = 3 lies left of the axis
    "pairgan": dict(label="Pair-conditional GAN", col=PAIR_2, curves=curves(runs.PAIRGAN),
                    pairs=(list(zip(MARGINALS, MARGINALS[1:])), "pairs of consecutive marginals")),
    # locality ladder: the same rule with a wider perception, parameter-matched to within 0.7 %
    "k5":      dict(label="REGIS, 5×5", col="#A34802", curves=curves(runs.REGIS_K5)),
    "k7":      dict(label="REGIS, 7×7", col="#6E3001", curves=curves(runs.REGIS_K7)),
}
METHODS["k3"] = dict(METHODS["regis"], label="REGIS, 3×3")
TKS, TARR = curves(runs.TRUTH)


def data_strip(ax, m, first, y=430.0):
    """Draw training marginals or example pairs above the growth curve."""
    up = 1.13
    ax.plot([3, HORIZON], [y, y], color=FAINT, lw=0.6, zorder=6, solid_capstyle="butt", clip_on=False)
    if "pairs" not in m:
        ax.plot(MARGINALS, [y] * len(MARGINALS), "o", color=INK, ms=2.6, mew=0, zorder=7, clip_on=False)
        txt = f"{len(MARGINALS)} marginals"
    else:                                     # an arc from t to t' per pair, a dot at t'
        pairs, txt = m["pairs"]
        for a, b in pairs:
            t = np.linspace(0, 1, 30)
            xs = np.exp(np.log(a) + t * (np.log(b) - np.log(a)))
            ax.plot(xs, y * up ** np.sin(np.pi * t), color=INK, lw=0.6, zorder=7, clip_on=False)
            ax.plot([b], [y], "o", color=INK, ms=2.0, mew=0, zorder=7, clip_on=False)
    lead = "trained on: " if first else ""
    ax.text(3, y * 1.3, lead + txt, fontsize=6.2, color=MUTED, va="bottom", ha="left", zorder=7, clip_on=False)


def loglog_panel(ax, m, first):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3, KMAX)
    ax.set_ylim(YLO, YHI)
    ax.spines["left"].set_bounds(YLO, YDATA)
    ymax = np.log(YDATA / YLO) / np.log(YHI / YLO)
    med = np.median(TARR, 0)
    ax.axvspan(HORIZON, KMAX, ymax=ymax, color=BEYOND, lw=0, zorder=0)
    ax.axvline(HORIZON, ymax=ymax, color=RULE_LINE, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.fill_between(TKS, TARR.min(0), TARR.max(0), color=ACCENT, alpha=A_BAND, lw=0, zorder=2)
    ax.plot(TKS, med, color=INK, lw=0.9, ls=(0, (4, 2.5)), zorder=4.7)
    k0, L = m["curves"]
    medL = np.median(L, 0)
    col = m["col"]
    if m.get("transport"):
        a = np.isin(k0, MARGINALS)
        for r in L:
            ax.plot(k0[a], r[a], "o", color=col, ms=3.4, mec=TEXT, mew=0.5, zorder=5)
        ax.plot(k0, medL, color=col, lw=1.3, ls=(0, (2, 1.4)), zorder=4)   # transitions, short dashes
        ax.text(2000, np.sqrt(YLO * YDATA), "no state beyond 1k", fontsize=5.8, color=FAINT, ha="center", va="center", rotation=90)
    else:
        for r in L:
            ax.plot(k0, r, color=col, lw=0.6, alpha=0.55, zorder=4)
        ax.plot(k0, medL, color=col, lw=1.4, zorder=4.5)
    data_strip(ax, m, first)
    ax.set_title(m["label"], pad=3)
    ax.set_xticks([10, 100, 1000, 4000])
    ax.set_xticklabels(["10", "100", "1k", "4k"])
    ax.set_yticks([10, 100])
    ax.set_yticklabels(["10", "100"])
    ax.tick_params(axis="both", which="minor", length=0)


def growth():
    GRID_COLS = [("regis", "control"), ("mmsfm", "mmtsbm"), ("ddpm", "pairgan")]
    fig = plt.figure(figsize=(5.5, 2.97))
    gs = fig.add_gridspec(2, 3, left=0.085, right=0.99, top=0.869, bottom=0.117, wspace=0.12, hspace=0.28)
    axes = {}
    for c, keys in enumerate(GRID_COLS):
        for r, key in enumerate(keys):
            ax = fig.add_subplot(gs[r, c])
            axes[key] = ax
            loglog_panel(ax, METHODS[key], (r, c) == (0, 0))
            if c == 0:
                ax.set_ylabel("domain length $L$")
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("simulation time $t$")
            else:
                ax.set_xticklabels([])
    for label, (c0, c1) in (("Marginals only", (0, 1)), ("Pairs", (2, 2))):
        x0 = axes[GRID_COLS[c0][0]].get_position().x0
        x1 = axes[GRID_COLS[c1][0]].get_position().x1
        fig.text(0.5 * (x0 + x1), 0.980, label, fontsize=8.5, fontweight="semibold", color=INK, ha="center", va="bottom")
        fig.add_artist(plt.Line2D([x0, x1], [0.969, 0.969], transform=fig.transFigure, color=TEXT, lw=0.8, solid_capstyle="butt"))
    save(fig, WORK / "figures", "fig3_ising_growth")


def locality():
    KEYS = ["k3", "k5", "k7", "control"]
    REACH = {"k3": "3 px", "k5": "5 px", "k7": "7 px", "control": "35 px"}
    fig = plt.figure(figsize=(5.5, 2.1))
    gs = fig.add_gridspec(1, 4, left=0.085, right=0.99, top=0.76, bottom=0.20, wspace=0.12)
    for c, key in enumerate(KEYS):
        ax = fig.add_subplot(gs[0, c])
        loglog_panel(ax, METHODS[key], c == 0)
        if c == 0:
            ax.set_ylabel("domain length $L$")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("simulation time $t$")
        ax.text(0.96, 0.05, REACH[key], transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=MUTED)
    save(fig, WORK / "figures", "fig_ising_locality")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["growth", "locality"]):
        {"growth": growth, "locality": locality}[w]()
