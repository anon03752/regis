"""Plot flip probabilities relative to the analytic heat-bath rates.

Counts are pooled over t <= 1000. Each point is a training seed; bars show
seed medians. Shading gives a binomial 95% interval at the median sample
count, assuming independent sites and the analytic rate. It does not model
spatial correlations. Reads eval/flip_table.json; fig_rule_time.py plots
the time-resolved probabilities."""
import json

import numpy as np
from scipy.stats import binom

from domains.ising.instruments.flip_probe import pooled
from domains.ising.results.runs import HORIZON
from figstyle import plt, save, ACCENT, INK, TEXT, STEP_1, STEP_2, PAIR_2
from workspace import WORK

d = json.loads((WORK / "eval/flip_table.json").read_text())
R = d["results"]
KS = [t for t in d["ks"] if t <= HORIZON]
CLS = [(1, -4), (1, -2), (1, 0), (1, 2), (1, 4)]                  # 0/4 .. 4/4 aligned neighbours
TITLE = {-4: "0/4", -2: "1/4", 0: "2/4 · wall", 2: "3/4", 4: "4/4 · bulk"}
# per-panel log axis: ticks with labels; each axis runs from its first to its last tick, symmetric about 1 in
# log, except the bulk class (1e-3 .. 10)
XL = {-4: ([0.8, 1, 1.25], ["0.8", "1", "1.25"]), -2: ([0.8, 1, 1.25], ["0.8", "1", "1.25"]),
      0: ([1 / 4, 1 / 2, 1, 2, 4], ["1/4", "1/2", "1", "2", "4"]),
      2: ([1 / 16, 1 / 4, 1, 4, 16], ["1/16", "1/4", "1", "4", "16"]),
      4: ([1e-3, 1e-2, 1e-1, 1, 10], ["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1", "10"])}
MODELS = [("REGIS", "regis_k3 s", STEP_1, "3 px"),
          ("REGIS, 7×7", "regis_k7 s", "#4E8F3A", "7 px"),
          ("Non-local control", "control_nonlocal s", STEP_2, "35 px"),
          ("Pair-conditional GAN", "pairgan s", PAIR_2, "35 px")]


def q_formula(c):
    """heat-bath flip probability of class c"""
    return d["formula"][f"{c[0]},{c[1]}"]


def counts(member, c):
    return pooled((R[member]["per_t"][str(t)] for t in KS), *c)


def ratio(member, c):
    n, k = counts(member, c)
    return k / n / q_formula(c)


def band(n, c):
    """exact binomial 95 % interval of the formula's rate at n samples, as a ratio to the formula"""
    q = q_formula(c)
    lo, hi = binom.ppf([0.025, 0.975], int(n), q)
    return lo / n / q, hi / n / q


MEM = {fam: sorted(k for k in R if k.startswith(p)) for fam, p, *_ in MODELS}
V = {fam: {c: np.array([ratio(m, c) for m in MEM[fam]]) for c in CLS} for fam in MEM}
N = {fam: {c: np.array([counts(m, c)[0] for m in MEM[fam]]) for c in CLS} for fam in MEM}
for c in CLS:                                  # calibration: the engine's own ratio must sit inside its band
    b = band(counts("engine", c)[0], c)
    r = ratio("engine", c)
    print(f"engine check {TITLE[c[1]]:14s} ratio {r:.5f}  95 % band [{b[0]:.5f}, {b[1]:.5f}]  {'inside' if b[0] <= r <= b[1] else 'OUTSIDE'}")


def panels():
    fig, axes = plt.subplots(1, len(CLS), figsize=(5.5, 1.95), sharey=True)
    fig.subplots_adjust(left=0.165, right=0.975, top=0.87, bottom=0.19, wspace=0.34)
    ny = len(MODELS)
    for ax, c in zip(axes, CLS):
        ticks, labels = XL[c[1]]
        lo, hi = ticks[0], ticks[-1]
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, ny - 0.4)
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.axvline(1.0, color=INK, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
        for i, (fam, _, col, _) in enumerate(MODELS):
            y = ny - 1 - i
            v = V[fam][c]
            blo, bhi = band(np.median(N[fam][c]), c)
            ax.fill_betweenx([y - 0.42, y + 0.42], max(blo, lo), min(bhi, hi), color=ACCENT, alpha=0.22, lw=0, zorder=0)
            for x, o in zip(v, np.linspace(-0.2, 0.2, len(v))):
                if x < lo or x > hi:                                     # off the panel's scale -> edge arrow
                    left = x < lo
                    ax.plot(-0.06 if left else 1.06, y + o, "<" if left else ">", color=col, ms=3.0,
                            mec=TEXT, mew=0.3, zorder=5, transform=ax.get_yaxis_transform(), clip_on=False)
                else:
                    ax.plot(x, y + o, "o", color=col, ms=2.9, mec="white", mew=0.35, zorder=4)
            med = np.median(v)
            if lo <= med <= hi:
                ax.plot([med, med], [y - 0.34, y + 0.34], color=col, lw=2.0, solid_capstyle="butt", zorder=3)
        ax.set_title(TITLE[c[1]], loc="center", fontsize=7.2, pad=4, color=INK, fontweight="bold" if c[1] == 0 else "semibold")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=5.6)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(ny))
    axes[0].set_yticklabels([f"{m[0]}  ·  {m[3]}" for m in MODELS[::-1]], fontsize=6.4)
    fig.text(0.58, 0.02, "flip rate ÷ heat-bath formula"
             "   (each panel on its own scale; shaded: 95 % binomial CI at the row's sample size)",
             ha="center", va="bottom", fontsize=6.6, color=TEXT)
    save(fig, WORK / "figures", "fig_ising_rules")


panels()
