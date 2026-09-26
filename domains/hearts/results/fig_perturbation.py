"""Perturbation effect by functional group, and the border-zone cascade.

Panel A. One cell type is suppressed throughout the rollout and we measure how
far the spatial arrangement of the REMAINING types moved, in units of the
distance between two unperturbed replicates -- so the dashed line at 1 is "a
perturbation indistinguishable from rerunning the model", not zero. Bars are
group means, whiskers a bootstrap over the member types, circles the individual
types. The comparison is between groups fixed in advance from the literature,
which is what makes it a test.

Panel B. Abundance of the five cardiomyocyte states through regeneration:
unperturbed, then each state suppressed in turn, then two controls. If the rule
has learnt the cascade as a chain, suppressing a state should flatten those
downstream of it and leave those upstream intact; a rule that has only learnt
WHEN each state appears will be unaffected.

    python -m domains.hearts.results.fig_perturbation --run hearts_regis_seed0

Writes <workspace>/figures/hearts/fig_perturbation_<stem>.pdf.
"""
import argparse

import numpy as np

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts.results import runs, style
from domains.hearts.results import stats as S


SHORT = {"A": "regeneration\nprogramme", "B": "regulatory\nstroma",
         "C": "blood /\nunassigned", "D": "constitutive\nstructure"}


def panel_groups(ax, cache):
    rows = [r for r in S.load(cache)["rows"] if r["channel"] != C.NOT_A_CELL_TYPE]
    ax.axhline(1.0, color=style.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.annotate("two unperturbed replicates", (0.012, 1.06),
                xycoords=("axes fraction", "data"), ha="left", va="bottom",
                fontsize=5.5, color=style.MUTED)
    for x, g in enumerate("ABCD"):
        member = [r for r in rows if C.GROUP[r["channel"]] == g]
        if not member:
            continue
        v = np.array([r["rho"] for r in member])
        # one disc per cell type, in that type's own atlas colour and with no
        # outline -- at this size an outline reads as a second colour
        ax.scatter(_swarm(v, x), v, s=14, zorder=4, edgecolor="none",
                   color=[style.CELL_COLOUR[r["channel"]] for r in member])
        style.mean_mark(ax, x, v.mean(), style.GROUP_COLOUR[g], half=0.24,
                        lw=2.4, zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{g}  {SHORT[g]}" for g in "ABCD"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("knock-out effect\n(× replicate distance)")
    ax.set_ylim(0, None)
    ax.set_xlim(-0.55, 3.55)
    ax.set_title("A   Effect of removing one cell type, by a-priori group", color=style.INK)


def _swarm(v, x, half=0.17):
    """Spread ties sideways so every cell type stays visible."""
    order = np.argsort(v)
    off = np.zeros(len(v))
    for rank, i in enumerate(order):
        near = np.abs(v - v[i]) < (v.max() - v.min() + 1e-9) * 0.05
        k = int(near[:i].sum())
        off[i] = ((k % 3) - 1) * half * 0.9 if near.sum() > 1 else 0.0
    return x + off


def panel_cascade(axes, cache, real_dots=True):
    d = S.load(cache)
    idx = [d["channels"].index(c) for c in d["cascade"]]
    spl = d["meta"]["steps_per_iter"] if "steps_per_iter" in d["meta"] \
        else d["meta"]["steps_per_leg"]
    curves = {b["channel"]: np.array(b["curve"]) for b in d["blocks"]}
    ctrl = np.array(d["ctrl"])
    # leg position: the end of leg j is stage j, so a recorded step sits at
    # j-1 + i/spl and the observed stages land on the integers
    t = np.arange(1, ctrl.shape[0] + 1) / spl
    real = np.array(d["real"])
    conditions = ([("unperturbed", ctrl)]
                  + [(f"− {style.label(c)}", curves[c]) for c in d["cascade"]]
                  + [("no injury", np.array(d["uninjured"]))]
                  + [(f"− {style.label(d['controls'][0])} (bystander)",
                      curves[d["controls"][0]])])
    ymax = max(float(np.asarray(v)[:, idx].max()) for _n, v in conditions)
    for n, (name, cur) in enumerate(conditions):
        ax = axes[n]
        for k, c in zip(idx, d["cascade"]):
            ax.plot(t, cur[:, k], color=style.CELL_COLOUR[c], lw=1.1,
                    label=style.label(c), solid_capstyle="round")
        if real_dots and name == "unperturbed":
            for k, c in zip(idx, d["cascade"]):
                ax.scatter(np.arange(C.N_STAGES), real[:, k], s=5,
                           color=style.CELL_COLOUR[c], edgecolor="none", zorder=5)
        style.stage_axis(ax, labelled=n >= 4)
        ax.set_ylim(min(0, ymax * -0.08), ymax * 1.08)
        ax.set_title(name, color=style.INK, fontsize=6.5)
        if n % 4:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("abundance")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=runs.REGIS[0], choices=runs.ALL)
    a = p.parse_args()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(style.TEXT_WIDTH, 5.0))
    gs_a = fig.add_gridspec(1, 1, top=0.965, bottom=0.685)
    gs_b = fig.add_gridspec(2, 4, top=0.565, bottom=0.135, hspace=0.50, wspace=0.24)
    panel_groups(fig.add_subplot(gs_a[0, 0]),
                 WORK / f"eval/hearts/knockout_{a.run}.json")
    axes = [fig.add_subplot(gs_b[r, c]) for r in range(2) for c in range(4)]
    panel_cascade(axes, WORK / f"eval/hearts/cascade_{a.run}.json")
    fig.text(0.005, 0.615, "B   Cardiomyocyte states through regeneration, "
             "unperturbed and with each state suppressed",
             color=style.INK, fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.1, columnspacing=1.2)
    style.save(fig, WORK / "figures/hearts", f"fig_perturbation_{a.run}")


if __name__ == "__main__":
    main()
