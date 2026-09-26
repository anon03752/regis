"""A simulated wounded heart regenerating under each single-type knock-out.

Columns are the uninjured seed, the synthetic wound, and the seven measured
stages. Each bin is coloured by its dominant cell type and its opacity set by
density, under one display rule pooled over every panel, so a faint bin means
low mass and not "outside the heart". The dashed outline marks the wound
footprint. Rows are the unperturbed rollout followed by each knock-out, ordered
by measured effect.

This is one rollout per row, not a distribution -- it is the qualitative
companion to the perturbation figure, which is where the numbers live. A single
stochastic rollout is a draw; read the shapes, not the details.

    python -m domains.hearts.results.fig_unroll --run hearts_regis_seed0

Writes <workspace>/figures/hearts/fig_unroll_<stem>.pdf.
"""
import argparse

import numpy as np
import torch

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts import rollout as R
from domains.hearts.results import runs, style
from domains.hearts.results import stats as S
from domains.hearts.rule import N_VISIBLE, DAMAGE_IDX

CAPACITY = 4.0
SEED = 3                   # which uninjured section is cut, and where


def _crop(mask, pad=1):
    """Bounding box of the tissue. A section occupies roughly 30x40 of the 48x48
    canvas, so drawing the full canvas spends most of the panel on margin."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return slice(None), slice(None)
    return (slice(max(ys.min() - pad, 0), ys.max() + pad + 1),
            slice(max(xs.min() - pad, 0), xs.max() + pad + 1))


def _rgb(comp, mask, palette, gamma=0.65):
    """(H, W, 3): dominant type by hue, density by opacity onto white."""
    comp = np.clip(comp, 0.0, None)
    dom = comp.argmax(axis=0)
    dens = np.clip(comp.sum(axis=0) / CAPACITY, 0.0, 1.0) ** gamma
    col = palette[dom]                                    # (H, W, 3)
    out = 1.0 - dens[..., None] * (1.0 - col)
    return np.where(mask[..., None] > 0, out, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=runs.REGIS[0], choices=runs.ALL)
    a = p.parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rule, cfg = R.load_rule(runs.checkpoint(WORK, a.run), dev)
    holdout = cfg.get("holdout_heart") or ()
    names = C.CELL_TYPES + [C.DAMAGE_CHANNEL]
    palette = np.array([[int(style.CELL_COLOUR[n].lstrip("#")[i:i + 2], 16) / 255
                         for i in (0, 2, 4)] for n in names])

    # one heart, seeded twice: unwounded for the first column, then wounded
    rng = np.random.default_rng(SEED)
    clean, mask = R.seed_batch(C.PATH, 1, rng, dev, wound=False, holdout_hearts=holdout)
    wounded, _m = R.seed_batch(C.PATH, 1, np.random.default_rng(SEED), dev,
                               holdout_hearts=holdout)
    footprint = (wounded[0, DAMAGE_IDX] > 0.05).cpu().numpy()
    m = (mask[0, 0] > 0).cpu().numpy()
    box = _crop(m)

    # rows ordered by the screen's own effect, so the figure and the numbers agree
    cache = WORK / f"eval/hearts/knockout_{a.run}.json"
    order = ([(None, "unperturbed", None)] +
             [(C.CELL_TYPES.index(r["channel"]), style.label(r["channel"]), r["rho"])
              for r in S.load(cache)["rows"] if r["channel"] in C.CASCADE])

    rows = []
    for ko, label, rho in order:
        _s, legs = R.roll(rule, wounded.clone(), knockout=ko, ko_mode="zero", seed=7)
        panels = ([clean[0, :N_VISIBLE].cpu().numpy(), wounded[0, :N_VISIBLE].cpu().numpy()]
                  + [s[0, :N_VISIBLE].cpu().numpy() for s in legs])
        rows.append((label, rho, panels))
        print(f"  rolled {label}", flush=True)

    heads = ["uninjured", "wound"] + C.TP_ORDER[1:]
    nr, nc = len(rows), len(heads)
    fig, axes = plt.subplots(nr, nc, figsize=(style.TEXT_WIDTH, style.TEXT_WIDTH * nr / nc),
                             squeeze=False)
    for r, (label, rho, panels) in enumerate(rows):
        for c in range(nc):
            ax = axes[r][c]
            ax.imshow(_rgb(panels[c], m, palette)[box], interpolation="nearest")
            ax.contour(footprint[box].astype(float), levels=[0.5], colors=[style.TEXT],
                       linewidths=0.45, linestyles="dashed")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(heads[c], fontsize=5.6, color=style.TEXT, pad=2)
        txt = label if rho is None else f"− {label}"
        axes[r][0].set_ylabel(txt, fontsize=5.8, color=style.INK, rotation=0,
                              ha="right", va="center", labelpad=4)
        if rho is not None:
            axes[r][-1].annotate(f"{rho:.1f}×", (1.06, 0.5), xycoords="axes fraction",
                                 fontsize=5.6, color=style.MUTED, va="center")
    fig.legend(handles=[Patch(facecolor=style.CELL_COLOUR[n], label=style.label(n))
                        for n in C.CASCADE + [C.DAMAGE_CHANNEL]],
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.0, handleheight=0.7, columnspacing=1.0)
    fig.subplots_adjust(hspace=0.06, wspace=0.04)
    style.save(fig, WORK / "figures/hearts", f"fig_unroll_{a.run}")


if __name__ == "__main__":
    main()
