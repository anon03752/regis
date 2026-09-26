"""The observed cohort: every section, and how much it varies.

Two figures, both from the data alone -- no model is loaded, so they are the
description the chapter's caveats rest on.

`sections`  all 105 cross-sections, one column per sampling stage, each bin
            coloured by its dominant cell type with opacity set by density.

`diversity` why a strict marginal-matching score would be misleading here.
            Left: heart size per section, which varies with a coefficient of
            variation of roughly a quarter -- the uninjured cohort is far larger
            than the 12 hpa one, which is technical bias, not biology. Right:
            each cell type's abundance spread over the WHOLE cohort. The types
            above 200% are the injury-induced ones, which are absent at most
            stages by construction; the point of the panel is that a fit score
            pooled over this cohort would mostly measure which sections were
            cut. (This is a pooled statistic, not the within-stage compartment
            CV quoted in the text.)

    python -m domains.hearts.results.fig_cohort

Writes <workspace>/figures/hearts/fig_cohort_{sections,diversity}.pdf.
"""
import argparse

import numpy as np

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts.results import style
from domains.hearts.rule import N_VISIBLE

CAPACITY = 4.0


def _crop(mask, pad=1):
    """Bounding box of the tissue. A section occupies roughly 30x40 of the 48x48
    canvas, so drawing the full canvas spends most of the panel on margin."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return slice(None), slice(None)
    return (slice(max(ys.min() - pad, 0), ys.max() + pad + 1),
            slice(max(xs.min() - pad, 0), xs.max() + pad + 1))


def _rgb(comp, mask, palette, gamma=0.65):
    comp = np.clip(comp, 0.0, None)
    dens = np.clip(comp.sum(axis=0) / CAPACITY, 0.0, 1.0) ** gamma
    col = palette[comp.argmax(axis=0)]
    return np.where(mask[..., None] > 0, 1.0 - dens[..., None] * (1.0 - col), 1.0)


def sections(banks, palette, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    # stages down, sections across: the cohort is 8 stages of 12-18 sections, so
    # stacking the long axis horizontally gives a page-shaped figure instead of
    # a column of specks
    ncol = max(banks[tp][0].shape[0] if tp in banks else 0 for tp in C.TP_ORDER)
    fig, axes = plt.subplots(C.N_STAGES, ncol,
                             figsize=(style.TEXT_WIDTH, style.TEXT_WIDTH * C.N_STAGES / ncol),
                             squeeze=False)
    for row, tp in enumerate(C.TP_ORDER):
        comp, mask = banks.get(tp, (None, None))
        for col in range(ncol):
            ax = axes[row][col]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if comp is None or col >= comp.shape[0]:
                ax.axis("off")
            else:
                mk = (mask[col, 0] > 0).cpu().numpy()
                ax.imshow(_rgb(comp[col, :N_VISIBLE].cpu().numpy(), mk, palette)[_crop(mk)],
                          interpolation="nearest")
            if col == 0:
                ax.set_ylabel(tp, fontsize=5.6, color=style.TEXT, rotation=0,
                              ha="right", va="center", labelpad=3)
    fig.legend(handles=[Patch(facecolor=style.CELL_COLOUR[n], edgecolor=style.HAIR,
                              linewidth=0.3, label=style.label(n))
                        for n in C.CELL_TYPES],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.16),
               handlelength=1.0, handleheight=0.7, columnspacing=1.0)
    fig.text(0.005, 1.0, "Observed heart sections across sampling stages",
             color=style.INK, fontsize=7.5, va="bottom")
    fig.subplots_adjust(hspace=0.06, wspace=0.04)
    style.save(fig, out_dir, "fig_cohort_sections")
    plt.close(fig)


def diversity(banks, out_dir):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    size, stage, per_type = [], [], []
    for i, tp in enumerate(C.TP_ORDER):
        if tp not in banks:
            continue
        comp, mask = banks[tp]
        m = (mask[:, 0] > 0)
        for k in range(comp.shape[0]):
            size.append(float(m[k].sum()))
            stage.append(i)
            per_type.append(comp[k, :len(C.CELL_TYPES)][:, m[k]].mean(dim=1).cpu().numpy())
    size, stage = np.array(size), np.array(stage)
    per_type = np.stack(per_type)                         # (n_sections, n_types)

    fig, axes = plt.subplots(1, 2, figsize=(style.TEXT_WIDTH, 2.0),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    for i in range(C.N_STAGES):
        sel = stage == i
        if not sel.any():
            continue
        ax.scatter(i + rng.uniform(-0.16, 0.16, sel.sum()), size[sel], s=9,
                   facecolor="white", edgecolor=style.INK, linewidth=0.55, zorder=3)
        style.mean_mark(ax, i, size[sel].mean(), style.TEXT, half=0.3, lw=1.3,
                        zorder=2)
        sd_ = size[sel].std(ddof=1) if sel.sum() > 1 else 0.0
        ax.vlines(i, size[sel].mean() - sd_, size[sel].mean() + sd_,
                  color=style.MUTED, lw=0.7, zorder=1)
    ax.set_xticks(range(C.N_STAGES))
    ax.set_xticklabels([t.replace("uninjured", "uninj.") for t in C.TP_ORDER],
                       rotation=45, ha="right")
    ax.set_ylim(0, None)
    ax.set_ylabel("occupied bins")
    ax.set_title(f"heart size per section  (CV {size.std(ddof=1) / size.mean() * 100:.0f}%)",
                 color=style.INK, fontsize=6.5)

    # Per-bin DENSITY is pinned near the bin capacity by construction, so its
    # spread says nothing; the variation that matters is compositional. This is
    # the quantity the chapter's caveat rests on.
    ax = axes[1]
    cv = per_type.std(axis=0, ddof=1) / np.maximum(per_type.mean(axis=0), 1e-9) * 100
    order = np.argsort(cv)
    # a dot at the value, not a bar: a bar encodes magnitude from zero and
    # invites comparing areas, and the near-white atlas hues read as empty when
    # filled. The hairline to the axis carries the reading without the ink.
    y = np.arange(len(order))
    ax.hlines(y, 0, cv[order], color=style.HAIR, lw=0.6, zorder=1)
    ax.scatter(cv[order], y, s=16, edgecolor="none", zorder=3,
               color=[style.CELL_COLOUR[C.CELL_TYPES[i]] for i in order])
    ax.set_yticks(y)
    ax.set_yticklabels([style.label(C.CELL_TYPES[i]) for i in order], fontsize=5.0)
    ax.set_xlabel("coefficient of variation across sections (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_title("abundance spread, by cell type", color=style.INK, fontsize=6.5)

    fig.text(0.005, 1.03, "Variation in heart size and composition across the cohort",
             color=style.INK, fontsize=7.5, va="bottom")
    fig.subplots_adjust(wspace=0.55)
    style.save(fig, out_dir, "fig_cohort_diversity")
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    palette = np.array([[int(style.CELL_COLOUR[n].lstrip("#")[i:i + 2], 16) / 255
                         for i in (0, 2, 4)] for n in C.CELL_TYPES + [C.DAMAGE_CHANNEL]])
    _seeds, _ids, banks, _refs = C.load_cohort(C.PATH, "cpu")
    out = WORK / "figures/hearts"
    out.mkdir(parents=True, exist_ok=True)
    sections(banks, palette, out)
    diversity(banks, out)


if __name__ == "__main__":
    main()
