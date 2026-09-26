"""Plot Ising rollouts from a shared initial field, with methods in rows.

MMSFM, MMtSBM, and DDPM use continuous states from the raw strip caches;
the other methods use field 0 of the binary evaluation caches. MMSFM and
MMtSBM have no entries beyond their final observation time.

The short figure shows ground truth, REGIS, MMSFM, MMtSBM, and DDPM. The
full figure adds the perception-radius variants and controls."""
import sys

import numpy as np

from domains.ising.instruments import rollouts
from domains.ising.results import runs
from domains.ising.results.runs import HORIZON
from figstyle import plt, save, ACCENT, INK, TEXT, MUTED, FAINT, HAIR, STEP_1, TRANS_1, paint_cmap
from workspace import WORK

RO = WORK / "data/rollouts"
# the raw (continuous) strip caches of reproduce.md, Ising §3, by row stem.
# The DDPM needs two chains from t = 3: 3->17 onwards for the later columns,
# and 3->13, whose frame fills the t = 10 column (3->10 is below the trained
# jump floor of 10).
RAW = {runs.MMSFM[0]: ["mmsfm_seed0_strip__lat512_ns2.npz"],
       runs.MMTSBM[0]: ["mmtsbm_seed13_strip__lat512_ns4.npz"],
       runs.DDPM[0]: ["ddpm_pairs15000_seed0_strip__lat512_ns2.npz", "ddpm_pairs15000_seed0_strip13__lat512_ns2.npz"]}
SHOWN = {10: 13}   # the DDPM's earliest frame (t = 3 + its minimum jump of 10) fills the t = 10 column
ROWS_ALL = [                       # (group, label, cache stem, transport?)
    ("Reference", "Ground truth", runs.TRUTH[0], False),
    ("Marginals only", "REGIS", runs.REGIS[0], False),
    ("Marginals only", "MMSFM", runs.MMSFM[0], True),
    ("Marginals only", "MMtSBM", runs.MMTSBM[0], True),
    ("Marginals only", "REGIS, 7×7", runs.REGIS_K7[0], False),
    ("Marginals only", "REGIS, 5×5", runs.REGIS_K5[0], False),
    ("Marginals only", "Non-local control", runs.CONTROL[0], False),
    ("Pairs", "DDPM", runs.DDPM[0], False),
    ("Pairs", "Pair-conditional GAN", runs.PAIRGAN[0], False),
]
SHORT = {"Ground truth", "REGIS", "MMSFM", "MMtSBM", "DDPM"}
COLS = [10, 17, 150, 300, 700, 1000, 2000, 4000]
KIND = {10: "observed", 17: "observed", 150: "unobserved", 300: "observed", 700: "unobserved",
        1000: "observed", 2000: "extrapolation", 4000: "extrapolation"}
KIND_COL = {"observed": ACCENT, "unobserved": STEP_1, "extrapolation": TRANS_1}
CROP = 256                          # top-left window of the 512^2 field, so the structure is legible in print


def load(stem):
    """{read-out time: field-0 crop}: the raw strips for the three sampler rows, else the binary cache"""
    if stem in RAW:
        frames = {}                                       # k -> field-0 crop, over all raw files
        for f in RAW[stem]:
            z = np.load(RO / f)
            for k, r in zip(z["ks"], z["raw"]):
                frames.setdefault(int(k), np.asarray(r[0], dtype=np.float32)[:CROP, :CROP])
        return frames
    bits, ks, m = rollouts.load(runs.cache_path(stem))
    return {k: rollouts.unpack(row, m["lat"], 1)[0, :CROP, :CROP] for row, k in zip(bits, ks)}


def strip(name, rows):
    data = {stem: load(stem) for _, _, stem, _ in rows}
    nr, nc = len(rows), len(COLS)
    cell, LEFT, RIGHT, TOP, BOT = 0.545, 1.16, 0.06, 0.52, 0.06
    Wi, Hi = LEFT + cell * nc + RIGHT, TOP + cell * nr + BOT
    fig = plt.figure(figsize=(Wi, Hi))
    gs = fig.add_gridspec(nr, nc, left=LEFT / Wi, right=1 - RIGHT / Wi, top=1 - TOP / Hi, bottom=BOT / Hi,
                          wspace=0.05, hspace=0.05)
    cmap = paint_cmap()
    axes, blanks = {}, {}
    for r, (grp, lab, stem, is_tr) in enumerate(rows):
        for c, k in enumerate(COLS):
            ax = fig.add_subplot(gs[r, c])
            axes[(r, c)] = ax
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():   # Restore all four borders around each image.
                s.set_visible(True)
                s.set_color(HAIR)
                s.set_linewidth(0.6)
            if is_tr and k > HORIZON:
                for s in ax.spines.values():
                    s.set_visible(False)
                blanks.setdefault(r, []).append(ax)
                continue
            kk = k if k in data[stem] else SHOWN[k]
            ax.imshow(np.clip(data[stem][kk], 0, 1), cmap=cmap, vmin=0, vmax=1, interpolation="none")
            if kk != k:                      # Label the actual DDPM time when it differs from the column time.
                ax.text(0.03, 0.97, f"$t = {kk}$", transform=ax.transAxes, ha="left", va="top",
                        fontsize=5.8, color=MUTED,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
            if r == 0:
                ax.set_title(f"$t = {k}$", fontsize=7.6, pad=9, color=INK)
                ax.text(0.5, 1.012, KIND[k], transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6.4, color=KIND_COL[KIND[k]])
        p0 = axes[(r, 0)].get_position()
        fig.text(p0.x0 - 0.010, 0.5 * (p0.y0 + p0.y1), lab, ha="right", va="center", fontsize=7.4, color=INK)
    for r, axs in blanks.items():          # one centred note across the blank span
        b0, b1 = axs[0].get_position(), axs[-1].get_position()
        fig.text(0.5 * (b0.x0 + b1.x1), 0.5 * (b0.y0 + b0.y1), "no state beyond $t = 1000$",
                 ha="center", va="center", fontsize=6.4, color=FAINT)
    # group brackets down the left edge
    for grp in dict.fromkeys(g for g, *_ in rows):
        idx = [r for r, (g, *_) in enumerate(rows) if g == grp]
        top = axes[(idx[0], 0)].get_position().y1
        bot = axes[(idx[-1], 0)].get_position().y0
        x = (0.012 * 5.5) / Wi
        fig.add_artist(plt.Line2D([x, x], [bot, top], color=TEXT, lw=0.9, solid_capstyle="butt"))
        fig.text(x - 0.004, 0.5 * (bot + top), grp, rotation=90, ha="right", va="center",
                 fontsize=7.0, fontweight="semibold", color=INK)
    save(fig, WORK / "figures", name)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("short", "both"):
        strip("fig3_ising_filmstrip_strip", [r for r in ROWS_ALL if r[1] in SHORT])
    if which in ("full", "both"):
        strip("fig_ising_strip_full", ROWS_ALL)
