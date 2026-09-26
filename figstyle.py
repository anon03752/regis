"""Shared plotting style for the Ising and MNIST figures.

Uses Source Sans 3, with optional font files under <work>/fonts. Matplotlib
falls back to its default font if Source Sans 3 is unavailable."""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from workspace import WORK

for _f in glob.glob(str(WORK / "fonts/**/*.otf"), recursive=True):
    if "Variable" in _f or "Display" in _f:
        continue
    fm.fontManager.addfont(_f)

SANS = "Source Sans 3"
ACCENT = "#2D54A0"                  # RGB 45 84 160; also the ground truth
INK = "#0b0b0b"                     # headings, panel titles, anything read first
TEXT = "#2a2a28"                    # axes, tick marks, marker edges
MUTED = "#52514e"                   # secondary annotations
FAINT = "#8a8983"                   # captions, footnotes
RULE_LINE = "#6a6a68"               # reference lines (dashed baselines)
GRID = "#ececec"                    # horizontal grid
HAIR = "#cccccc"                    # panel outlines in small-multiple grids
A_BAND = 0.16                       # opacity of shaded bands (truth spread, reference range)
# one hue per family, two lightness levels within a family
STEP_1, STEP_2 = "#D95F02", "#F4A259"       # update rules, marginals only: REGIS / non-local control
TRANS_1, TRANS_2 = "#7B3FA0", "#B58BD1"     # transport, marginals only: MMSFM / MMtSBM
PAIR_1, PAIR_2 = "#1B9E77", "#7CCBA2"       # paired rows: DDPM / pair-conditional GAN
BEYOND = "#f4f4f3"                          # ground beyond the training horizon

plt.rcParams.update({
    # type
    "font.family": SANS, "font.size": 8.0, "axes.titlesize": 8.5, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "axes.titlepad": 4.0, "axes.labelpad": 2.5, "axes.titlelocation": "left", "axes.titleweight": "semibold",
    # frame: no box
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.edgecolor": TEXT, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": TEXT, "ytick.color": TEXT, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
    # grid: horizontal only, faint, behind the data
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.alpha": 1.0,
    # Legend
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5, "legend.labelspacing": 0.35,
    # lines and markers
    "lines.linewidth": 1.4, "lines.markersize": 4.0, "lines.solid_capstyle": "round", "patch.linewidth": 0.0,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 110,
    "mathtext.fontset": "custom", "mathtext.rm": SANS, "mathtext.it": SANS + ":italic", "mathtext.bf": SANS + ":semibold",
    "mathtext.cal": SANS, "mathtext.sf": SANS, "mathtext.tt": "DejaVu Sans Mono",
})


def group_rule(fig, x_text, x_rule, y_bot, y_top, label, size=8.0):
    """rotated group label beside a vertical rule spanning [y_bot, y_top] (figure coords)"""
    fig.text(x_text, 0.5 * (y_top + y_bot), label, fontsize=size, color=INK, fontweight="semibold", rotation=90, va="center", ha="center")
    fig.add_artist(plt.Line2D([x_rule, x_rule], [y_bot, y_top], transform=fig.transFigure, color=TEXT, lw=0.8, solid_capstyle="butt"))


def paint_cmap():
    """continuous white -> accent blue for raw (un-binarised) fields"""
    return matplotlib.colors.LinearSegmentedColormap.from_list("ising", ["#FFFFFF", ACCENT])


def save(fig, out_dir: Path, name: str, dpi=300):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_dir / name}.png/.pdf")
