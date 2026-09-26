"""What the heart figures add to the house style in figstyle.py: the atlas
palette, the stage axis, and a group mean mark.

The cell-type colours are the source publication's, keyed on the full prefixed
annotation string -- the label a figure prints and what the dataset stores, so
re-ordering channels cannot shift the mapping. They are fixed by correspondence
with the atlas and are not ours to re-step.

Twenty categorical hues is more than one axes can carry, so anything
per-cell-type is small multiples with one type per panel; group panels carry
four series and share one.
"""
from figstyle import INK, TEXT, MUTED, FAINT, GRID, HAIR, save  # noqa: F401

TEXT_WIDTH = 5.5           # the paper's text width, so a PDF drops in at scale 1

# Damage is ours, not the atlas's: the paper outlines the wound with a dashed
# line. A mid grey cannot be read as a cell type -- every atlas colour is either
# saturated or near-white.
CELL_COLOUR = {
    "1:BZm (activ.)": "#efcea0", "2:BZm (dediff.)": "#eca86a",
    "3:BZm (dediff.-prolif.)": "#e4561e", "4:BZm (prolif.)": "#91130c",
    "5:BZm (re-diff.)": "#eddf37", "6:RZm1": "#2f59a0", "7:RZm2": "#4d8cc0",
    "8:Vm (comp.)": "#afcb34", "9:Am": "#e4585b", "10:VAm": "#78b7d6",
    "11:MC": "#ce9be8", "12:FB (reg.)": "#aa1dff", "13:MFE": "#430e8d",
    "14:Epi": "#2c20ff", "15:SMC": "#e3c137", "16:Valves": "#f3ff9c",
    "17:RBC": "#f09498", "18:RBC (V)": "#c96281", "19:RBC (A)": "#e72721",
    "20:Others": "#f5f2f3", "21:Damage": "#9a9a9e",
}
# a group's colour is the atlas colour of a representative member
GROUP_COLOUR = {"A": "#e4561e", "B": "#430e8d", "C": "#f09498", "D": "#78b7d6"}

# The eight stages are 0, 6, 12, 24, 72, 168, 336 and 672 hours, so every
# trajectory's x axis is the STAGE INDEX, evenly spaced. On a time axis the four
# early stages collapse into the left margin, and the between-stage curve reads
# as data, which it is not.
STAGE_TICKS = [(0, "0"), (3, "1d"), (6, "14d")]


def label(name: str) -> str:
    return name.split(":", 1)[-1]


def stage_axis(ax, labelled: bool = True) -> None:
    ax.set_xlim(-0.3, 7.3)
    ax.set_xticks([p for p, _ in STAGE_TICKS])
    ax.set_xticklabels([l for _, l in STAGE_TICKS] if labelled else [])


def mean_mark(ax, x, value, colour, half=0.3, lw=2.8, **kw) -> None:
    """A group summary is a segment at the mean, never a bar: a bar encodes a
    magnitude from zero and none of these quantities has one."""
    ax.plot([x - half, x + half], [value, value], color=colour, lw=lw,
            solid_capstyle="round", **kw)
