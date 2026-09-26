"""Plot generated digits at five rollout horizons, with two seeds per method.

Each cell contains examples classified as 0, 1, 2, and 3. For each digit,
select the median-confidence example among the capture's stored grid frames.
If no frame is classified as that digit, use the highest-scoring candidate
and print the digit's population frequency. Examples are selected separately
at each horizon; columns do not track individual trajectories.

Frames are clipped to [0, 1] for display only. SHOWN selects runs by the
indices in runs.py, also used in the per-seed table.

    python -m domains.mnist.results.fig_samples

Writes <workspace>/figures/mnist/fig_perseed.{pdf,png}.
"""
import argparse
import numpy as np
import torch

from workspace import WORK
from figstyle import plt, save, paint_cmap, group_rule, INK, TEXT, HAIR
from domains.mnist.classifier import load_classifier
from domains.mnist.results import runs

COLS = [1, 5, 10, 100, 1000]
DIGITS = [0, 1, 2, 3]
SHOWN = {"REGIS": (3, 4), "Non-local control": (0, 2), "MMSFM": (1, 0), "MMtSBM": (4, 1)}


def main():
    clf = load_classifier("cpu")

    @torch.no_grad()
    def softmax_reads(x):
        # reads are taken on the CLIPPED view, as the MMtSBM sampler selects its grid
        # rows (MMSFM's reads the unclamped frame); on a diverged seed the two differ
        t = torch.from_numpy(np.ascontiguousarray(np.clip(x, 0, 1), np.float32))[:, None]
        return torch.softmax(clf(t), 1).numpy()

    def pick(grid_h):
        pool = grid_h.reshape(-1, 32, 32).astype(np.float32)
        pool = pool[np.any(pool != 0, axis=(1, 2))]              # drop zero-padded slots
        pr = softmax_reads(pool)
        rd = pr.argmax(1)
        out = []
        for d in DIGITS:
            idx = np.where(rd == d)[0]
            if len(idx):
                out.append((pool[idx[np.argsort(pr[idx, d])[len(idx) // 2]]], False))
            else:
                out.append((pool[int(np.argmax(pr[:, d]))], True))
        return out

    rows = [(name, stems[s]) for name, stems in runs.METHODS for s in SHOWN[name]]
    NR, NC = len(rows), len(COLS)
    # placed in inches: authored at 4.4 in, the width it is included at (0.8 of the text width)
    FW, INNER, GAPX, GAPY, GROUPGAP = 4.4, .022, .075, .075, .11
    LEFT, RIGHT, TOP, BOT = .62, .01, .36, .02
    CELL = (FW - LEFT - RIGHT - (NC - 1) * GAPX) / NC
    SUB = (CELL - INNER) / 2
    ytops, y = [], TOP
    for r in range(NR):
        ytops.append(y)
        y += CELL + (GROUPGAP if r + 1 < NR and rows[r + 1][0] != rows[r][0] else GAPY)
    FH = ytops[-1] + CELL + BOT
    fig = plt.figure(figsize=(FW, FH))
    cmap = paint_cmap().reversed()      # white digits on the accent blue
    rect = lambda x, yy, w, h: [x / FW, (FH - yy - h) / FH, w / FW, h / FH]

    for r, (name, stem) in enumerate(rows):
        path = runs.capture(stem)
        with np.load(path) as d:
            keep, grid, pred = [int(x) for x in d["keep"]], d["grid"], d["pred"]
        ytop = ytops[r]
        fig.text((LEFT - .05) / FW, (FH - ytop - CELL / 2) / FH, f"seed {runs.shown_seed(stem)}",
                 ha="right", va="center", fontsize=6.6, color=INK)
        for c, t in enumerate(COLS):
            k = keep.index(t)
            xleft = LEFT + c * (CELL + GAPX)
            if r == 0:
                fig.text((xleft + CELL / 2) / FW, (FH - TOP + .045) / FH, f"{t:,}", ha="center", va="bottom",
                         fontsize=7.4, color=INK)
            for i, (im, fallback) in enumerate(pick(grid[k])):
                a = fig.add_axes(rect(xleft + (i % 2) * (SUB + INNER), ytop + (i // 2) * (SUB + INNER), SUB, SUB))
                a.set_xticks([])
                a.set_yticks([])
                a.grid(False)
                a.imshow(np.clip(im, 0, 1), cmap=cmap, vmin=0, vmax=1, interpolation="none", aspect="auto")
                for sp in a.spines.values():
                    sp.set_visible(True)
                    sp.set_color(HAIR)
                    sp.set_linewidth(0.6)
                if fallback:
                    print(f"  {name} seed {runs.shown_seed(stem)}, transition {t}: no stored frame read as {DIGITS[i]} "
                          f"(population frequency {100 * float((pred[k] == DIGITS[i]).mean()):.2f}%)")
    for name in dict.fromkeys(n for n, _ in rows):
        idx = [r for r, (n, _) in enumerate(rows) if n == name]
        group_rule(fig, .075 / FW, .16 / FW, (FH - ytops[idx[-1]] - CELL) / FH, (FH - ytops[idx[0]]) / FH, name, size=7.0)
    fig.text((LEFT + (FW - LEFT - RIGHT) / 2) / FW, (FH - .035) / FH, "Digit transitions completed", ha="center",
             va="top", fontsize=7.4, color=TEXT)
    save(fig, WORK / "figures" / "mnist", "fig_perseed")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
