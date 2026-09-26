"""Plot domain growth for the marginal-placement experiments.

Each row shows the training marginals on a log-time axis and L/L_GT at
t = 1000 and t = 4000. The filled marker denotes t = 4000. Shading shows the
range across reference runs on the standard five-marginal grid.

Reads <work>/eval/marginals512.json.
Usage: python -m domains.ising.results.fig_marginals512
Writes <work>/figures/fig_ising_marginals.{png,pdf}.
"""
import json

from domains.ising.results.runs import HORIZON
from figstyle import plt, save, ACCENT, INK, TEXT, MUTED, FAINT, GRID, A_BAND, STEP_1
from workspace import WORK

J = json.load(open(WORK / "eval/marginals512.json"))
BAND = J["reference_band"]["4000"]
RUNS = {(tuple(r["times"]), r["seed"] >= 200): r for r in J["runs"]}   # (grid, second seed?) -> run

# Section and row labels, keyed by (grid, second seed), in display order.
SECTIONS = [
    ("Number of marginals", [((3, 1000), False, "two"), ((3, 55, 1000), False, "three"),
                             ((3, 17, 175, 1000), False, "four"), ((3, 10, 55, 300, 1000), False, "five, evenly spaced"),
                             ((3, 10, 17, 55, 175, 300, 1000), False, "seven"),
                             ((3, 10, 17, 30, 55, 100, 175, 300, 550, 1000), False, "ten")]),
    ("Placement of the inner marginals", [((3, 10, 17, 30, 1000), False, "early"), ((3, 30, 55, 100, 1000), False, "middle"),
                                          ((3, 100, 175, 300, 1000), False, "mid to late"),
                                          ((3, 175, 300, 550, 1000), False, "late")]),
    ("Time of the first marginal", [((10, 30, 100, 300, 1000), False, "$t=10$"), ((30, 55, 175, 550, 1000), False, "$t=30$"),
                                    ((30, 55, 175, 550, 1000), True, "$t=30$, second seed"),
                                    ((100, 175, 300, 550, 1000), False, "$t=100$")]),
    ("One long gap", [((3, 5, 7, 10, 1000), False, "gap at the end"), ((3, 5, 7, 10, 1000), True, "gap at the end, second seed"),
                      ((3, 550, 700, 850, 1000), False, "gap at the start"),
                      ((3, 550, 700, 850, 1000), True, "gap at the start, second seed")]),
]
KMIN = 3
XLO, XHI = 0.55, 1.22

rows = []                                        # (kind, payload) top to bottom
for title, items in SECTIONS:
    rows.append(("head", title))
    for grid, second, lab in items:
        rows.append(("run", (RUNS[grid, second], lab)))
n = len(rows)

fig = plt.figure(figsize=(5.5, 0.165 * n + 0.90))
gs = fig.add_gridspec(1, 2, left=0.305, right=0.985, top=1 - 0.42 / (0.165 * n + 0.90),
                      bottom=0.80 / (0.165 * n + 0.90), wspace=0.06, width_ratios=[1.0, 1.25])
axL, axR = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
for ax in (axL, axR):
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
axL.set_xscale("log")
axL.set_xlim(KMIN * 0.75, HORIZON * 1.35)
axL.set_xticks([10, 100, 1000])
axL.set_xticklabels(["10", "100", "1k"])
axL.set_xlabel("marginal times $t$", labelpad=1.5)
axR.set_xlim(XLO, XHI)
axR.set_xticks([0.6, 0.8, 1.0, 1.2])
axR.set_xlabel("$L/L_{\\mathrm{GT}}$ at $t=4000$", labelpad=1.5)
axR.axvspan(BAND[0], BAND[1], color=ACCENT, alpha=A_BAND, lw=0, zorder=1)
axR.axvline(1.0, color=INK, lw=0.8, ls=(0, (4, 2.5)), zorder=2)

for i, (kind, payload) in enumerate(rows):
    if kind == "head":
        pos = axL.get_position()
        y = pos.y0 + (pos.y1 - pos.y0) * (1 - (i + 0.5) / n)
        fig.text(0.012, y, payload, ha="left", va="center", fontsize=7.0, fontweight="semibold", color=INK)
        continue
    r, lab = payload
    axL.text(KMIN * 0.66, i, lab, ha="right", va="center", fontsize=6.4, color=TEXT, clip_on=False)
    axL.plot([KMIN, HORIZON], [i, i], color=GRID, lw=0.8, zorder=1, solid_capstyle="butt")
    axL.plot(r["times"], [i] * len(r["times"]), "o", color=INK, ms=2.5, mew=0, zorder=3, clip_on=False)
    a, b = r["ratio"]["1000"], r["ratio"]["4000"]
    axR.plot([XLO, XHI], [i, i], color=GRID, lw=0.8, zorder=0, solid_capstyle="butt")
    if abs(a - b) > 0.004:                 # dumbbell only when the two times differ visibly
        axR.plot([a, b], [i, i], color=STEP_1, lw=1.5, alpha=0.55, zorder=3, solid_capstyle="round")
        axR.plot([a], [i], "o", mfc="white", mec=STEP_1, ms=3.4, mew=0.9, zorder=4)
    axR.plot([b], [i], "o", color=STEP_1, ms=4.2, mew=0, zorder=5)
    if not (XLO < b < XHI):
        axR.annotate(f"{b:.2f}", (XLO + 0.012, i), fontsize=5.8, color=STEP_1, ha="left", va="center")

for dx, kw in ((0.0, dict(mfc="white", mec=STEP_1, ms=3.4, mew=0.9)), (0.045, dict(color=STEP_1, ms=4.2, mew=0))):
    fig.add_artist(plt.Line2D([0.016 + dx], [0.028], marker="o", ls="none", **kw))
fig.add_artist(plt.Line2D([0.021, 0.056], [0.028, 0.028], color=STEP_1, lw=1.5, alpha=0.55))
fig.text(0.072, 0.028, "$t=1000 \\rightarrow 4000$", fontsize=5.8, color=MUTED, ha="left", va="center")
fig.text(0.985, 0.028, "shaded: the reference runs on the standard five-marginal grid",
         fontsize=5.8, color=FAINT, ha="right", va="bottom")
save(fig, WORK / "figures", "fig_ising_marginals")
