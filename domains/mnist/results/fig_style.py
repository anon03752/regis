"""Plot digit styles associated with persistent noise vectors.

Sample 128 noise vectors from N(0, I), clipped to [-1.8, 1.8], and start
each trajectory from the same real zero. After 30 transitions, record 2.6
cycles. For each digit, select the midpoint of its longest classifier-stable
sequence of at least five updates.

Exclude styles with missing digits or more than 1.6 connected ink components
per digit on average. Select eight diverse rows by farthest-point sampling
in standardised ink, slant, and fill features, starting farthest from the mean.

    python -m domains.mnist.results.fig_style

Writes <workspace>/figures/mnist/fig_style.{pdf,png}.
"""
import argparse
import itertools

import numpy as np
import torch
from scipy import ndimage

from workspace import WORK
from figstyle import plt, save, paint_cmap, group_rule, INK, TEXT, HAIR
from domains.mnist.classifier import load_classifier
from domains.mnist.data import digits_by_class
from domains.mnist.instruments.measure import load_ema_generator
from domains.mnist.results import runs

RUN, K, SHOW, SEED, MAX_FRAG = runs.REGIS[0], 128, 8, 7, 1.6     # run, styles sampled/shown, seed, mean component limit


def stable_frames(pred):
    """digit -> midpoint step of its longest stable run of at least 5 updates; None if a digit is missing"""
    best, i = {}, 0
    for d, run in itertools.groupby(pred.tolist()):
        L = len(list(run))
        if L >= 5 and L > best.get(d, (0, 0))[1]:
            best[d] = (i + L // 2, L)
        i += L
    return None if len(best) < 10 else {d: best[d][0] for d in range(10)}


def hand_features(img):
    """boldness, slant, fill: all on the ink mask, all scale-free"""
    m = img > .5
    if m.sum() < 8:
        return np.nan, np.nan, np.nan
    ys, xs = np.nonzero(m)
    x0, y0 = xs.mean(), ys.mean()
    cyy = ((ys - y0) ** 2).mean()
    cxy = ((xs - x0) * (ys - y0)).mean()
    return float(m.mean()), float(cxy / max(cyy, 1e-6)), float(m.sum() / max((np.ptp(xs) + 1) * (np.ptp(ys) + 1), 1))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    G, cfg = load_ema_generator(runs.checkpoint(RUN), dev)
    C, ZD, updates = cfg["channel_n"], cfg["z_dim"], cfg["max_nca_steps"]
    clf = load_classifier(dev)
    val = digits_by_class(False, dev)

    # Use one shared zero image to compare styles across latent vectors.
    x = torch.zeros(K, C, 32, 32, device=dev)
    x[:, -1] = val[0][torch.randint(0, len(val[0]), (1,)).item(), 0]
    z = torch.randn(K, ZD, device=dev).clamp(-1.8, 1.8)
    vis, rd = [], []
    with torch.no_grad():
        for _ in range(30 * updates):
            x = G.step(x, z=z)
        for _ in range(int(2.6 * 10 * updates)):
            x = G.step(x, z=z)
            rd.append(clf(x[:, -1:].float()).argmax(1).cpu().numpy())
            vis.append(x[:, -1].half().cpu())
    VIS, RD = torch.stack(vis).numpy(), np.stack(rd)

    ROWS = []
    for k in range(K):
        kn = stable_frames(RD[:, k])
        if kn is None:
            continue
        imgs = np.stack([VIS[kn[d], k].astype(np.float32) for d in range(10)])
        frag = float(np.mean([ndimage.label(im > .5)[1] for im in imgs]))
        ROWS.append((imgs, np.nanmean(np.array([hand_features(im) for im in imgs]), 0), frag))
    print(f"{len(ROWS)}/{K} styles produced all ten digits in stable runs", flush=True)
    kept = [r for r in ROWS if r[2] <= MAX_FRAG]
    print(f"{len(kept)}/{len(ROWS)} meet the fragmentation limit (<= {MAX_FRAG} components per digit)", flush=True)
    assert len(kept) >= SHOW, "too few styles meet the fragmentation limit"
    Xs = np.array([r[1] for r in kept])
    Xs = np.nan_to_num((Xs - np.nanmean(Xs, 0)) / (np.nanstd(Xs, 0) + 1e-9))
    idx = [int(np.argmax(np.linalg.norm(Xs, axis=1)))]
    while len(idx) < SHOW:
        d = np.min(np.linalg.norm(Xs[:, None] - Xs[idx][None], axis=2), axis=1)
        d[idx] = -1
        idx.append(int(np.argmax(d)))
    ROWS = [kept[i] for i in sorted(idx, key=lambda i: kept[i][1][0])]       # display thin -> bold

    NR, NC = len(ROWS), 10
    FW, GAP, LEFT, RIGHT, TOP, BOT = 5.5, .04, .26, .01, .33, .02
    SUB = (FW - LEFT - RIGHT - (NC - 1) * GAP) / NC
    FH = TOP + NR * SUB + (NR - 1) * GAP + BOT
    fig = plt.figure(figsize=(FW, FH))
    cmap = paint_cmap().reversed()      # white digits on the accent blue
    for r, (imgs, _st, _frag) in enumerate(ROWS):
        y = TOP + r * (SUB + GAP)
        for c in range(NC):
            xx = LEFT + c * (SUB + GAP)
            ax = fig.add_axes([xx / FW, (FH - y - SUB) / FH, SUB / FW, SUB / FH])
            ax.imshow(np.clip(imgs[c], 0, 1), cmap=cmap, vmin=0, vmax=1, interpolation="none", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(HAIR)
                sp.set_linewidth(0.6)
            if r == 0:
                fig.text((xx + SUB / 2) / FW, (FH - TOP + .04) / FH, str(c), ha="center", va="bottom", fontsize=7.4, color=INK)
    group_rule(fig, .075 / FW, .16 / FW, BOT / FH, (FH - TOP) / FH, "Style vector", size=7.0)
    fig.text((LEFT + (FW - LEFT - RIGHT) / 2) / FW, (FH - .03) / FH, "Digit", ha="center", va="top", fontsize=7.4, color=TEXT)
    save(fig, WORK / "figures" / "mnist", "fig_style")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
