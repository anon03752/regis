"""The nine marginals MMtSBM trains on, from the cohort.

    python -m domains.hearts.data_bridges     # -> data/hearts/bridges/

The eight measured stages and the return to uninjured: uninjured -> 6 hpa ->
... -> 28 dpa -> uninjured. The 18 uninjured sections are split into disjoint
halves, one per end of the loop, so closing it is a claim about the marginal
rather than an identity. The 21 generator channels are kept (DamageRef is not
modelled) and each is z-scored over all marginals; the centre and scale go to
anchors.json and every evaluation inverts them before measuring.

Writes hearts_t<i>.pt (N, 21, 48, 48), hearts_mask<i>.pt (N, 1, 48, 48) and
anchors.json, the layout baselines/mmtsbm/train.py reads with --prefix hearts_t.
"""
import argparse
import json

import numpy as np
import torch

from workspace import WORK
from domains.hearts import cohort as C

OUT = WORK / "data/hearts/bridges"
CHANNELS = C.CELL_TYPES + [C.DAMAGE_CHANNEL]
SPLIT_SEED = 0


def build():
    secs = C._read(C.PATH)
    stack = lambda ss: (torch.stack([s["composition"][:len(CHANNELS)] for s in ss]).float(),
                        torch.stack([s["mask"] for s in ss]).float())
    by_stage = [[s for s in secs if s["timepoint"] == tp] for tp in C.TP_ORDER]
    uninjured = by_stage[0]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(uninjured))
    half = len(uninjured) // 2
    start = [uninjured[i] for i in perm[:half]]
    back = [uninjured[i] for i in perm[half:2 * half]]
    marginals = [stack(ss) for ss in [start] + by_stage[1:] + [back]]
    names = C.TP_ORDER + ["uninjured (return)"]

    allx = torch.cat([x for x, _ in marginals])
    centre = allx.mean(dim=(0, 2, 3))
    scale = allx.std(dim=(0, 2, 3)).clamp(min=1e-3)
    z = lambda x: (x - centre[None, :, None, None]) / scale[None, :, None, None]
    return [(z(x), m) for x, m in marginals], names, centre, scale


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    marginals, names, centre, scale = build()
    OUT.mkdir(parents=True, exist_ok=True)
    for i, ((x, m), name) in enumerate(zip(marginals, names)):
        torch.save(x, OUT / f"hearts_t{i}.pt")
        torch.save(m, OUT / f"hearts_mask{i}.pt")
        print(f"  t{i} {name:<20} {tuple(x.shape)}")
    (OUT / "anchors.json").write_text(json.dumps(
        {"names": names, "channels": CHANNELS, "channel_centre": centre.tolist(),
         "channel_scale": scale.tolist()}, indent=1))
    print(f"wrote {len(marginals)} marginals to {OUT}")


if __name__ == "__main__":
    main()
