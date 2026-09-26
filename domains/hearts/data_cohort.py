"""Build the heart cohort from the published atlas and our wound annotations.

The input is one plain-text table, 21.4 MB, downloaded from the atlas's Zenodo
deposit:

    https://zenodo.org/records/14991776  ->  Stereo-seq-regeneration.meta.txt

Zhao et al., the processed data for Li et al. 2025, Nat. Commun. 16:3716
(doi 10.1038/s41467-025-59070-0), deposited CC BY 4.0. The same atlas is at
STOmicsDB STT0000071 and CNGB CNP0005245, and its browser at
db.cngb.org/stomics/zebrafish_VRH -- but that site publishes only binary .rds
and .h5ad objects, so the Zenodo copy is the one to take: it carries the spot
table directly and needs no R.

Attribution is a licence condition, so cite the atlas alongside this work in
anything built from it.

No expression data is used anywhere in this chapter. The model's channels are
one-hot cell-type labels from the `annotation` column, densified -- everything
needed is in eleven columns of that table, and the multi-gigabyte objects on
the same deposit are not touched.

What this module does with it:

1. One-hot per spot. Every spot carries exactly one of the 20 annotations, and
   the bin's grid index comes out of its barcode -- not from the `grid_x`,
   `grid_y` columns, which are a different coordinate. Each section is cropped
   to the bounding box of its own spots.
2. Pad to 96 centred. The hand-painted wound coordinates live in that frame.
3. Stamp the wound, in TWO layers, from `annotations/wounds/`. The painted
   region marks where the wound IS, not a hole to be excised: 85% of it at 6 hpa
   and 95% from 3 dpa is intact tissue at full cell mass. So the annotation
   never destroys cells.
     - to 1 dpa: the bins the painted region leaves EMPTY are the genuine hole.
       They take damage 1.0 and join the section mask -- they have to, or the
       silhouette treats the wound as outside the tissue and shape collapses.
       The reference layer takes the whole painted footprint.
     - from 3 dpa: the cohort has no clot, so no damage is written at all and
       the reference takes only the filled part. The residual empty bins are
       treated as uncaptured and stay OUTSIDE the heart: giving them mask 1 with
       neither cells nor damage would put empty bins inside the tissue and fight
       the capacity loss for nothing.
4. Densify under the mask -- normalised convolution at sigma 1, so void bins
   stay exactly zero and the wound rim is as soft as the tissue around it.
5. Sum-pool 96 -> 48, as an area interpolation rescaled by the area ratio. This
   is why a full interior bin holds 4.0 and the damage ceiling is 4.

The output is checked against the invariants of the tensor the paper trained on
before it is written, so a wrong or re-released download fails here rather than
after a day of training.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from workspace import WORK
from domains.hearts import cohort as C
from domains.hearts.wounds import ANNOTATIONS, CANVAS

BARCODE = re.compile(r"cell(\d+)_(\d+)$")
SPOTS = WORK / "data/hearts/Stereo-seq-regeneration.meta.txt"   # the download
GRID = 48
SIGMA = 1.0
DAMAGE_UNTIL = "1 dpa"          # the cohort shows no clot after this stage

# Measured on the tensor the paper's runs trained on. A producer that agrees
# with these is producing the same cohort.
EXPECT_SECTIONS = 105
EXPECT_DAMAGE = {"uninjured": 0.0, "6 hpa": 0.0244, "12 hpa": 0.0604,
                 "1 dpa": 0.0464, "3 dpa": 0.0, "7 dpa": 0.0,
                 "14 dpa": 0.0, "28 dpa": 0.0}


def _sections(tsv: Path):
    """Yield (section_id, timepoint, comp (20,h,w), mask (1,h,w)) per section.

    Read with the csv module rather than pandas: the table is 160k rows of
    twelve columns and four of them are used, which is not worth a dependency
    the rest of this repo does not have.
    """
    idx = {n: i for i, n in enumerate(C.CELL_TYPES)}
    per_section: dict[str, list] = {}
    stage: dict[str, str] = {}
    with open(tsv, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        head = next(reader)
        # the row-name column is unnamed, so the header is one field short
        cols = {n: i + 1 for i, n in enumerate(head)}
        for need in ("orig.ident", "time_points", "annotation"):
            assert need in cols, f"the table lacks a {need!r} column"
        ci, ct, ca = cols["orig.ident"], cols["time_points"], cols["annotation"]
        for row in reader:
            m = BARCODE.search(row[0])
            assert m, f"barcode {row[0]!r} does not match cell<row>_<col>"
            ann = row[ca]
            assert ann in idx, f"unexpected annotation {ann!r}"
            sid = row[ci]
            per_section.setdefault(sid, []).append(
                (int(m.group(1)), int(m.group(2)), idx[ann]))
            stage.setdefault(sid, row[ct])

    for sid, spots in per_section.items():
        a = np.asarray(spots)                     # (n, 3): row, col, channel
        r, c, ch = a[:, 0], a[:, 1], a[:, 2]
        r0, c0 = r.min(), c.min()
        h, w = int(r.max() - r0 + 1), int(c.max() - c0 + 1)
        comp = torch.zeros(len(C.CELL_TYPES), h, w)
        mask = torch.zeros(1, h, w)
        comp[ch, r - r0, c - c0] = 1.0
        mask[0, r - r0, c - c0] = 1.0
        yield sid, stage[sid], comp, mask


def _pad(t: torch.Tensor, n: int) -> torch.Tensor:
    _, h, w = t.shape
    assert h <= n and w <= n, f"section {(h, w)} exceeds the {n}px canvas"
    out = torch.zeros(t.shape[0], n, n)
    top, left = (n - h) // 2, (n - w) // 2
    out[:, top:top + h, left:left + w] = t
    return out


def _densify(comp: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Normalised convolution: blur signal and mask by the same kernel and
    divide, so the result stays exactly zero outside the tissue."""
    rad = max(1, int(round(3.0 * sigma)))
    t = torch.arange(-rad, rad + 1, dtype=torch.float32)
    k = torch.exp(-(t ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()

    def blur(x):                                   # x: (C, H, W)
        c = x.shape[0]
        y = F.conv2d(x[None], k.view(1, 1, 1, -1).expand(c, 1, 1, -1),
                     padding=(0, rad), groups=c)
        y = F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1),
                     padding=(rad, 0), groups=c)
        return y[0]

    return blur(comp * mask) / blur(mask).clamp_min(1e-8) * mask


def _pool(t: torch.Tensor, n: int) -> torch.Tensor:
    """96 -> 48 preserving per-channel mass, i.e. a 2x2 sum."""
    src = t.shape[-1]
    y = F.interpolate(t[None], size=(n, n), mode="area")[0]
    return y * (src * src) / float(n * n)


def _painted(section_id: str, wounds_dir: Path) -> np.ndarray | None:
    f = wounds_dir / f"{section_id}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    m = np.zeros((CANVAS, CANVAS), dtype=bool)
    for r, c in d.get("damage_pixels", []):
        if 0 <= r < CANVAS and 0 <= c < CANVAS:
            m[r, c] = True
    return m if m.any() else None


def build(tsv: Path, wounds_dir: Path, grid: int = GRID, sigma: float = SIGMA):
    late_from = C.TP_ORDER.index(DAMAGE_UNTIL)
    out = []
    for sid, tp, comp, mask in _sections(tsv):
        assert tp in C.TIMEPOINTS, f"{sid}: unknown timepoint {tp!r}"
        comp, mask = _pad(comp, CANVAS), _pad(mask, CANVAS)
        # two extra channels: the hole the generator sees, and the persistent
        # site only the critic sees
        comp = torch.cat([comp, torch.zeros(2, CANVAS, CANVAS)], dim=0)

        painted = _painted(sid, wounds_dir)
        if painted is not None:
            tissue = mask[0].numpy() > 0
            if C.TP_ORDER.index(tp) > late_from:
                ref = painted & tissue
            else:
                empty = torch.from_numpy(painted & ~tissue)
                comp[-2][empty] = 1.0
                mask[0][empty] = 1.0
                ref = painted
            comp[-1][torch.from_numpy(ref)] = 1.0

        comp = _densify(comp, mask, sigma).clamp(0.0, 1.0)
        out.append({"section_id": sid, "timepoint": tp,
                    "hours": C.TIMEPOINTS[tp],
                    "composition": _pool(comp, grid),
                    "mask": _pool(mask, grid).clamp_(0.0, 1.0)})
    return out


def check(sections: list[dict]) -> None:
    """Fail here rather than after a day of training."""
    assert len(sections) == EXPECT_SECTIONS, \
        f"{len(sections)} sections, expected {EXPECT_SECTIONS}"
    per_stage: dict[str, list[float]] = {}
    for s in sections:
        comp, m = s["composition"], s["mask"][0] > 0
        vis = comp[:len(C.CELL_TYPES) + 1]
        assert float(comp[:, ~m].abs().max()) < 1e-5, \
            f"{s['section_id']}: non-zero composition outside the heart"
        cap = vis[:, m].sum(dim=0)
        assert float(cap.max()) <= 4.0 + 1e-3, \
            f"{s['section_id']}: bin capacity {float(cap.max()):.3f} exceeds 4"
        per_stage.setdefault(s["timepoint"], []).append(
            float(vis[len(C.CELL_TYPES)][m].mean()))
    for tp, want in EXPECT_DAMAGE.items():
        got = float(np.mean(per_stage[tp]))
        assert abs(got - want) < 0.004, \
            (f"{tp}: within-mask mean damage {got:.4f}, expected about {want:.4f}. "
             "The download or the wound annotations do not match the cohort the "
             "paper trained on.")
    print(f"  checks passed: {len(sections)} sections, capacity <= 4, "
          f"per-stage damage as measured")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    sections = build(SPOTS, ANNOTATIONS / "wounds")
    check(sections)
    out = C.PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    comp = np.stack([s["composition"].numpy() for s in sections]).astype(np.float32)
    np.savez_compressed(
        out,
        # packed as x/2 - 1 into [-1, 1]; cohort.py inverts it
        composition=(comp / 2.0 - 1.0),
        mask=np.stack([s["mask"][0].numpy() for s in sections]).astype(np.uint8),
        channels=np.array(C.CELL_TYPES + [C.DAMAGE_CHANNEL, C.REF_CHANNEL]),
        section_id=np.array([s["section_id"] for s in sections]),
        timepoint=np.array([s["timepoint"] for s in sections]),
        hours=np.array([s["hours"] for s in sections], dtype=np.float32))
    print(f"wrote {out}  ({len(sections)} sections, {comp.shape[1]} channels, "
          f"{GRID}x{GRID})")


if __name__ == "__main__":
    main()
