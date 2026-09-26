"""The zebrafish cohort: 105 Stereo-seq sections on a 48x48 grid.

Eight sampling stages of heart regeneration (105 sections from 35 fish, three
serial slices each, 4-6 fish per stage), annotated to 20 mutually exclusive
cell types. Spots are one-hot, binned by the grid index in their barcode,
densified under the section mask, then 2x2 SUM-pooled from the native 96 grid
-- so a fully occupied interior bin sums to 4.0, not 1.0, and bins carry
mixtures rather than single types. That capacity of 4 is why the damage
channel's ceiling is 4 and not 1.

Twenty-two channels on disk:
    0..19   the annotated cell types, in cluster-id order
    20      damage, the physical hole -- the generator sees this; it is the
            trigger, and it is stamped at runtime so a wound can be introduced
            at any time
    21      the damage REFERENCE, the persistent wound site -- present at every
            stage once the wound exists, including 28 dpa. The CRITIC sees this
            and the generator never does. Localisation is a relation between
            the response and the injury, and a critic scoring composition alone
            has no reference to judge it against: "BZm in a blob somewhere"
            would score like "BZm in the wound".

The rule's state adds an alive channel and six hidden ones on top of the 21 the
generator sees (see `expand`), for the 28 of `rule.py`.

The release reads the packaged `hearts_48.npz`; `.pt` blobs in the research
tree's format load too, so a run can be checked against the original.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from workspace import WORK

CELL_TYPES = [
    "1:BZm (activ.)", "2:BZm (dediff.)", "3:BZm (dediff.-prolif.)",
    "4:BZm (prolif.)", "5:BZm (re-diff.)", "6:RZm1", "7:RZm2", "8:Vm (comp.)",
    "9:Am", "10:VAm", "11:MC", "12:FB (reg.)", "13:MFE", "14:Epi", "15:SMC",
    "16:Valves", "17:RBC", "18:RBC (V)", "19:RBC (A)", "20:Others",
]
DAMAGE_CHANNEL = "21:Damage"
REF_CHANNEL = "22:DamageRef"

# the eight stages and their hours post-amputation
TIMEPOINTS = {"uninjured": 0.0, "6 hpa": 6.0, "12 hpa": 12.0, "1 dpa": 24.0,
              "3 dpa": 72.0, "7 dpa": 168.0, "14 dpa": 336.0, "28 dpa": 672.0}
TP_ORDER = list(TIMEPOINTS)
N_STAGES = len(TP_ORDER)

ALIVE_DAMAGE_THR = 0.10   # a bin is alive if it is tissue and not (yet) wounded
PATH = WORK / "data/hearts/hearts_48.npz"      # what data_cohort.py writes


# ---- the chapter's a-priori vocabulary --------------------------------------
# Fixed from published loss-of-function evidence BEFORE the screen was run, so
# the group comparison is a test and not a description.
#
#   A  regeneration programme -- the border-zone cardiomyocyte cascade that
#      rebuilds muscle. Jopling 2010 and Kikuchi 2010 established
#      dedifferentiation plus proliferation of border-zone CMs as the mechanism.
#   B  regulatory stroma -- non-muscle populations with loss-of-function
#      evidence: col12a1a+ pro-regenerative fibroblasts (ablation reduces CM
#      proliferation, Sanchez-Iranzo 2022), macrophages (clodronate depletion
#      impairs regeneration, Lai 2017), epicardium. MFE sits here provisionally.
#   C  blood / unassigned -- red blood cells and the catch-all cluster, with no
#      established regenerative function.
#   D  constitutive structure -- remote-zone and chamber myocardium, smooth
#      muscle, valves: present before injury, not part of the response.
#
# A and B are the implicated side of both contrasts. NOTE the letters: the
# research tree had C and D the other way round and four scripts each carried
# their own relabel map, one of them inverted. This is the paper's convention
# and the only one in the release.
GROUP = {
    "1:BZm (activ.)": "A", "2:BZm (dediff.)": "A", "3:BZm (dediff.-prolif.)": "A",
    "4:BZm (prolif.)": "A", "5:BZm (re-diff.)": "A",
    "11:MC": "B", "12:FB (reg.)": "B", "13:MFE": "B", "14:Epi": "B",
    "17:RBC": "C", "18:RBC (V)": "C", "19:RBC (A)": "C", "20:Others": "C",
    "6:RZm1": "D", "7:RZm2": "D", "8:Vm (comp.)": "D", "9:Am": "D",
    "10:VAm": "D", "15:SMC": "D", "16:Valves": "D",
}
GROUP_NAME = {"A": "regeneration programme (BZm cascade)",
              "B": "regulatory stroma", "C": "blood / unassigned",
              "D": "constitutive structure"}
# the two pre-specified contrasts, implicated side first
CONTRASTS = [("AB_CD", ("A", "B"), ("C", "D")), ("A_D", ("A",), ("D",))]

# The border-zone cascade, in the order the atlas describes it: each state is
# enriched in a narrow window (6 h; 12 h and 1 d; 3 d; 7 d; 14 d). Any model
# that fits the marginals reproduces the succession; what a knockout adds is the
# DEPENDENCY between the states.
CASCADE = CELL_TYPES[:5]
# cell types prior biology does not implicate in the wound response. Zeroing any
# channel drives the state off-distribution, so without this control a general
# derailment would be indistinguishable from a captured dependency.
CASCADE_CONTROLS = ["9:Am", "16:Valves", "19:RBC (A)"]
# a catch-all for spots the atlas could not assign, not a cell type: a knockout
# of it has no biological reading, so it is dropped from the group statistics
NOT_A_CELL_TYPE = "20:Others"


def _read(path: Path) -> list[dict]:
    """-> one dict per section: composition (22,48,48), mask (1,48,48),
    section_id, timepoint, hours."""
    path = Path(path)
    if path.suffix == ".pt":
        blob = torch.load(path, weights_only=False, map_location="cpu")
        assert blob["channels"] == CELL_TYPES + [DAMAGE_CHANNEL, REF_CHANNEL], \
            "unexpected channel list; this is not the two-layer cohort"
        return blob["sections"]
    z = np.load(path, allow_pickle=False)
    assert list(z["channels"]) == CELL_TYPES + [DAMAGE_CHANNEL, REF_CHANNEL], \
        "unexpected channel list; this is not the two-layer cohort"
    comp = 2.0 * (z["composition"].astype(np.float32) + 1.0)   # packed as x/2 - 1
    return [{"composition": torch.from_numpy(comp[i]),
             "mask": torch.from_numpy(z["mask"][i].astype(np.float32))[None],
             "section_id": str(z["section_id"][i]),
             "timepoint": str(z["timepoint"][i]),
             "hours": float(z["hours"][i])} for i in range(len(comp))]


def load_cohort(path, device, holdout_hearts=()):
    """-> (seeds, seed_ids, real_banks, ref_banks).

    `holdout_hearts` takes isolate ids (T#_S#) and drops every section of those
    fish, from BOTH the critic's real banks and the uninjured seed pool. Holding
    out a section instead would leak: the other two slices are serial sections
    of the same three-dimensional heart.

    real_banks[tp] = (composition (n, 21, H, W), mask (n, 1, H, W)) -- the
    damage reference is split off into ref_banks[tp] (n, 1, H, W), because the
    generator must never be handed it.
    """
    sections = _read(path)
    if holdout_hearts:
        def heart_of(s):
            t, _chip, fish = s["section_id"].split("_")
            return f"{t}_{fish}"
        want = set(holdout_hearts)
        unknown = want - {heart_of(s) for s in sections}
        assert not unknown, f"no such heart(s) {sorted(unknown)}"
        kept = [s for s in sections if heart_of(s) not in want]
        print(f"  held out {sorted(want)}: {len(sections) - len(kept)} sections, "
              f"training on {len(kept)} of {len(sections)}")
        sections = kept

    by_tp: dict[str, list] = {}
    for s in sections:
        by_tp.setdefault(s["timepoint"], []).append(s)
    real_banks, ref_banks = {}, {}
    for tp in TP_ORDER:
        if tp not in by_tp:
            continue
        comp = torch.stack([s["composition"] for s in by_tp[tp]]).to(device)
        real_banks[tp] = (comp[:, :-1].contiguous(),
                          torch.stack([s["mask"] for s in by_tp[tp]]).to(device))
        ref_banks[tp] = comp[:, -1:].contiguous()
    seeds = real_banks["uninjured"]
    seed_ids = [s["section_id"] for s in by_tp["uninjured"]]
    return seeds, seed_ids, real_banks, ref_banks


def expand(comp: torch.Tensor, mask: torch.Tensor, n_hidden: int) -> torch.Tensor:
    """(n, 21, H, W) observation -> (n, 21 + 1 + n_hidden, H, W) rule state.

    The alive channel is tissue that is not wounded; the hidden channels start
    at zero and are written only by the model, which is the only place a phase
    signal can live once the rule is denied a clock.
    """
    damage = comp[:, -1:]
    alive = mask * (damage < ALIVE_DAMAGE_THR).to(comp.dtype)
    hidden = comp.new_zeros(comp.shape[0], n_hidden, *comp.shape[2:])
    return torch.cat([comp, alive, hidden], dim=1).contiguous()


def zero_mask(real_banks: dict, n_visible: int, device, tol: float = 1e-6) -> torch.Tensor:
    """(N_STAGES, n_visible): 1 where the cohort has that channel identically
    zero at that stage. The zero loss reads it, so a cell type never observed at
    a stage is explicitly punished rather than merely unrewarded.

    Damage is always exempt. It is the one visible channel already policed from
    two directions -- the rule's monotone cap forbids growth, and damage above
    the alive threshold destroys the alive channel, which the critic sees and
    the heal loss penalises -- so scoring it here would police it a third time.
    """
    m = torch.zeros(N_STAGES, n_visible, device=device)
    for i, tp in enumerate(TP_ORDER):
        if tp in real_banks:
            m[i] = (real_banks[tp][0].amax(dim=(0, 2, 3)) <= tol).float()
    m[:, n_visible - 1] = 0.0
    return m


def bin_capacity(comp: torch.Tensor) -> torch.Tensor:
    """(n, 1, H, W) per-bin capacity: what the capacity loss holds each bin to.
    Measured on the state itself, so a wound that conserves capacity does not
    move the target."""
    return comp.sum(dim=1, keepdim=True)


class Augment(torch.nn.Module):
    """Geometric augmentation only: rotation, independent flips, and an
    elastic deformation, all NEAREST with zero fill so a bin's composition is
    never blended into its neighbour's.

    Every plane goes through ONE sampled transform, concatenated and split
    again. That matters for the two passengers: the critic-only damage
    reference would stop corresponding to the section it labels if it drew its
    own rotation, and a pending wound would stop corresponding to the tissue it
    is about to cut.
    """

    def __init__(self) -> None:
        super().__init__()
        import torchvision.transforms.v2 as T
        near = T.InterpolationMode.NEAREST
        self.transform = T.Compose([
            T.RandomRotation(degrees=(-180, 180), interpolation=near, fill=0.0),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ElasticTransform(alpha=50.0, sigma=5.0, interpolation=near, fill=0.0),
        ])

    def forward(self, x, mask, extra=None, damage=None):
        parts = [x, mask] + [p for p in (extra, damage) if p is not None]
        out = self.transform(torch.cat(parts, dim=1))
        c = x.shape[1]
        res = [out[:, :c].contiguous(), out[:, c:c + mask.shape[1]].contiguous()]
        i = c + mask.shape[1]
        if extra is not None:
            res.append(out[:, i:i + extra.shape[1]].contiguous()); i += extra.shape[1]
        if damage is not None:
            # re-threshold: NEAREST plus elastic can round a 0/1 grid off
            res.append((out[:, i:i + damage.shape[1]] > 0.5).to(x.dtype).contiguous())
        return tuple(res)
