"""Injuries: where a wound may land, what shape it takes, and what it does.

The atlas carries no annotation of injury location, so the wound footprint of
every injured section was hand-painted -- 87 masks, one per injured section,
which no producer can regenerate from the atlas. They ship with this repo
(`annotations/wounds/`), together with 18 apex zones painted on the uninjured
sections (`annotations/zones/`) that say where on an uninjured heart a new
wound is allowed to land. Both are on the native 96 grid.

Training does not use procedural capsules. A capsule is measurably rounder and
less elongated than a real amputation -- Cohen's d = -0.64 on circularity and
+0.40 on elongation against the 87 hand masks -- which is a weak difference on
its own but becomes a free win for the critic the moment it is shown the wound
footprint directly, because it is a feature the generator cannot change.
Drawing the footprint from the annotations removes the difference by
construction: a real footprint is flipped, rotated and transplanted onto
another heart's apex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ANNOTATIONS = Path(__file__).resolve().parent / "annotations"
CANVAS = 96                  # the grid the annotations were painted on


# ---- where a wound may land ------------------------------------------------

def _rescale(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize of a 0/1 grid onto (h, w)."""
    sh, sw = grid.shape
    if (sh, sw) == (h, w):
        return grid.astype(np.uint8)
    rs = (np.arange(h) * (sh / h) + 0.5 * sh / h).astype(np.int32).clip(0, sh - 1)
    cs = (np.arange(w) * (sw / w) + 0.5 * sw / w).astype(np.int32).clip(0, sw - 1)
    return grid[rs[:, None], cs[None, :]].astype(np.uint8)


def load_zones(seed_ids: list[str], h: int, w: int, device,
               zones_dir=ANNOTATIONS / "zones"):
    """-> (usable, grids, tips): which uninjured sections carry an apex zone,
    that zone rescaled to the state grid, (K, 1, h, w), and the apex tip at
    state scale, (K, 2).

    A section without a zone cannot seed a wounded trajectory, so it is dropped
    from the seed bank rather than wounded somewhere arbitrary. Training only
    needs the zone; the evaluation anchors its capsule on the tip.
    """
    zones = {}
    for f in sorted(Path(zones_dir).glob("*.json")):
        d = json.loads(f.read_text())
        pix = np.array(d.get("zone_pixels", []), dtype=np.int32)
        if pix.size and d.get("tip") is not None:
            zones[d["section_id"]] = (pix, d["tip"])
    usable, grids, tips = [], [], []
    for i, sid in enumerate(seed_ids):
        z = zones.get(sid)
        if z is None:
            continue
        pix, tip = z
        g = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
        g[np.clip(pix[:, 0], 0, CANVAS - 1), np.clip(pix[:, 1], 0, CANVAS - 1)] = 1
        grids.append(_rescale(g, h, w))
        tips.append([int(np.clip(round((tip[0] + 0.5) * h / CANVAS - 0.5), 0, h - 1)),
                     int(np.clip(round((tip[1] + 0.5) * w / CANVAS - 0.5), 0, w - 1))])
        usable.append(i)
    assert usable, f"no uninjured section in {zones_dir} carries an apex zone"
    print(f"  zones: {len(usable)} of {len(seed_ids)} uninjured sections can be wounded")
    return (usable, torch.from_numpy(np.stack(grids)).unsqueeze(1).to(device),
            torch.tensor(tips, dtype=torch.long, device=device))


# ---- what shape it takes ---------------------------------------------------

def load_wound_shapes(h: int, w: int, min_bins: int = 4,
                      wounds_dir=ANNOTATIONS / "wounds") -> list[np.ndarray]:
    """The hand-painted footprints, each cropped to its own bounding box so it
    can be dropped anywhere on a target heart.

    Pooled from 96 to 48 by 2x2 MAX, not resampled: a wound bin stays a wound
    bin. (The composition is sum-pooled instead -- that is a density, this is a
    footprint.)
    """
    out = []
    for f in sorted(Path(wounds_dir).glob("*.json")):
        d = json.loads(f.read_text())
        sh, sw = d.get("shape", [CANVAS, CANVAS])
        m = np.zeros((sh, sw), dtype=bool)
        for r, c in d.get("damage_pixels", []):
            if 0 <= r < sh and 0 <= c < sw:
                m[r, c] = True
        if m.sum() < min_bins:
            continue
        if (sh, sw) != (h, w):
            fy, fx = sh // h, sw // w
            if fy < 1 or fx < 1:
                continue
            m = m[:h * fy, :w * fx].reshape(h, fy, w, fx).any(axis=(1, 3))
        ys, xs = np.nonzero(m)
        if len(ys):
            out.append(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy())
    assert out, f"no usable wound footprints in {wounds_dir}"
    return out


def sample_shaped_wound(zone: np.ndarray, tissue: np.ndarray, rng,
                        shapes: list[np.ndarray], max_tries: int = 40) -> np.ndarray:
    """Place one footprint on this heart's apex; -> (H, W) bool damage grid.

    The footprint is flipped and rotated by a multiple of 90 degrees (the
    annotations have no canonical orientation) and centred on a random allowed
    bin. Placements are scored by how much of the shape lands inside the zone,
    and the best of `max_tries` is kept rather than retrying to failure, so an
    awkward section never stalls training.
    """
    tis = np.asarray(tissue) > 0
    allowed = (np.asarray(zone) > 0) & tis
    if not allowed.any():
        allowed = tis
    ys, xs = np.nonzero(allowed)
    h, w = tis.shape
    best, best_score = None, -1.0
    for _ in range(max_tries):
        sh = shapes[int(rng.integers(len(shapes)))]
        k = int(rng.integers(4))
        if k:
            sh = np.rot90(sh, k)
        if rng.random() < 0.5:
            sh = sh[::-1]
        if rng.random() < 0.5:
            sh = sh[:, ::-1]
        sy, sx = sh.shape
        p = int(rng.integers(len(ys)))
        r0 = int(np.clip(ys[p] - sy // 2, 0, max(h - sy, 0)))
        c0 = int(np.clip(xs[p] - sx // 2, 0, max(w - sx, 0)))
        m = np.zeros((h, w), dtype=bool)
        m[r0:r0 + sy, c0:c0 + sx] = sh[:min(sy, h - r0), :min(sx, w - c0)]
        inside = m & tis
        if not inside.any():
            continue
        score = float((m & allowed).sum()) / float(m.sum())
        if score > best_score:
            best_score, best = score, inside
        if score > 0.9:
            break
    return allowed if best is None else best


# ---- the evaluation's wound ------------------------------------------------
# Training cuts with a transplanted real footprint (above). The evaluation cuts
# with a PARAMETERISED capsule instead, and deliberately so: a knockout effect
# is only comparable across conditions if every condition is cut the same way,
# and a standardised wound is what makes the size of the injury a constant
# rather than a draw from 86 annotations. Its footprint covers 7.71% of the
# heart against a real damaged area of 6.45-7.95%.
#
# Two kinds, as in the published screens: with probability `p_stab` a capsule
# entering on the zone boundary and aimed inward at the zone centroid, else a
# disc centred on the apex tip. Radii are in bins at the 96 canvas, so they
# halve on the 48 grid. Both are damage-only -- the erase core the interactive
# tool offers is never used by an assay.

STAB_P = 0.20
CHUNK_R_MU, CHUNK_R_SIGMA = 10.0, 0.30
STAB_R = (3.5, 8.0)
STAB_LEN = (6.0, 24.0)
STAB_ANGLE_JITTER = 0.90


def _boundary(zone: np.ndarray) -> np.ndarray:
    g = zone > 0
    pad = np.pad(g, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return np.stack(np.where(g & ~inner), axis=1)


def sample_capsule_wound(zone: np.ndarray, tissue: np.ndarray, tip, rng,
                         scale: float = 0.5) -> np.ndarray:
    """-> (H, W) bool damage grid. `scale` converts the 96-canvas radii to the
    state grid."""
    zone_b, tis = np.asarray(zone) > 0, np.asarray(tissue) > 0
    if rng.random() < STAB_P:
        b = _boundary(zone_b)
        rr, cc = np.where(zone_b)
        p0 = (b[rng.integers(len(b))].astype(np.float64) if len(b)
              else np.array([rr[0], cc[0]], dtype=np.float64))
        vec = np.array([rr.mean(), cc.mean()]) - p0
        n = float(np.linalg.norm(vec))
        if n < 1e-6:
            th = rng.uniform(0.0, 2.0 * np.pi)
            vec = np.array([np.cos(th), np.sin(th)])
        else:
            vec = vec / n
        th = rng.uniform(-STAB_ANGLE_JITTER, STAB_ANGLE_JITTER)
        c_, s_ = float(np.cos(th)), float(np.sin(th))
        vec = np.array([vec[0] * c_ - vec[1] * s_, vec[0] * s_ + vec[1] * c_])
        p1 = p0 + vec * rng.uniform(*[v * scale for v in STAB_LEN])
        r = rng.uniform(*[v * scale for v in STAB_R])
    else:
        # inner_frac is 0 for every assay, so the jitter is scaled to zero and
        # the disc sits exactly on the tip
        # the clip is in STATE bins and is not scaled with the mean: it binds on
        # roughly 4% of draws at the 48-grid mean of 5.0, so scaling it too
        # would quietly widen the small tail of eval wounds
        r = float(np.clip(rng.lognormal(mean=np.log(CHUNK_R_MU * scale),
                                        sigma=CHUNK_R_SIGMA), 3.0, 28.0))
        p0 = p1 = np.asarray(tip, dtype=np.float64)

    rr, cc = np.where(zone_b & tis)
    out = np.zeros(tis.shape, dtype=bool)
    if not len(rr):
        return out
    pts = np.stack([rr, cc], axis=1).astype(np.float64)
    ab = p1 - p0
    n2 = float(ab @ ab)
    proj = (np.broadcast_to(p0, pts.shape) if n2 < 1e-9 else
            p0 + np.outer(np.clip(((pts - p0) @ ab) / n2, 0.0, 1.0), ab))
    inside = np.linalg.norm(pts - proj, axis=1) <= r
    out[rr[inside], cc[inside]] = True
    return out


# ---- what it does to the state ---------------------------------------------

def _densify(x: torch.Tensor, sigma: float, mask: torch.Tensor) -> torch.Tensor:
    """Normalised convolution under the mask: blur the signal and the mask by
    the same Gaussian and divide. Void bins stay exactly zero, so the blur
    never leaks tissue outside the heart."""
    radius = max(1, int(round(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-(t ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    pad = radius

    def blur(v):
        b, c, hh, ww = v.shape
        v = v.reshape(b * c, 1, hh, ww)
        v = F.conv2d(v, k.view(1, 1, 1, -1), padding=(0, pad))
        v = F.conv2d(v, k.view(1, 1, -1, 1), padding=(pad, 0))
        return v.reshape(b, c, hh, ww)

    return blur(x * mask) / blur(mask).clamp_min(1e-8) * mask


def stamp(state: torch.Tensor, mask: torch.Tensor, damage: torch.Tensor,
          *, n_visible: int, damage_idx: int, alive_idx: int,
          sigma: float = 0.5, alive_threshold: float = 0.10) -> torch.Tensor:
    """Apply a wound to a batched rule state; -> the new state.

    At a wounded bin every cell type is cleared and damage takes over the
    bin's whole capacity. Pinning damage to 1.0 instead would delete three
    quarters of a 48-grid bin's capacity and teach the rule, on every wound,
    that capacity is not conserved -- the real data conserves it exactly
    (median mass + damage is 4.000 at every damage level).

    The wounded slots are then densified and the alive channel re-derived from
    the blurred damage, the same construction the real sections get, so the
    stamp leaves no hard binary edge for the critic to key on.
    """
    dmg = damage.to(torch.bool)
    fired = dmg.any(dim=3).any(dim=2).any(dim=1).view(-1, 1, 1, 1)
    if not bool(fired.any()):
        return state
    state = state.clone()
    ch = torch.arange(state.shape[1], device=state.device)
    capacity = state[:, :n_visible].sum(dim=1, keepdim=True)
    state[((ch < n_visible) & (ch != damage_idx)).view(1, -1, 1, 1) & dmg] = 0.0
    flat = dmg.squeeze(1)
    state[:, damage_idx][flat] = capacity[:, 0][flat]

    vis = torch.where(fired, _densify(state[:, :n_visible], sigma, mask),
                      state[:, :n_visible])
    alive = torch.where(fired,
                        mask * (vis[:, damage_idx:damage_idx + 1] < alive_threshold).to(state.dtype),
                        state[:, alive_idx:alive_idx + 1])
    return torch.cat([vis, alive, state[:, alive_idx + 1:]], dim=1)
