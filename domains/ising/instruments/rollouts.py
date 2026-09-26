"""Shared initial fields and storage format for Ising rollout caches.

Chunk j contains 64 fields drawn with torch seed 1234 + j. All models use
the same initial fields; smaller ensembles use a prefix of these chunks.
Caches are stored at data/rollouts/<label>__lat<lat>_ns<ns>.npz:

    bits  (n_times, ns * lat * lat / 8) uint8, packed in C order; 1 = spin up
    ks    (n_times,) readout times in sweeps
    meta  JSON metadata, including lat and ns
    raw   optional (n_times, ns, lat, lat) float16 states on the 0..1 scale;
          clamped to [0, 1] except for MMSFM"""
from __future__ import annotations

import json

import numpy as np
import torch

from workspace import WORK

CHUNK = 64
# every read-out time a grader or figure uses, to t = 4000
READOUT_TIMES = [3, 5, 7, 10, 13, 17, 24, 30, 42, 55, 74, 100, 130, 150, 175, 231, 300, 408,
                 550, 700, 850, 961, 1000, 1278, 1400, 1700, 2000, 2261, 2800, 3007, 4000]


def chunks(ns):
    """(j, number of fields) for each chunk of an ns-field ensemble"""
    return [(j, min(CHUNK, ns - CHUNK * j)) for j in range(-(-ns // CHUNK))]


def quench(j, n, lat, dev):
    """the first n fields of chunk j: bool (n, lat, lat), True = spin up"""
    torch.manual_seed(1234 + j)
    # the whole chunk is drawn, so a smaller ensemble starts from the same first fields
    return torch.rand(CHUNK, lat, lat, device=dev)[:n] < 0.5


def between_marginals(marginal_times, steps):
    """Configurations between marginals: a transport sampler crosses each
    transition between consecutive marginal times in `steps` equal solver
    steps; its state part-way through a transition is read out and placed at
    the simulation time reached by advancing linearly along the transition.
    Returns, per transition, {solver step: READOUT_TIMES time strictly inside
    it, read at the nearest step before its end}; and every read-out time,
    i.e. those plus the marginal times after t = 0."""
    inside = []
    for ka, kb in zip(marginal_times, marginal_times[1:]):
        at = {}
        for k in READOUT_TIMES:
            i = max(1, round(steps * (k - ka) / (kb - ka)))
            if ka < k < kb and i < steps:
                at.setdefault(i, k)
        inside.append(at)
    return inside, sorted(set(marginal_times[1:]) | {k for at in inside for k in at.values()})


def load(p):
    """a cache's bits (decompressed once: every access to an npz member
    decompresses it again), read-out times and meta"""
    z = np.load(p)
    return z["bits"], [int(k) for k in z["ks"]], json.loads(str(z["meta"]))


def unpack(row, lat, n):
    """the first n fields of one read-out's packed bits: uint8 (n, lat, lat)"""
    return np.unpackbits(row[:n * lat * lat // 8]).reshape(n, lat, lat)


def save(label, lat, ns, ks, bits, raw=None, **meta):
    """bits (and raw): for each read-out time, the list of per-batch arrays"""
    p = WORK / "data" / "rollouts" / f"{label}__lat{lat}_ns{ns}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if raw is None else {"raw": np.stack([np.concatenate(r) for r in raw])}
    np.savez_compressed(p, bits=np.stack([np.concatenate(b) for b in bits]), ks=np.array(ks),
                        meta=json.dumps(dict(lat=lat, ns=ns, **meta)), **extra)
    print(f"wrote {p}: {len(ks)} read-out times, {ns} fields", flush=True)
