"""Generate Ising training data with Glauber heat-bath dynamics at T = 1.

    marginals  Independent full fields at each observation time (REGIS/control).
    windows    192x192 crops on the [-1, 1] scale (MMSFM/MMtSBM).
    pairs      Aligned states separated by 10..1000 sweeps (DDPM).

Marginal files share fields at shared times; smaller data budgets use a
prefix of the same fields. Random draws depend on the device and PyTorch
version. Commands and output paths are in docs/reproduce.md."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from domains.ising.glauber import sweep_
from workspace import WORK

LAT = 512
TIMES = [3, 10, 17, 300, 1000]
N_BASE = 100
PER_FIELD = 2                          # windows cut from each field
WINDOW_SEED, PAIR_SEED = 31000, 77000
JUMP_MIN = 10                          # the DDPM's smallest trained jump, in sweeps
CHUNK = 128                            # part of the seed rule: chunk b of a time is seeded from its offset
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# the fields at time t of every marginals file come from seed_of(t), so a time
# shared by two marginals files has the same fields in both (seeds numbered
# along the ten-time grid)
_SEED_ORDER = [3, 10, 17, 30, 55, 100, 175, 300, 550, 1000]


def seed_of(t):
    return 100_000 * (_SEED_ORDER.index(t) + 1) if t in _SEED_ORDER else 50_000_000 + 1000 * t


def coarsened(n, t, seed, lat=LAT):
    """n independent iid quenches evolved t sweeps: uint8 {0, 1}, (n, lat, lat)."""
    torch.manual_seed(seed)
    s = torch.where(torch.rand(n, lat, lat, device=DEV) < 0.5, 1.0, -1.0)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(t):
        sweep_(s, 1.0, gen)
    return (s > 0).to(torch.uint8).cpu().numpy()


def marginals(times, n_train=N_BASE, lat=LAT):
    # a smaller budget is the first F fields of the 100-field file (the
    # trainer's --n-train), never a separate draw
    assert n_train >= N_BASE, f"n_train below {N_BASE}: use --n-train in the trainer"
    out = {"stage_ts": np.array(times), "lat": lat}
    for i, t in enumerate(times, start=1):
        fields = [coarsened(N_BASE, t, seed_of(t), lat)]
        if n_train > N_BASE:
            fields.append(coarsened(n_train - N_BASE, t, seed_of(t) + 50_000, lat))
        out[f"train_s{i}"] = np.concatenate(fields)
        print(f"t = {t}: {n_train} fields", flush=True)
    return out


def windows(out, window=192, n_fields=3000, n_eval=512, lat=LAT):
    """Writes out/{ising,eval}_t<i>.pt, +-1 float (N, 1, window, window), for
    t_0 = 0 (iid spins) and then TIMES; independent fields per time, so the
    marginals leak no pairing."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(WINDOW_SEED)

    def crops(fields):
        w = np.empty((len(fields) * PER_FIELD, window, window), np.uint8)
        for j in range(len(w)):
            oy, ox = rng.integers(0, lat), rng.integers(0, lat)
            w[j] = np.roll(fields[j // PER_FIELD], (-int(oy), -int(ox)), (0, 1))[:window, :window]
        return w

    ks = [0] + TIMES
    n_eval_fields = n_eval // PER_FIELD
    for i, k in enumerate(ks):
        if k == 0:
            train = (rng.random((n_fields * PER_FIELD, window, window)) < 0.5).astype(np.uint8)
            held = (rng.random((n_eval, window, window)) < 0.5).astype(np.uint8)
        else:
            n = n_fields + n_eval_fields
            fields = np.concatenate([coarsened(min(CHUNK, n - b), k, WINDOW_SEED + 1000 * i + b, lat)
                                     for b in range(0, n, CHUNK)])
            train, held = crops(fields[n_eval_fields:]), crops(fields[:n_eval_fields])[:n_eval]
        for name, w in (("ising", train), ("eval", held)):
            torch.save(torch.from_numpy(w[:, None]).float() * 2 - 1, out / f"{name}_t{i}.pt")
        print(f"t = {k}: {len(train)} train, {len(held)} held-out windows", flush=True)


def pairs(n, lat=LAT):
    """Each field of a chunk is snapshotted at its own t and t + delta during
    one shared evolution."""
    rng = np.random.default_rng(PAIR_SEED)

    def draw(m):
        t = np.exp(rng.uniform(np.log(3), np.log(1000), m)).astype(np.int64)
        d = np.exp(rng.uniform(np.log(JUMP_MIN), np.log(1000), m)).astype(np.int64)
        keep = t + d <= 1000
        return t[keep], d[keep]

    ts, ds = draw(n)
    while len(ts) < n:
        t, d = draw(n)
        ts, ds = np.concatenate([ts, t]), np.concatenate([ds, d])
    ts, ds = ts[:n], ds[:n]

    starts = np.empty((n, lat, lat), np.uint8)
    targets = np.empty((n, lat, lat), np.uint8)
    t0 = time.time()
    for b0 in range(0, n, CHUNK):
        m = min(CHUNK, n - b0)
        bt = torch.from_numpy(ts[b0:b0 + m]).to(DEV)
        be = torch.from_numpy(ts[b0:b0 + m] + ds[b0:b0 + m]).to(DEV)
        torch.manual_seed(PAIR_SEED + 13 * b0)
        spins = torch.where(torch.rand(m, lat, lat, device=DEV) < 0.5, 1.0, -1.0)
        gen = torch.Generator(device=DEV).manual_seed(PAIR_SEED + 13 * b0 + 1)
        for s in range(1, int(be.max()) + 1):
            sweep_(spins, 1.0, gen)
            for when, dst in ((bt, starts), (be, targets)):
                idx = (when == s).nonzero(as_tuple=True)[0]
                if len(idx):
                    dst[b0 + idx.cpu().numpy()] = (spins[idx] > 0).to(torch.uint8).cpu().numpy()
        print(f"{b0 + m}/{n} pairs ({time.time() - t0:.0f}s)", flush=True)
    return dict(starts=starts, targets=targets, t=ts, delta=ds, seed=PAIR_SEED)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    m = sub.add_parser("marginals")
    m.add_argument("--times", default=",".join(map(str, TIMES)))
    m.add_argument("--n-train", type=int, default=N_BASE)
    sub.add_parser("windows")
    p = sub.add_parser("pairs")
    p.add_argument("--n", type=int, default=15000)
    a = ap.parse_args()

    data = WORK / "data"
    data.mkdir(parents=True, exist_ok=True)
    if a.kind == "marginals":
        times = [int(t) for t in a.times.split(",")]
        name = ("ising_marginals" + ("" if times == TIMES else "_" + "_".join(map(str, times)))
                + ("" if a.n_train == N_BASE else f"_n{a.n_train}"))
        np.savez_compressed(data / f"{name}.npz", **marginals(times, a.n_train))
    elif a.kind == "windows":
        windows(data / "ising_windows")
    else:
        np.savez_compressed(data / f"ddpm_pairs_{a.n}.npz", **pairs(a.n))
    print("done", flush=True)


if __name__ == "__main__":
    main()
