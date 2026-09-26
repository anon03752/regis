"""Measure per-site flip probabilities P(flip | s, h) on real Ising states.

Classify each site just before its checkerboard half-step and count changes
in the thresholded readout. The engine updates sublattice 0 first; the learned
rules update sublattice 1 first. Both halves are pooled with their spin-flipped
counterparts. The analytic reference is sigmoid(-2*s*h) at T = 1.

Use the first 64 fields of each of three truth caches at every time in KS.
Rules run in float32 here; rule_cache uses bf16 autocast on GPU."""
from __future__ import annotations

import json

import numpy as np
import torch

from domains.ising.glauber import checkerboard, neighbour_sum, sweep_
from domains.ising.instruments import rollouts
from domains.ising.instruments.rule_cache import load_rule
from domains.ising.results import runs
from workspace import WORK

KS = [3, 5, 10, 17, 30, 55, 100, 175, 300, 550, 1000, 2000, 4000]
NF = 64       # fields per truth cache
CHUNK = 16    # fields per forward pass (the rules' noise draws depend on it)
CLASSES = [(1, 4), (1, 2), (1, 0), (1, -2), (1, -4)]
MEMBERS = {"regis_k3": runs.REGIS, "regis_k7": runs.REGIS_K7,
           "control_nonlocal": runs.CONTROL, "pairgan": runs.PAIRGAN}


def count(before, after, mask, acc):
    """add to acc["s,h"] = (sites, flips) the sites in `mask` (the sublattice that updated), classified by the
    state before the half"""
    s = torch.where(before > 0.5, 1.0, -1.0)
    h = neighbour_sum(s)
    flipped = (after > 0.5) != (before > 0.5)
    for sv in (1, -1):
        for hv in (-4, -2, 0, 2, 4):
            m = (s == sv) & (h == hv) & mask
            n, f = acc.get(f"{sv},{hv}", (0, 0))
            acc[f"{sv},{hv}"] = (n + int(m.sum()), f + int(flipped[m].sum()))


def pooled(accs, s, h):
    """(sites, flips) of class (s, h) plus its mirror (-s, -h), summed over
    accs, per-time dicts of the JSON's per_t"""
    n = f = 0
    for acc in accs:
        for key in (f"{s},{h}", f"{-s},{-h}"):
            dn, df = acc.get(key, (0, 0))
            n, f = n + dn, f + df
    return n, f


def probe_engine(states, sub):
    gen = torch.Generator(device=sub.device).manual_seed(4321)
    per_t = {}
    for k, x in states.items():
        acc = per_t[str(k)] = {}
        for c0 in range(0, len(x), CHUNK):
            s = torch.where(torch.from_numpy(x[c0:c0 + CHUNK]).to(sub.device) > 0.5, 1.0, -1.0)
            for c in (0, 1):
                before = (s > 0).float()
                sweep_(s, 1.0, gen, sublattices=(c,))
                count(before, (s > 0).float(), sub == c, acc)
    return per_t


def probe_rule(G, hp, states, sub):
    torch.manual_seed(4321)
    odd = sub.float()
    per_t = {}
    for k, x in states.items():
        acc = per_t[str(k)] = {}
        for c0 in range(0, len(x), CHUNK):
            xc = torch.from_numpy(x[c0:c0 + CHUNK]).to(sub.device)
            n, lat = len(xc), xc.shape[-1]
            st = xc.float()[:, None]
            before = G.readout(st)[:, 0].float()
            for c, mask in ((1, odd), (0, 1.0 - odd)):
                noise = torch.randn(n, hp["noise_channels"], lat, lat, device=xc.device)
                st = G.masked_update(st, mask, noise=noise)
                after = G.readout(st)[:, 0].float()
                count(before, after, sub == c, acc)
                before = after
    return per_t


def main():
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    states = {k: [] for k in KS}
    for stem in runs.TRUTH:
        bits, ks, m = rollouts.load(runs.cache_path(stem))
        for k in KS:
            states[k].append(rollouts.unpack(bits[ks.index(k)], m["lat"], NF))
    states = {k: np.concatenate(v) for k, v in states.items()}      # uint8 0/1
    sub = checkerboard(states[KS[0]].shape[-1], dev)

    results = {"engine": {"per_t": probe_engine(states, sub)}}
    print("engine done", flush=True)
    for fam, stems in MEMBERS.items():
        for i, stem in enumerate(stems):
            G, hp, _ = load_rule(stem, dev)
            results[f"{fam} s{i}"] = {"per_t": probe_rule(G, hp, states, sub)}
            print(f"{fam} s{i} done", flush=True)

    def rate(name, s, h):
        n, f = pooled(results[name]["per_t"].values(), s, h)
        return f / n if n else float("nan")

    formula = {f"{s},{h}": float(1 / (1 + np.exp(2 * h))) for s, h in CLASSES}
    names = list(results)
    lines = ["| class (s=+1, h) | aligned nbrs | sites | formula | " + " | ".join(names) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(names)]
    for s, h in CLASSES:
        lines.append(f"| h = {h:+d} | {(4 + h) // 2} of 4 | {pooled(results['engine']['per_t'].values(), s, h)[0]:,} | "
                     f"{formula[f'{s},{h}']:.2e} | " + " | ".join(f"{rate(nm, s, h):.2e}" for nm in names) + " |")
    lines += ["", f"P(flip | s, h), each checkerboard half measured on its own; both halves pooled; times and "
                  f"the +-s mirror pooled here, kept apart in the JSON. {len(states[KS[0]])} real 512^2 states "
                  f"per time ({', '.join(runs.TRUTH)}) at t = {KS}. The engine column checks the analytic rates."]
    md = "\n".join(lines)
    print(md)
    (WORK / "eval").mkdir(parents=True, exist_ok=True)
    (WORK / "eval" / "flip_table.md").write_text(md)
    (WORK / "eval" / "flip_table.json").write_text(json.dumps(dict(ks=KS, formula=formula, results=results), indent=1))


if __name__ == "__main__":
    main()
