"""Ising run names shared by training commands and evaluation scripts.

Use these names for --run-name and --label, as shown in docs/reproduce.md.

Numbers in the paper come from three training seeds per model (MMtSBM was
trained with seeds 13-15; everything else with 0-2) and three independent
simulator references, each rolled from quench seed 1234 at evaluation.
"""
from workspace import WORK

SEEDS = 0, 1, 2
MMTSBM_SEEDS = 13, 14, 15
TRUTH_SEEDS = 4321, 8765, 2468

TRUTH = [f"truth_seed{s}" for s in TRUTH_SEEDS]
REGIS = [f"regis_k3_seed{s}" for s in SEEDS]
REGIS_K5 = [f"regis_k5_seed{s}" for s in SEEDS]
REGIS_K7 = [f"regis_k7_seed{s}" for s in SEEDS]
CONTROL = [f"control_nonlocal_seed{s}" for s in SEEDS]
PAIRGAN = [f"pairgan_seed{s}" for s in SEEDS]
DDPM = [f"ddpm_pairs15000_seed{s}" for s in SEEDS]
MMSFM = [f"mmsfm_seed{s}" for s in SEEDS]
MMTSBM = [f"mmtsbm_seed{s}" for s in MMTSBM_SEEDS]

MARGINALS = 3, 10, 17, 300, 1000   # the five training marginals (TIMES in domains/ising/data.py)
HORIZON = MARGINALS[-1]            # the last training marginal

# data ladders (the data-efficiency figure): REGIS at F fields per marginal
# (5F images), fixed (0,0) windows; the DDPM at n training pairs
LADDER_F = 10, 30, 100, 300, 1000, 3000
LADDER_N = 50, 150, 500, 1500, 5000, 15000
LADDER_REGIS = {5 * F: [f"regis_k3_fixedcrops_f{F}_seed{s}" for s in SEEDS] for F in LADDER_F}
LADDER_DDPM = {n: [f"ddpm_pairs{n}_seed{s}" for s in SEEDS] for n in LADDER_N}

# the marginal-placement sweep (REGIS 3x3): grid i trained at seed 100 + i,
# grids 11, 13 and 14 also at 200 + i; stems keyed by (grid, seed)
GRIDS = [(3, 1000), (3, 55, 1000), (3, 17, 175, 1000), (3, 10, 55, 300, 1000), (3, 10, 17, 55, 175, 300, 1000),
         (3, 10, 17, 30, 55, 100, 175, 300, 550, 1000), (3, 10, 17, 30, 1000), (3, 30, 55, 100, 1000),
         (3, 100, 175, 300, 1000), (3, 175, 300, 550, 1000), (10, 30, 100, 300, 1000), (30, 55, 175, 550, 1000),
         (100, 175, 300, 550, 1000), (3, 5, 7, 10, 1000), (3, 550, 700, 850, 1000)]
PLACEMENT = {(g, s): f"marginals_{'_'.join(map(str, g))}_seed{s}"
             for i, g in enumerate(GRIDS) for s in (100 + i, 200 + i) if s < 200 or i in (11, 13, 14)}


def cache_path(stem: str):
    """The stem's 512-field rollout cache, where instruments/rollouts.py writes it."""
    return WORK / "data/rollouts" / f"{stem}__lat512_ns512.npz"
