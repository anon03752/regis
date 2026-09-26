"""Public run names for the hearts domain: the single source of truth for every
stem the instruments, tables and figures reference, and the names to pass as
--run-name when retraining.

Eighteen models of record. REGIS has seven seeds, the two ablation arms three
each, one further REGIS run holds heart T1_S1 out of training entirely, and four
single-seed REGIS runs each drop one auxiliary loss. All were trained for 100k
iterations; `--seed N` is the value shown.
"""
from __future__ import annotations
from pathlib import Path

SEEDS = tuple(range(7))
ABLATION_SEEDS = (0, 1, 2)
TRAIN_STEPS = 100_000
HOLDOUT = "T1_S1"

REGIS = [f"hearts_regis_seed{s}" for s in SEEDS]
TIME = [f"hearts_time_seed{s}" for s in ABLATION_SEEDS]
CONTROL = [f"hearts_control_nonlocal_seed{s}" for s in ABLATION_SEEDS]
LOO = [f"hearts_regis_loo_{HOLDOUT}"]
# one REGIS seed-0 run per auxiliary loss removed, in the paper's row order
# (domains/hearts/train.py --drop-loss)
AUX = {loss: f"hearts_regis_no_{loss}" for loss in ("silhouette", "capacity", "heal", "zero")}
ALL = REGIS + TIME + CONTROL + LOO + list(AUX.values())

# the transport baselines (baselines/): MMSFM at seeds 0 and 1, and one MMtSBM run,
# trained with --seed 13 as run 0 is in the MNIST chapter
MMSFM = [f"hearts_mmsfm_seed{s}" for s in (0, 1)]
MMTSBM = ["hearts_mmtsbm_seed0"]

# rows of the tables and figures, in the paper's order
METHODS = [("REGIS (ours)", REGIS), ("REGIS + time", TIME),
           ("REGIS non-local", CONTROL), ("REGIS held-out heart", LOO)]

# which generator each family is built from (domains/hearts/train.py --g-arch)
ARCH = {**{s: "regis" for s in REGIS + LOO + list(AUX.values())},
        **{s: "time" for s in TIME}, **{s: "control" for s in CONTROL}}


def family(stem: str) -> str:
    return next(name for name, stems in METHODS if stem in stems)


def checkpoint(work: Path, stem: str, step: int = TRAIN_STEPS) -> Path:
    """The checkpoint of a run: `<run>/ckpts/ckpt_NNNNNN.pt`."""
    return Path(work) / "runs" / stem / "ckpts" / f"ckpt_{step:06d}.pt"
