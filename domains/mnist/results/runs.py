"""MNIST run names shared by training commands and evaluation scripts.

Every method has seven runs, indexed 0..6 in the stems, tables and figures;
run i was trained with --seed 42+i (REGIS and the non-local control), 13+i
(MMtSBM) or i (MMSFM).
"""
from __future__ import annotations
from pathlib import Path

from workspace import WORK

SEEDS = tuple(range(7))

REGIS = [f"mnist_regis_seed{s}" for s in SEEDS]
CONTROL = [f"mnist_control_nonlocal_seed{s}" for s in SEEDS]
MMSFM = [f"mnist_mmsfm_seed{s}" for s in SEEDS]
MMTSBM = [f"mnist_mmtsbm_seed{s}" for s in SEEDS]

# rows of the tables and figures, in the paper's order: (display name, stems)
METHODS = [("REGIS", REGIS), ("Non-local control", CONTROL), ("MMSFM", MMSFM), ("MMtSBM", MMTSBM)]


def shown_seed(stem: str) -> int:
    return int(stem[stem.rindex("_seed") + 5:])


def checkpoint(stem: str) -> Path:
    """The last training checkpoint of a run trained here (REGIS, control)."""
    ckpts = sorted((WORK / "runs" / stem / "ckpts").glob("ckpt_*.pt"))
    assert ckpts, f"no checkpoint in {WORK / 'runs' / stem / 'ckpts'}"
    return ckpts[-1]


def capture(stem: str) -> Path:
    """Path to a frame capture; its array format is defined in capture_frames.py."""
    return WORK / "data" / "mnist" / "captures" / f"{stem}.npz"
