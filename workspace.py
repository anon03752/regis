"""Paths for generated data, runs, evaluation results, and figures.

Outputs are stored in the repository by default. Set REGIS_WORK to use another
directory, such as a scratch filesystem or an existing artifact directory.
"""
import os
from pathlib import Path

WORK = Path(os.environ.get("REGIS_WORK") or Path(__file__).resolve().parent)


def resolve(p) -> Path:
    """Resolve a command-line path relative to WORK; preserve absolute paths."""
    p = Path(p)
    return p if p.is_absolute() else WORK / p
