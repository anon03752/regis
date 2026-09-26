"""Measure domain length after flipping isolated spins.

A singleton has all four neighbours opposite to it. Domain length is
L = 1 / rho_of(despeckle(field)); iso_frac measures singleton density.
Inputs are {0, 1} or {-1, +1} arrays of shape (..., H, W), with periodic
boundaries."""
from __future__ import annotations

import numpy as np


def _pm(b):
    """any {0,1} or {-1,+1} array -> int8 {-1,+1}"""
    return np.where(np.asarray(b) > 0, np.int8(1), np.int8(-1))


def _nbsum(s):
    return sum(np.roll(s, sh, ax) for sh, ax in ((1, -1), (-1, -1), (1, -2), (-1, -2)))


def iso_frac(b):
    """fraction of sites that are pure singletons"""
    s = _pm(b)
    return (s * _nbsum(s) == -4).mean(axis=(-1, -2))


def despeckle(b):
    """Flip isolated spins; return an int8 {-1, +1} array."""
    s = _pm(b)
    return np.where(s * _nbsum(s) == -4, -s, s)


def rho_of(b):
    """interface density: fraction of unlike nearest-neighbour bonds"""
    s = _pm(b)
    return 0.5 * ((s * np.roll(s, 1, -1) < 0).mean(axis=(-1, -2))
                  + (s * np.roll(s, 1, -2) < 0).mean(axis=(-1, -2)))
