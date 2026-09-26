"""The ground truth: Glauber heat-bath dynamics of the 2D Ising model on a
torus, batched over fields. One sweep updates the two checkerboard sublattices
in turn; a site flips with probability 1 / (1 + exp(2 s h / T)), h the sum of
its four neighbours. tests/test_glauber_calibration.py checks the measured
flip rates against these analytic values.
"""
from __future__ import annotations

import torch


def checkerboard(L: int, device) -> torch.Tensor:
    """each site's sublattice, (y + x) % 2: an (L, L) map of 0 and 1"""
    r = torch.arange(L, device=device)
    return (r[:, None] + r) % 2


def neighbour_sum(s: torch.Tensor) -> torch.Tensor:
    """h: the sum of each site's four neighbours on the torus (the last two dims)"""
    return s.roll(1, -2) + s.roll(-1, -2) + s.roll(1, -1) + s.roll(-1, -1)


@torch.no_grad()
def sweep_(s: torch.Tensor, T: float, gen: torch.Generator, sublattices=(0, 1)) -> None:
    """One in-place sweep on a batch s of shape (B, L, L), values +-1.
    sublattices=(c,) updates sublattice c only: half a sweep."""
    sub = checkerboard(s.shape[-1], s.device)
    for c in sublattices:
        p_flip = torch.sigmoid(-2.0 * s * neighbour_sum(s) / T)    # == 1/(1+exp(2 s h /T))
        u = torch.rand(s.shape, generator=gen, device=s.device, dtype=s.dtype)
        flip = (u < p_flip) & (sub == c)                          # sub broadcasts over B
        s.mul_(torch.where(flip, -1.0, 1.0))
