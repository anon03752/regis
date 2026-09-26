"""Brownian-bridge training targets and Euler-Maruyama sampling for MMtSBM.

Reimplemented from the paper (Gravier, Boyer & Genovesio, arXiv:2510.01894).
The authors' reference implementation carries no licence, so no code was
copied from it.

K marginals sit at times 0, 1, ..., K-1 (the index of a marginal, not its
time in sweeps), and adjacent marginals define K-1 bridges, each integrated
in its own normalised time s in [0, 1]. Two drift networks are trained:
`forward` points at later times, `backward` at earlier ones. Both functions
take the drift as a callable `b(x, t) -> tensor`, with `t` of shape (B, 1) in
absolute time.
"""
from __future__ import annotations

import torch


def bridge_batch(z0, z1, sigma: float, eps: float = 1e-3):
    """One training sample per pair: a point on the Brownian bridge, and both targets.

    Given endpoints (z0 at the earlier anchor, z1 at the later one) the bridge
    at normalised time s is Gaussian with mean (1-s)*z0 + s*z1 and standard
    deviation sigma*sqrt(s*(1-s)) -- pinned at both ends, widest in the middle.
    Regressing on the two targets below is bridge matching: the forward target
    is the displacement z1-z0 corrected by the noise actually drawn, so its
    conditional mean is the drift of the bridge SDE, and likewise backward.

    `eps` keeps s off 0 and 1, where the corrections have a 1/sqrt(s) pole.
    Returns (z_s, s, target_forward, target_backward); s has z0's rank so it
    broadcasts, and the caller maps it to absolute time.
    """
    s = torch.rand(z0.shape[0], *([1] * (z0.ndim - 1)), device=z0.device)
    s = eps + (1.0 - 2.0 * eps) * s
    w = torch.randn_like(z0)
    z_s = (1 - s) * z0 + s * z1 + sigma * (s * (1 - s)).sqrt() * w
    step = z1 - z0
    return (z_s, s,
            step - sigma * (s / (1 - s)).sqrt() * w,
            -step - sigma * ((1 - s) / s).sqrt() * w)


@torch.no_grad()
def simulate(drift, z, t_a, t_b, sigma: float, direction: str, n: int, on_step=None):
    """Integrate from t_a to t_b with n Euler-Maruyama steps.

    Both directions use positive ds; the backward drift predicts the negated
    displacement. on_step(i, z) receives the state after step i = 1..n so callers
    can retain selected readouts without storing the whole trajectory."""
    ds = 1.0 / n
    for i in range(n):
        s = i * ds if direction == "forward" else 1.0 - i * ds
        t = torch.full((z.shape[0], 1), float(t_a + s * (t_b - t_a)),
                       device=z.device)
        z = (z + drift(z, t) * ds
             + sigma * ds ** 0.5 * torch.randn_like(z))
        if on_step is not None:
            on_step(i + 1, z)
    return z
