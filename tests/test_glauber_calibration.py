"""Compare measured Glauber flip rates with p(s, h) = 1 / (1 + exp(2*s*h)).

Classify sublattice-0 sites by their pre-sweep s*h, since their neighbours
are fixed during that half-sweep. Initial and coarsened fields supply
samples for all five classes. Allow six binomial standard deviations,
with a floor of ten flips.

Run on CPU: python -m tests.test_glauber_calibration"""
import numpy as np
import torch

from domains.ising.glauber import checkerboard, neighbour_sum, sweep_


def measure(n_fields=64, lat=64, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    sub = checkerboard(lat, "cpu")
    counts = {h: [0, 0] for h in (-4, -2, 0, 2, 4)}    # s*h -> [sites, flips]: the rate depends on s*h only
    for t_target in (0, 10):
        s = torch.where(torch.rand(n_fields, lat, lat) < 0.5, 1.0, -1.0)
        for _ in range(t_target):
            sweep_(s, 1.0, g)
        before = s.clone()
        sweep_(s, 1.0, g)
        sh = before * neighbour_sum(before)
        for h, c in counts.items():
            m = (sh == h) & (sub == 0)
            c[0] += int(m.sum())
            c[1] += int((s != before)[m].sum())
    return counts


def test_rates():
    counts = measure()
    for hv in (-4, -2, 0, 2, 4):
        p = 1.0 / (1.0 + np.exp(2.0 * hv))          # s = +1 convention
        n, f = counts[hv]
        assert n > 1000, f"class h={hv:+d}: only {n} sites measured"
        expect, sigma = n * p, np.sqrt(n * p * (1 - p))
        tol = max(6.0 * sigma, 10.0)
        assert abs(f - expect) <= tol, (
            f"class (s=+1, h={hv:+d}): {f} flips of {n} sites "
            f"(p_emp={f/n:.3e}) vs analytic {p:.3e} (expect {expect:.1f} +- {tol:.1f})")
        print(f"h={hv:+d}: sites {n:>9,}  p_emp {f/n:.4e}  analytic {p:.4e}  OK")


if __name__ == "__main__":
    test_rates()
    print("GLAUBER_CALIBRATION_PASSED")
