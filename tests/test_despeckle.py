"""Check singleton removal, periodic domain walls, and spin encodings.

Run on CPU: python -m tests.test_despeckle"""
import numpy as np

from domains.ising.instruments.despeckle import despeckle, iso_frac, rho_of


LAT = 64


def _singleton_field(k=5, seed=0):
    """all-up field with k isolated flipped spins, pairwise non-adjacent."""
    rng = np.random.default_rng(seed)
    f = np.ones((LAT, LAT), np.int8)
    placed = []
    while len(placed) < k:
        y, x = rng.integers(0, LAT, 2)
        if all(max(abs(y - py), abs(x - px)) > 2 for py, px in placed):
            f[y, x] = -1
            placed.append((y, x))
    return f[None]      # (1, H, W)


def test_singletons_removed():
    f = _singleton_field()
    assert float(iso_frac(f)[0]) == 5 / LAT**2, "iso_frac must count exactly the singletons"
    d = despeckle(f)
    assert (d == 1).all(), "despeckle must remove every singleton"
    assert float(iso_frac(d)[0]) == 0.0 and float(rho_of(d)[0]) == 0.0


def test_stripe_untouched():
    f = np.ones((1, LAT, LAT), np.int8)
    f[:, LAT // 2:] = -1                      # two periodic horizontal walls
    assert float(iso_frac(f)[0]) == 0.0
    d = despeckle(f)
    assert (d == f).all(), "despeckle must not touch domain walls"
    rho = float(rho_of(f)[0])
    assert abs(1 / rho - LAT) < 1e-9, f"stripe L = 1/rho should be {LAT}, got {1/rho}"


def test_mixed():
    f = np.ones((1, LAT, LAT), np.int8)
    f[:, LAT // 2:] = -1
    f[0, LAT // 2 - 1, 20] = -1                              # a bump on the wall: three opposite neighbours, kept
    g = f.copy()
    g[0, 5, 5] = -1
    g[0, 10, 40] = -1        # singletons in the up domain
    d = despeckle(g)
    assert (d == f).all(), "despeckle must remove the singletons and only them"


def test_encodings_agree():
    f01 = (np.random.default_rng(1).random((2, LAT, LAT)) < 0.5).astype(np.uint8)
    fpm = np.where(f01 > 0, 1, -1).astype(np.int8)
    assert (despeckle(f01) == despeckle(fpm)).all()
    assert np.allclose(iso_frac(f01), iso_frac(fpm))


if __name__ == "__main__":
    test_singletons_removed()
    test_stripe_untouched()
    test_mixed()
    test_encodings_agree()
    print("DESPECKLE_PASSED")
