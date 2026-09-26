"""End-to-end smoke of the hearts chapter: build a cohort, train all three
arms on it, then roll a knockout through the evaluation driver.

Runs on a fabricated cohort rather than the atlas, so it needs no download: a
handful of synthetic sections in the real spot-table format, wounded with the
repo's own annotations. That makes it a test of the CODE PATH -- the producer's
one-hot, canvas padding, two-layer stamp, normalised densification and
sum-pool; the trainer's pool, wound coin, stage transitions, six auxiliary
losses, EMA and checkpointing; and the rollout driver's arc, settle and both
knockout modes. It is not a test of the cohort's values, which is what
`data_cohort.check()` does against the real download.

CPU, about a minute.

Run: python -m tests.test_smoke_hearts
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from domains.hearts import cohort as C
from domains.hearts.wounds import ANNOTATIONS, CANVAS

_REPO = Path(__file__).resolve().parents[1]      # the subprocesses' cwd

# one section per stage, plus a second uninjured one so the seed bank is not a
# single heart. Ids must match annotation filenames for the wound path to fire.
SECTIONS = ["T1_C1_S1", "T1_C1_S2", "T2_C1_S1", "T3_C1_S1", "T4_C1_S1",
            "T5_C1_S1", "T6_C1_S1", "T7_C1_S1", "T8_C1_S1"]
STAGE_OF = dict(zip(SECTIONS, ["uninjured", "uninjured"] + C.TP_ORDER[1:]))


def _fake_spot_table(path, rng):
    """A disc of spots per section, in the atlas's own column layout."""
    rows = ["\t".join(["orig.ident", "nCount_Spatial", "nFeature_Spatial", "x", "y",
                       "grid_x", "grid_y", "isolate", "cid", "time_points",
                       "annotation"])]
    for sid in SECTIONS:
        yy, xx = np.mgrid[0:40, 0:40]
        inside = ((yy - 20) ** 2 + (xx - 20) ** 2) < 15 ** 2
        for r, c in zip(*np.nonzero(inside)):
            # the bin index lives in the barcode, offset so the producer's
            # bounding-box crop has something to do
            bc = f"{sid}:cell{r + 200}_{c + 260}"
            ann = C.CELL_TYPES[int(rng.integers(len(C.CELL_TYPES)))]
            rows.append("\t".join([bc, sid, "1000", "300", "0.0", "0.0", "0.0",
                                   "0.0", sid[:2], sid[:5], STAGE_OF[sid], ann]))
    Path(path).write_text("\n".join(rows) + "\n")
    return len(rows) - 1


def test_smoke_hearts():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        spots = work / "spots.tsv"
        n = _fake_spot_table(spots, rng)

        # ---- the producer ----
        from domains.hearts import data_cohort as D
        sections = D.build(spots, ANNOTATIONS / "wounds")
        assert len(sections) == len(SECTIONS), f"{len(sections)} sections built"
        for s in sections:
            comp, m = s["composition"], s["mask"][0] > 0
            assert comp.shape == (22, 48, 48), comp.shape
            assert float(comp[:, ~m].abs().max()) < 1e-5, "leaked outside the mask"
            assert float(comp[:21][:, m].sum(0).max()) <= 4.0 + 1e-3, "capacity > 4"
        # the wound path must actually have fired: the annotated stages carry a
        # reference layer, the uninjured ones cannot
        ref = {s["timepoint"]: float(s["composition"][21].sum()) for s in sections}
        assert ref["uninjured"] == 0.0, "uninjured section carries a wound site"
        assert any(v > 0 for k, v in ref.items() if k != "uninjured"), \
            "no wound annotation was applied to any injured section"
        npz = work / "data/hearts/hearts_48.npz"      # where the trainer looks under WORK
        npz.parent.mkdir(parents=True, exist_ok=True)
        _write(npz, sections)
        print(f"producer: {n} spots -> {len(sections)} sections, wound site present")

        # ---- the trainer, all three arms ----
        env = dict(os.environ, REGIS_WORK=str(work))
        for arch in ("regis", "control", "time"):
            r = subprocess.run(
                [sys.executable, "-m", "domains.hearts.train", "--g-arch", arch,
                 "--run-name", f"smoke_{arch}", "--smoke-test"],
                cwd=_REPO, env=env, capture_output=True, text=True)
            assert r.returncode == 0, f"{arch} trainer failed:\n{r.stdout}\n{r.stderr}"
            run = work / "runs" / f"smoke_{arch}"
            assert (run / "config.json").exists() and (run / "events.jsonl").exists()
            ck = sorted((run / "ckpts").glob("ckpt_*.pt"))
            assert ck, f"{arch}: no checkpoint written"
            print(f"trainer {arch}: completed, {len(ck)} checkpoint(s)")

        # ---- the evaluation driver, on what the trainer just wrote ----
        from domains.hearts import rollout as R
        rule, args = R.load_rule(sorted((work / "runs/smoke_regis/ckpts").glob("*.pt"))[-1], "cpu")
        state, mask = R.seed_batch(npz, 2, np.random.default_rng(1), "cpu")
        assert float(R.damage_over_heart(state, mask).mean()) > 0, "the seed was not wounded"
        for mode in ("no-increase", "zero"):
            settled, legs, every = R.roll(rule, state.clone(), knockout=3, ko_mode=mode,
                                          record=True, steps_per_leg=2, settle=2, seed=0)
            assert len(legs) == R.N_LEGS and len(every) == 1 + R.N_LEGS * 2
            assert settled.shape == state.shape
        print(f"rollout: {R.N_LEGS} legs under both knockout modes")


def _write(path, sections):
    np.savez_compressed(
        path,
        composition=np.stack([s["composition"].numpy() for s in sections]) / 2.0 - 1.0,
        mask=np.stack([s["mask"][0].numpy() for s in sections]).astype(np.uint8),
        channels=np.array(C.CELL_TYPES + [C.DAMAGE_CHANNEL, C.REF_CHANNEL]),
        section_id=np.array([s["section_id"] for s in sections]),
        timepoint=np.array([s["timepoint"] for s in sections]),
        hours=np.array([s["hours"] for s in sections], dtype=np.float32))


if __name__ == "__main__":
    test_smoke_hearts()
    print("SMOKE_HEARTS_PASSED")
