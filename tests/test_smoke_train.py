"""Train Ising models briefly on 32x32 data, then generate two-field caches.

Covers REGIS, the pair-conditional GAN, DDPM, and MMtSBM warm-up plus one
IMF iteration. Run on CPU: python -m tests.test_smoke_train"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from domains.ising.data import marginals, pairs, windows

_REPO = Path(__file__).resolve().parents[1]      # the subprocesses' cwd


def run(tmp, *steps):
    """each step: a module and its arguments, run with the workspace in tmp"""
    env = dict(os.environ, REGIS_WORK=str(tmp))
    for step in steps:
        r = subprocess.run([sys.executable, "-m", *step], cwd=_REPO, env=env,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"{step[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"


def cache(tmp, label, ns=2, lat=32):
    return np.load(tmp / "data" / "rollouts" / f"{label}__lat{lat}_ns{ns}.npz")


def test_smoke_train():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "marginals.npz", **marginals([3, 10], lat=32))
        run(tmp, ["domains.ising.train", "--smoke-test", "--device", "cpu", "--no-compile-step",
                  "--fire-mode", "checker2", "--data", str(tmp / "marginals.npz"), "--run-name", "smoke"])

        rd = tmp / "runs" / "smoke"
        status = json.loads((rd / "status.json").read_text())
        assert status["status"] == "completed", status
        assert (rd / "config.json").exists() and (rd / "events.jsonl").exists()
        ckpts = sorted((rd / "ckpts").glob("ckpt_*.pt"))
        assert ckpts, "no checkpoint written"

        run(tmp, ["domains.ising.instruments.rule_cache", "--run", "smoke", "--ns", "2", "--lat", "32"],
            ["domains.ising.instruments.rule_cache", "--engine", "1", "--ns", "2", "--lat", "32"])
        run(tmp, ["domains.ising.train", "--smoke-test", "--device", "cpu", "--no-compile-step", "--g-arch", "control",
                  "--pair-cond", "--data", str(tmp / "marginals.npz"), "--run-name", "smoke_pairgan"],
            ["domains.ising.instruments.rule_cache", "--run", "smoke_pairgan", "--ns", "2", "--lat", "32"])
        for label in ("smoke", "smoke_pairgan", "truth_seed1"):
            z = cache(tmp, label)
            assert z["ks"][-1] == 4000 and z["bits"].shape == (len(z["ks"]), 2 * 32 * 32 // 8)
        print("REGIS, pair GAN and engine caches written")


def test_smoke_ddpm():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "pairs.npz", **pairs(8, lat=32))
        run(tmp, ["models.ddpm", "--run-name", "smoke", "--data", str(tmp / "pairs.npz"),
                  "--total-steps", "6", "--batch", "2", "--crop", "16", "--halo", "8", "--base", "8"],
            ["domains.ising.instruments.ddpm_cache", "--run", "smoke", "--ns", "2", "--lat", "32",
             "--grid", "13,24", "--sample-steps", "3", "--raw"])
        assert sorted((tmp / "runs" / "smoke" / "ckpts").glob("ckpt_*.pt")), "no checkpoint written"
        z = cache(tmp, "smoke")
        assert list(z["ks"]) == [13, 24] and z["bits"].shape == (2, 2 * 32 * 32 // 8)
        assert z["raw"].shape == (2, 2, 32, 32)
        print("ddpm smoke completed: trained, sampled, cached")


def test_smoke_mmtsbm():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        windows(tmp / "windows", window=16, n_fields=4, n_eval=4, lat=32)
        run(tmp, ["baselines.mmtsbm.train", "--smoke", "--run-name", "smoke", "--data", str(tmp / "windows"),
                  "--symmetrise", "z2rot180", "--circular", "--halo", "4"],
            ["domains.ising.instruments.mmtsbm_cache", "--run", "smoke", "--ns", "2", "--lat", "32",
             "--steps-per-bridge", "2", "--raw"])
        z = cache(tmp, "smoke")
        assert {3, 10, 17, 300, 1000} <= set(z["ks"].tolist())
        assert z["bits"].shape == (len(z["ks"]), 2 * 32 * 32 // 8) and z["raw"].shape == (len(z["ks"]), 2, 32, 32)
        print("mmtsbm smoke completed: warm-up, one IMF iteration, bridge chain cached")


if __name__ == "__main__":
    test_smoke_train()
    test_smoke_ddpm()
    test_smoke_mmtsbm()
    print("SMOKE_TRAIN_PASSED")
