# MMSFM baseline

Uses the authors' implementation of *Multi-Marginal Stochastic Flow Matching for High-Dimensional Snapshot Data at Irregular Time Points* (Lee, Moradijamei, Shakeri; ICML 2025). `fetch_upstream.sh` downloads and patches these pinned versions:

- `Shakeri-Lab/MMSFM` @ `f8c702b15fba75677d3a89fed2190c6ca9f3e106`
- `atong01/conditional-flow-matching` @ `af8fec6f6dc3a0dc7f8fb25d2ee0ca819fa5412f`

## Training

From the repository root, after installing `requirements.txt`:

```bash
pip install torchsde==0.2.6 POT joblib tqdm wandb
bash baselines/mmsfm/fetch_upstream.sh
python -m domains.ising.data windows
python -m baselines.mmsfm.train --run-name mmsfm_seed0 --seed 0
python -m domains.mnist.data digitloop
python -m baselines.mmsfm.train --domain mnist --run-name mnist_mmsfm_seed0 --seed 0
```

Settings are in `SETTINGS` in `train.py` and saved to `runs/<run-name>/config.json`. Ising uses seeds 0–2 and takes about 18 h per run on one H100; MNIST uses seeds 0–6 and takes about 5 h. The driver checks symmetry and compiled gradients before training and saves a checkpoint every 2000 updates. Evaluation commands are in [Reproducing the paper](../../docs/reproduce.md).

## Upstream changes

[mmsfm_ising.patch](patches/mmsfm_ising.patch) makes three changes:

- Fixes the row index when chaining OT couplings: each sample must select from its row in the resampled batch's coupling matrix.
- Adds batched spline evaluation (`MMSFM_VECSPLINE=1`).
- Adds loss micro-batching (`MMSFM_CHUNKS`) and a central loss window (`MMSFM_HALO`).

The OT correction is always active; the driver sets the optional environment variables. `net.py` adds Ising spin-flip antisymmetry and circular padding, and a periodic time embedding for MNIST.

## Reproduction notes

The original environment used torch 2.7.1, numpy 1.26.4, scipy 1.15.3, POT 0.9.5, joblib 1.5.1, and torchsde 0.2.6. Training uses TF32 for Ising and fp32 for MNIST. Install dependencies as above; the upstream requirements pin a different torch version.

The trainer retains otherwise unused network initialisations to preserve the original random-number sequence. Their purpose is documented beside the calls in `train.py`. Compilation and kernel autotuning can change floating-point results between runs. The Ising cache writer enables deterministic algorithms; the original caches used autotuning. The MNIST sampler now runs its SRK steps eagerly; the original captures used CUDA graphs.
