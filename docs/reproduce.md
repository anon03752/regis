# Reproducing the paper

Run commands from the repository root after installing `requirements.txt`. Outputs go under `data/`, `runs/`, `eval/`, and `figures/`; set `REGIS_WORK` to change their location. Each run saves its settings in `<run>/config.json`. MMSFM requires the additional packages in its [README](../baselines/mmsfm/README.md).

Use a GPU for data generation, training, and rollouts. Evaluation summaries and figures run on CPU, except `fig_style`. The tests below run on CPU. Reported timings are for one H100.

The current requirements pin torch 2.13 and torchvision 0.28. The original Ising rollouts and MNIST baseline captures used torch 2.7.1 and torchvision 0.22.1. Random draws and floating-point results can differ with the device, library versions, and kernel selection. The MNIST baseline captures use autotuning, so repeated runs can also differ numerically.

## Tests

```bash
python -m tests.test_glauber_calibration
python -m tests.test_despeckle
python -m tests.test_smoke_train
python -m tests.test_smoke_hearts
```

The smoke tests train for a few steps on small synthetic datasets and check the resulting checkpoints and rollouts. They take a few minutes on CPU.

## Index

Scripts and output paths for the paper's figures and tables. Commands are listed by experiment below.

| Paper item | Made by | Output |
|---|---|---|
| `fig:graph-abstract`, the method scheme | drawn by hand, not generated | |
| **Cycling MNIST** | | |
| `fig:fig2_mnist`, digit transitions | a composite, not generated | |
| `tab:horizon`, image quality over transitions | `fd_ci`, then `fd_tables` (MNIST §4) | `figures/mnist/horizon.tex` |
| `fig:app-mnist-samples`, samples per seed | `fig_samples` (MNIST §4) | `figures/mnist/fig_perseed.pdf` |
| `tab:horizon-seeds`, per-seed results | `fd_ci`, then `fd_tables` (MNIST §4) | `figures/mnist/table_supp.tex` |
| `fig:app-mnist-style`, latent style | `fig_style` (MNIST §4; GPU) | `figures/mnist/fig_style.pdf` |
| `fig:raster-seeds`, cycling and validity | not generated | |
| **Ising** | | |
| `fig:fig3_ising`, rollouts from one shared start | `fig_filmstrip512 short` (Ising §4); the paper figure is a composite of it | `figures/fig3_ising_filmstrip_strip.pdf` |
| `fig:ising_growth`, domain growth | `grade_campaign512` (Ising §3), then `fig_growth512 growth` (Ising §4) | `figures/fig3_ising_growth.pdf` |
| `fig:ising_ladder`, data efficiency | the data ladder (Ising §2), `grade_campaign512`, then `fig_ladder512` | `figures/fig3_ising_ladder.pdf` |
| `tab:ising_rates`, true flip probabilities | typed in the paper; `tests.test_glauber_calibration` checks the engine against them | |
| `tab:ising_recipe`, training recipe | `RECORD` in `domains.ising.train` (Ising §2) | |
| `tab:fig3_ising`, domain growth against the ground truth | `grade_campaign512`, then `build_tables512` (Ising §4) | `eval/tables512/fig3_ising.tex` |
| `tab:ising_locality`, perception radius | `grade_campaign512`, then `build_tables512` | `eval/tables512/ising_locality.tex` |
| `fig:ising_locality`, perception radius | `grade_campaign512`, then `fig_growth512 locality` | `figures/fig_ising_locality.pdf` |
| `fig:ising_rules`, learned flip rates | `flip_probe` (Ising §3), then `fig_rule_values` | `figures/fig_ising_rules.pdf` |
| `fig:ising_rules_time`, flip rates over time | `flip_probe`, then `fig_rule_time` | `figures/fig_ising_rules_time.pdf` |
| `fig:ising_marginals`, choice of training marginals | the placement sweep (Ising §2), `grade_campaign512`, `grade_marginals512`, then `fig_marginals512` | `figures/fig_ising_marginals.pdf` |
| `fig:ising_strip_full`, generated fields, all methods | `fig_filmstrip512 full` | `figures/fig_ising_strip_full.pdf` |
| **Heart regeneration** | | |
| `fig:fig3_zebra`, panels B and C | `fig_perturbation` (hearts §4); panel A is drawn by hand | `figures/hearts/fig_perturbation_<run>.pdf` |
| `tab:heart-group-support`, the a-priori grouping | `tables` (hearts §4) | `figures/hearts/table_heart_celltype_grouping.tex` |
| `tab:heart-div`, section-to-section variability | `tables` | `figures/hearts/table_heart_variabiltiy.tex` |
| `fig:app-hearts-perturbed-trajectory`, the knock-out filmstrip | `fig_unroll` | `figures/hearts/fig_unroll_<run>.pdf` |
| `fig:app-hearts-group-recovery`, the per-seed figure | `fig_group_recovery` | `figures/hearts/fig_group_recovery.pdf` |
| `tab:app-heart-group-recovery`, `tab:app-heart-fidelity` | `tables` | `figures/hearts/table_heart_model_fidelity.tex` |
| `tab:app-heart-aux-ablations`, the auxiliary losses | `tables` | `figures/hearts/table_heart_aux_ablation.tex` |

## Reproducing the cycling-MNIST experiment

### 1. Frozen classifier

```bash
python -m domains.mnist.classifier   # data/model-checkpoints/mnist-classifier.pth
```
This classifier supplies digit labels and features for Frechet distances. Retraining can change both; `domains/mnist/classifier.py` records the SHA-256 of the checkpoint used in the paper.

### 2. Training

Seven runs per method, indexed 0..6 in the run names, tables and figures. Run i is trained with seed 42 + i (REGIS and the non-local control), 13 + i (MMtSBM) or i (MMSFM); `domains/mnist/results/runs.py` lists every stem.

```bash
python -m domains.mnist.train --g-arch regis   --seed 42 --run-name mnist_regis_seed0              # ... --seed 48, seed6
python -m domains.mnist.train --g-arch control --seed 42 --run-name mnist_control_nonlocal_seed0   # ... --seed 48, seed6
```
About 2.4 h per run. Defaults are in `RECORD` in `domains/mnist/train.py`; the settings used are saved in `<run>/config.json`.

The transport baselines train on the digit loop 0 -> 1 -> ... -> 9 -> 0:
```bash
python -m domains.mnist.data digitloop    # data/mnist/digitloop/{class,eval}_t<i>.pt: marginals 0, 1, ..., 9, 0
```
Marginal i holds the training digits of class i % 10; the two zero marginals are disjoint halves of the zeros.

**MMtSBM** (training seeds 13..19):
```bash
python -m baselines.mmtsbm.train --run-name mnist_mmtsbm_seed0 --seed 13 --data data/mnist/digitloop \
  --prefix class_t --marginals 11 --sigma 1.2 --batch 256                    # ... --seed 19, seed6
```
The trainer's defaults do the rest: a 50k-step warm-up on independent couplings, then two IMF iterations of 12.5k steps, both directions each time, with 60 Euler-Maruyama steps per bridge for the coupling. About 1.8 h per seed on one H100.

**MMSFM** (the authors' code, fetched and patched, see [baselines/mmsfm/](../baselines/mmsfm/README.md)):
```bash
bash baselines/mmsfm/fetch_upstream.sh
python -m baselines.mmsfm.train --domain mnist --seed 0 --run-name mnist_mmsfm_seed0   # ... --seed 6, seed6
```
About 5 h per seed. Settings are in `SETTINGS['mnist']` in `baselines/mmsfm/train.py`.

### 3. Captures

```bash
python -m domains.mnist.results.capture_frames --run mnist_regis_seed0     # data/mnist/captures/<stem>.npz; about 16 min (control: 27)
```
5000 trajectories from real test digits, each run for 1000 transitions. Frames are kept unclamped. The array format is documented in `capture_frames.py`.

```bash
python -m domains.mnist.results.capture_mmtsbm --run mnist_mmtsbm_seed0     # about 67 min per seed on one H100
python -m domains.mnist.results.capture_mmsfm --run mnist_mmsfm_seed0 --shard 0   # ... --shard 5, one GPU each
python -m domains.mnist.results.capture_mmsfm --run mnist_mmsfm_seed0 --merge     # data/mnist/captures/<stem>.npz
```
MMtSBM starts from 500 distinct test digits per class (`eval_t<i>.pt`) and crosses one bridge per transition using the final forward EMA drift. MMSFM uses the shared starts from `capture_frames`, in six shards of about 4 h each. The scripts document differences in classification and grid selection.

### 4. Tables and figures

```bash
python -m domains.mnist.results.fd_ci          # eval/mnist/fd_ci.json: distances, bootstrap intervals, the real-vs-real floor
python -m domains.mnist.results.fd_tables      # figures/mnist/horizon.tex, table_supp.tex
python -m domains.mnist.results.fig_samples    # figures/mnist/fig_perseed.pdf
python -m domains.mnist.results.fig_style      # figures/mnist/fig_style.pdf  (GPU)
```

The index at the top maps each output to its paper item.

## Reproducing the Ising experiment

### 1. Data

```bash
python -m domains.ising.data marginals          # data/ising_marginals.npz
python -m domains.ising.data windows            # data/ising_windows/       (MMSFM, MMtSBM)
python -m domains.ising.data pairs --n 15000    # data/ddpm_pairs_15000.npz  (DDPM)
```
Independent 512^2 Glauber quenches at T = 1, read out at t = 3, 10, 17, 300, 1000. REGIS, the non-local control and the pair-conditional GAN train on 100 fields per time; the transport baselines on 192^2 windows cut from 3000 fields per time (the data budget asymmetry is stated in the paper); the DDPM on pairs. The data-efficiency ladder and the marginal-placement study use:
```bash
python -m domains.ising.data marginals --n-train 3000      # REGIS ladder (smaller budgets: the trainer's --n-train)
python -m domains.ising.data pairs --n 50                  # DDPM ladder: 50, 150, 500, 1500, 5000
python -m domains.ising.data marginals --times 3,55,1000   # one file per placement grid
```
Every field is a function of a fixed seed, and a time shared by two marginals files has the same fields in both.

### 2. Training

**REGIS** (three seeds; the k5/k7 rows add `--perception-ksize 5` or `7`):
```bash
python -m domains.ising.train --seed 0 --run-name regis_k3_seed0
```
Defaults are in `RECORD` in `domains/ising/train.py`: 30k updates on 128^2 crops, batch size 8. Evaluation loads the EMA generator.

**Non-local control** (same trainer; the update network is a stack of dilated convolutions):
```bash
python -m domains.ising.train --g-arch control --seed 0 --run-name control_nonlocal_seed0
```
Dilations 1, 2, 5, 9 give a 35-pixel receptive field per half-step and 69 pixels per full step. About 6 h of training (REGIS k3: 2.5 h; k7: about 4 h).

**Pair-conditional GAN**: the control command plus `--pair-cond` (the critic sees (start, endpoint) pairs; real endpoints are engine-rolled from the same start):
```bash
python -m domains.ising.train --g-arch control --pair-cond --seed 0 --run-name pairgan_seed0
```
About 6.5 h.

**DDPM** (trained on aligned pairs):
```bash
python -m models.ddpm --run-name ddpm_pairs15000_seed0 --seed 0 --compile
python -m domains.ising.instruments.ddpm_cache --run ddpm_pairs15000_seed0
```
Default settings: v-prediction, cosine schedule over 1000 steps, 4.65M parameters, 192^2 windows with the loss on the central 128^2, 40k steps, 250 ancestral sampling steps. About 35 min of training; the 512-field cache takes about 6 h. The data-efficiency ladder repeats both for N = 50, 150, 500, 1500, 5000 and seeds S = 0-2, on the pairs of `data pairs --n N`:
```bash
python -m models.ddpm --data data/ddpm_pairs_N.npz --run-name ddpm_pairsN_seedS --seed S --compile
python -m domains.ising.instruments.ddpm_cache --run ddpm_pairsN_seedS
```

**MMtSBM** (seeds 13–15):
```bash
python -m baselines.mmtsbm.train --run-name mmtsbm_seed13 --seed 13 \
  --batch 32 --outer-iters 1 --only-direction forward --halo 32 --symmetrise z2rot180 --circular
```
Ising uses 192^2 windows with the loss on the central 128^2, z2+rot180 symmetry averaging, and circular padding. `--outer-iters 1 --only-direction forward` trains only the forward drift during warm-up; its final weights are saved as "ema". Rollouts use 120 Euler-Maruyama steps per bridge. About 12 h of training.

**MMSFM** (seeds 0–2; the authors' code, fetched and patched, see [baselines/mmsfm/](../baselines/mmsfm/README.md)):
```bash
bash baselines/mmsfm/fetch_upstream.sh
python -m baselines.mmsfm.train --run-name mmsfm_seed0 --seed 0
```
Settings are in `SETTINGS['ising']` in `baselines/mmsfm/train.py`. About 18 h of training.

**Data ladder and marginal-placement sweep** (REGIS k3 on other data):
```bash
# ladder: F = 10, 30, 100, 300, 1000, 3000 fields per marginal, seeds S = 0-2; about 1.7 h each
python -m domains.ising.train --fixed-crops --bf16-prefix --n-train F \
  --data data/ising_marginals_n3000.npz --seed S --run-name regis_k3_fixedcrops_fF_seedS
# placement: grid i of runs.GRIDS at seed 100 + i; grids 11, 13, 14 also at 200 + i (runs.PLACEMENT);
# 2-5 h each, depending on the grid
python -m domains.ising.train --data data/ising_marginals_<times>.npz \
  --seed 101 --run-name marginals_<times>_seed101   # <times> joined by "_", e.g. 3_55_1000
```
The first 100 fields of the 3000-field file are the 100-field file's, so for F <= 100 either file gives the same training data. Each run is cached with `rule_cache --run <run name>` (§3).

### 3. The 512-field evaluation protocol

All models use the same initial fields: 512 fields = 8 chunks of 64, chunk j quenched from torch seed 1234 + j. Ground truth is the median over three engine seeds (4321, 8765, 2468) rolled identically. One cache per run, `data/rollouts/<stem>__lat512_ns512.npz` (GPU):

```bash
python -m domains.ising.instruments.rule_cache    --engine 4321             # truth_seed4321; also 8765, 2468; 2 min
python -m domains.ising.instruments.rule_cache    --run regis_k3_seed0      # REGIS, control, pair GAN; 1.5-2.5 h
python -m domains.ising.instruments.ddpm_cache    --run ddpm_pairs15000_seed0   # about 6 h
python -m domains.ising.instruments.mmtsbm_cache  --run mmtsbm_seed13       # about 12 h
python -m domains.ising.instruments.mmsfm_cache   --run mmsfm_seed0         # about 21 h
```
Caches are packed bits at fixed read-out times (`domains/ising/instruments/rollouts.py`): the engine and the update rules to t = 4000, the DDPM from t = 13, the transport rows at their marginals and at read-outs part-way through each transition (`rollouts.between_marginals`), to t = 1000. Every metric is recomputed offline from them. The filmstrip also reads raw (un-thresholded) strips of a few fields; the sampler noise depends on the batch, so each strip is reproduced only at the `--ns` shown:
```bash
python -m domains.ising.instruments.mmsfm_cache  --run mmsfm_seed0 --ns 2 --raw --label mmsfm_seed0_strip
python -m domains.ising.instruments.mmtsbm_cache --run mmtsbm_seed13 --ns 4 --raw --label mmtsbm_seed13_strip
python -m domains.ising.instruments.ddpm_cache   --run ddpm_pairs15000_seed0 --ns 2 --raw \
  --grid 17,150,300,700,1000,1700,2000,2261,3007,4000 --label ddpm_pairs15000_seed0_strip
python -m domains.ising.instruments.ddpm_cache   --run ddpm_pairs15000_seed0 --ns 2 --raw \
  --grid 13 --label ddpm_pairs15000_seed0_strip13
```
The DDPM needs two chains from t = 3: 3 -> 17 onwards for the later columns, and 3 -> 13, whose frame fills the t = 10 column (3 -> 10 is below the trained jump floor of 10, and 13 -> 17 too short to share a chain).

```bash
python -m domains.ising.results.grade_campaign512   # -> eval/campaign512.json
python -m domains.ising.results.grade_marginals512  # -> eval/marginals512.{json,md} (placement sweep)
python -m domains.ising.instruments.flip_probe      # -> eval/flip_table.{json,md} (learned rules)
```
The metric is L = 1/rho on the despeckled field (see `domains/ising/instruments/despeckle.py` and the paper's appendix); the isolated-site fraction is also computed, but only the placement sweep's interim table (eval/marginals512.md) shows it, and the paper does not report it.

### 4. Tables and figures

```bash
python -m domains.ising.results.build_tables512     # eval/tables512/{fig3_ising,ising_locality}.tex
python -m domains.ising.results.fig_growth512       # fig3_ising_growth + fig_ising_locality: log-log growth, perception radius
python -m domains.ising.results.fig_ladder512       # fig3_ising_ladder: data efficiency
python -m domains.ising.results.fig_filmstrip512    # fig3_ising_filmstrip_strip + fig_ising_strip_full: generation examples
python -m domains.ising.results.fig_marginals512    # fig_ising_marginals: choice of training marginals
python -m domains.ising.results.fig_rule_values     # fig_ising_rules: flip rate per neighbour class
python -m domains.ising.results.fig_rule_time       # fig_ising_rules_time: flip rate over time
```
Figures land in `figures/`. The index at the top maps each output to its paper item.

### 5. Naming

`domains/ising/results/runs.py` defines run names, training seeds, and evaluation cache paths.

## Reproducing the heart-regeneration experiment

Everything runs from the repo root, with artifacts landing beside the code as in the other experiments. Training and the full evaluation want a GPU; the tables and figures run on CPU. No extra dependency over the other two experiments.

### 1. The cohort

The experiment's data is not in this repo, and the derived grid is not shipped: one plain-text table is downloaded and the producer rebuilds the cohort from it.

```bash
# https://zenodo.org/records/14991776 -> Stereo-seq-regeneration.meta.txt (21.4 MB)
#   the processed data for Li et al. 2025, Nat. Commun. 16:3716, CC BY 4.0.
#   Attribution is a licence condition: cite the atlas in anything built on it.
mkdir -p data/hearts && mv Stereo-seq-regeneration.meta.txt data/hearts/
python -m domains.hearts.data_cohort        # -> data/hearts/hearts_48.npz
```

No expression data is used anywhere in this experiment — the channels are one-hot cell-type labels from the `annotation` column — so the Zenodo table is the whole input.

The producer checks its output against the invariants of the tensor the paper's runs trained on (105 sections, per-bin capacity within 4.0, nothing outside the section mask, and the measured per-stage damage profile) before writing, so a wrong or re-released download fails in seconds rather than after a day of training. Verified against that tensor: all 105 sections agree to 2.4e-7, the float32 round-trip of the output encoding, with masks bit-identical.

The 87 hand-painted wound footprints and 18 apex zones in `domains/hearts/annotations/` ship WITH this repo and are the exception to its code-only rule. The atlas carries no annotation of injury location at all, so nothing can regenerate them.

### 2. Training

Eighteen REGIS-family runs of record: seven REGIS seeds, three per ablation arm, one REGIS run holding a whole heart out, and four REGIS runs each without one auxiliary loss. The three arms differ in `--g-arch` alone:

```bash
python -m domains.hearts.train --g-arch regis   --seed 0 --run-name hearts_regis_seed0
python -m domains.hearts.train --g-arch control --seed 0 --run-name hearts_control_nonlocal_seed0
python -m domains.hearts.train --g-arch time    --seed 0 --run-name hearts_time_seed0
python -m domains.hearts.train --g-arch regis   --seed 0 --holdout-heart T1_S1 \
    --run-name hearts_regis_loo_T1_S1
python -m domains.hearts.train --g-arch regis   --seed 0 --drop-loss silhouette \
    --run-name hearts_regis_no_silhouette          # ... and capacity, heal, zero
```

The configuration of every run is `RECORD` in the trainer; only the arm, the seed, the run name, the length, the held-out heart and the dropped loss are flags. 100k iterations at batch 32, about 335 ms per iteration on one H100. `--g-arch time` is the ablation the experiment is built to fail: it hands the rule the stage label, so it can read a clock instead of building one.

The transport baselines train on the loop uninjured -> 6 hpa -> ... -> 28 dpa -> uninjured.

**MMtSBM** (one run, training seed 13):
```bash
python -m domains.hearts.data_bridges      # data/hearts/bridges/: nine z-scored marginals, 21 channels
python -m baselines.mmtsbm.train --run-name hearts_mmtsbm_seed0 --seed 13 --data data/hearts/bridges \
  --prefix hearts_t --marginals 9 --sigma 1.2 --blocks 64,128,256
```
The two ends of the loop are disjoint halves of the 18 uninjured sections. The trainer's defaults do the rest: a 50k-step warm-up on independent couplings, then three IMF iterations of 12.5k steps. The checkpoint of record is the warm-up's, `ckpt_0000_forward.pt`: the fitting iterations degraded the cascade on this cohort. About 1.8 h on one H100.

**MMSFM** (seeds 0 and 1; the authors' code, fetched and patched, see [baselines/mmsfm/](../baselines/mmsfm/README.md)):
```bash
bash baselines/mmsfm/fetch_upstream.sh
python -m baselines.mmsfm.train --domain hearts --seed 0 --run-name hearts_mmsfm_seed0   # ... --seed 1, seed1
```
All 22 cohort channels are generated jointly, from the 81 sections of 27 training hearts; one heart per stage is held out (`VALIDATION_HEARTS`). Settings are in `SETTINGS['hearts']` in `baselines/mmsfm/train.py`. About 8 h per seed on one H100. The checkpoints of record were trained with an earlier copy of this driver, so a retrain reproduces their configuration, not their random stream.

### 3. Evaluation

Three instruments per run, each rolling the shared driver (`domains/hearts/rollout.py`) under a different intervention and writing one cache under `eval/hearts/`. Nothing downstream recomputes a measurement.

```bash
for r in hearts_regis_seed0 ... ; do
  python -m domains.hearts.instruments.marginals --run $r    # profiles vs the cohort
  python -m domains.hearts.instruments.cascade   --run $r    # the border-zone chain
  python -m domains.hearts.instruments.knockout  --run $r    # the perturbation screen
done
```

The screen compares each heart with itself: 100 wounded sections, each rolled twice unperturbed and once per knock-out from the same start, and ρ(c) is the mean over hearts of the knock-out's difference from the first unperturbed run, divided by that heart's own rerun difference. The trajectories are 48 rollouts, recorded from the wound; the cascade averages three unperturbed replicates and two per block over 64 hearts.

The transport baselines have their own instruments, which write the same caches:

```bash
python -m domains.hearts.instruments.mmtsbm --run hearts_mmtsbm_seed0   # profiles, chained and one-step; minutes
python -m domains.hearts.instruments.mmsfm  --run hearts_mmsfm_seed0    # profiles, cascade, screen; ~4 min on one H100
```
MMtSBM carries the nine uninjured sections of the loop's first end along the bridges with the noise off, either chained or restarting each bridge from real sections; knock-outs are not run on it. MMSFM follows the protocol of the evaluation of record of these checkpoints: twelve rollouts (four noise replicas of the three held-out uninjured sections), a knock-out that holds a channel's state, drift and noise at zero at every solver stage, readouts over each sample's occupied support, and the held-out sections as the cohort reference. The chained MMtSBM rollout is deterministic and amplifies last-bit differences along the chain: on the marginals the paper's run used, this instrument reproduces its MMtSBM values to the digit (0.495 / 0.511 chained, 0.842 / 0.922 one-step, curves to 1e-7), while the marginals rebuilt from the download, which agree with those to 4e-6, move the chained values by about 0.1. MMSFM's noise stream is the one of record, so its values reproduce to floating-point execution.

The two interventions are not the same operation. The screen clamps a channel to its pre-step value, so a type can decay but never be recruited — the closer analogue of a loss-of-function. The cascade read-out forces it to zero, because there the question is whether a downstream state can rise at all.

### 4. Tables and figures

```bash
python -m domains.hearts.results.tables                       # the four appendix tables
python -m domains.hearts.results.fig_perturbation --run hearts_regis_seed0
python -m domains.hearts.results.fig_group_recovery           # one panel per model
python -m domains.hearts.results.fig_marginals --run hearts_regis_seed0 \
    --compare hearts_time_seed0
python -m domains.hearts.results.fig_unroll --run hearts_regis_seed0
python -m domains.hearts.results.fig_cohort                   # the observed sections
```

Outputs land in `figures/hearts/`. The paper's figures are composites assembled from these panels, so the filenames differ: `fig_perturbation` is panels B and C of the main heart figure, `fig_group_recovery` the per-seed appendix figure, `fig_unroll` the knock-out filmstrip. Panel A of the main figure is a hand-drawn scheme and is not regenerated here.

`tables.py` writes the files the paper includes — `table_heart_model_fidelity`, `table_heart_aux_ablation`, `table_heart_celltype_grouping` and `table_heart_variabiltiy` — and names and skips any run without a cache, so the model tables build from a partial set of runs and simply carry fewer rows. Every per-seed row, family mean and across-seed sign test comes out of `domains/hearts/results/stats.py`; no cell is retyped and no summary line is computed by hand.

### 5. Naming

`domains/hearts/results/runs.py` is the single source of run names (`hearts_regis_seed0..6`, `hearts_time_seed0..2`, `hearts_control_nonlocal_seed0..2`, `hearts_regis_loo_T1_S1`, `hearts_regis_no_{silhouette,capacity,heal,zero}`, `hearts_mmsfm_seed0..1`, `hearts_mmtsbm_seed0`).

### 6. Tests

```bash
python -m tests.test_smoke_hearts     # producer + all three arms + the rollout driver, CPU
```
