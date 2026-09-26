"""Train MMSFM on Ising, cycling-MNIST or zebrafish-heart snapshots.

Requires the upstream code downloaded by fetch_upstream.sh. SETTINGS holds
the experiment configurations. Symmetry and eager/compiled gradient checks
run before training; checkpoints are saved under runs/<run-name>."""
import argparse
import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from baselines.mmsfm.net import build_heads, build_unets, configure_execution, periodic_time_embedding, upstream
from domains.hearts import cohort as C
from domains.ising.data import TIMES
from workspace import WORK, resolve

# Experiment settings: channels is U-Net width; attention lists feature-map sizes.
# The loss uses the central (side-2*halo)^2 pixels. chunks splits a batch for
# memory; accumulation combines batches before each optimiser update.
SETTINGS = dict(ising=dict(channels=96, attention='48,24', side=192, planes=1, halo=32, sigma=0.6, chunks=8,
                           accumulation=1, steps=20000, data='ising_windows'),
                mnist=dict(channels=64, attention='8,4', side=32, planes=1, halo=0, sigma=0.15, chunks=1,
                           accumulation=2, steps=40000, data='mnist/digitloop'),
                hearts=dict(channels=64, attention='12,6', side=48, planes=22, halo=0, sigma=0.05, chunks=1,
                            accumulation=2, steps=40000, data='hearts'))
WINDOW = 2   # upstream's K: each spline window spans K + 1 = 3 consecutive marginals (the paper's sliding triplets)
BATCH_PER_WINDOW = 16   # samples per marginal in each window
LR_MIN, LR_MAX, LR_WARMUP = 1e-8, 1e-4, 500   # linear up to LR_MAX over LR_WARMUP updates, then down to LR_MIN
EPOCH, CHECKPOINT = 200, 2000   # the LR schedule counts epochs of EPOCH updates
UPSTREAM_COMMITS = dict(upstream='f8c702b15fba75677d3a89fed2190c6ca9f3e106',
                        torchcfm='af8fec6f6dc3a0dc7f8fb25d2ee0ca819fa5412f')


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


class LossLog:
    """Adapter for upstream wandb calls that writes one losses.jsonl row per update."""

    def __init__(self, outdir):
        self.path = outdir / 'losses.jsonl'
        self.pending = {}
        self.started = time.monotonic()

    def log(self, values, commit=True):
        self.pending.update(values)
        if not commit:
            return
        row = {key: float(v) for key, v in self.pending.items()}
        row['step'] = int(row.pop('grad_step')) + 1
        row['elapsed_seconds'] = time.monotonic() - self.started
        if not all(np.isfinite(v) for v in row.values()):
            raise FloatingPointError(f'Nonfinite loss: {row}')
        with self.path.open('a') as f:
            f.write(json.dumps(row)+'\n')
        if row['step'] <= 3 or row['step'] % 20 == 0:
            print('TRAIN '+json.dumps(row), flush=True)
        self.pending.clear()


class QuietBar:
    """Stands in for upstream's tqdm bar."""

    def update(self, n):
        pass


@contextmanager
def opened(model):
    """Temporarily randomise zero weight matrices so validation exercises nonzero outputs."""
    saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    try:
        with torch.no_grad():
            for param in model.parameters():
                if param.ndim > 1 and not torch.count_nonzero(param):
                    param.normal_(0, .01)
        yield
    finally:
        model.load_state_dict(saved)


@torch.no_grad()
def check_heads(models, dims, device, **twins):
    """Check each head against the supplied symmetry transforms on nonzero weights."""
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    reports = []
    for model in models:
        with opened(model):
            try:
                model.eval()
                x = torch.randn(2, *dims, device=device)
                t = torch.tensor([.125, .875], device=device)
                # at fp32: TF32 reductions can depend on the batch position, which a twin may swap
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                y = model(t, x)
                row = dict(output_rms=float(y.square().mean().sqrt()),
                           **{name: float((y - twin(model, t, x)).abs().max()) for name, twin in twins.items()})
                assert row['output_rms'] > 1e-6 and all(row[name] < 2e-5 for name in twins), row
                reports.append(row)
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
                model.train()
    return dict(passed=True, heads=reports)


def compile_parity(models, batch, fm, outdir):
    """Check eager and compiled outputs and gradients, then return the compiled heads."""
    t, xt, ut, eps, ab = batch
    lam = fm.compute_lambda(t, ab)[:, None]
    report, compiled_heads = [], []
    for model, kind in zip(models, ['flow', 'score']):
        with opened(model):
            params = list(model.parameters())
            def compute(call):
                value = call(t, xt)
                loss = (value-ut).square().mean() if kind == 'flow' else (lam*value.reshape_as(eps)+eps).square().mean()
                grads = torch.autograd.grad(loss, params)
                return value.detach(), [g.detach() for g in grads]
            eager, eg = compute(model)
            compiled = torch.compile(model)
            actual, cg = compute(compiled)
        out_error = float((actual-eager).norm() / eager.norm().clamp_min(1e-8))
        diff = sum((a-b).double().square().sum() for a, b in zip(cg, eg))
        norm = sum(g.double().square().sum() for g in eg)
        grad_error = float((diff/norm.clamp_min(1e-16)).sqrt())
        row = dict(model=kind, output_relative_l2=out_error, gradient_relative_l2=grad_error)
        report.append(row)
        print('COMPILE_PARITY '+json.dumps(row), flush=True)
        if not (out_error < 1e-3 and grad_error < 1e-3):   # also catches NaN
            raise RuntimeError(f'Compile parity failed: {row}')
        compiled_heads.append(compiled)
    json_write(outdir/'compile_parity.json', dict(passed=True, comparisons=report))
    return compiled_heads


def ising(tr, s, data, dims, seed, device):
    """The Ising paper runs: the coarsening marginals on t in [0, 1] and Z2 heads.
    Returns the marginals, their times, the heads, the RNG state the batches start
    from, the head check and the config."""
    n_marg = len(TIMES) + 1   # t = 0 and the five marginals
    X = []
    for i in range(n_marg):
        x = torch.load(data/f'ising_t{i}.pt', weights_only=True)
        assert x.shape == (6000, *dims) and (x.abs()==1).all()
        X.append(x.to(device))
    # RNG order of the paper runs: the heads initialise from the seeded stream; the
    # batches continue it after a 64-channel U-Net pair, built here and discarded.
    before_models = rng_state()
    build_unets(tr, dims, 64, s['attention'], device)
    sampler_rng = rng_state()
    restore_rng(before_models)
    models = build_heads(tr, dims, s['channels'], s['attention'], device)
    config = dict(dims=dims, seed=seed, steps=s['steps'], checkpoint_steps=CHECKPOINT, n_steps_epoch=EPOCH,
                  warmup_steps=LR_WARMUP, lr_min=LR_MIN, lr_max=LR_MAX, sigma=s['sigma'],
                  channels=s['channels'], window=WINDOW, symmetry='z2', chunks=s['chunks'], halo=s['halo'],
                  spline='monotone cubic', batch_per_window=BATCH_PER_WINDOW,
                  effective_batch=BATCH_PER_WINDOW*(n_marg-WINDOW),
                  anchors=n_marg, normalization='+-1 spins', tf32_matmul=True, tf32_cudnn=True,
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), **UPSTREAM_COMMITS,
                  data=str(data.resolve()), attention_resolutions=s['attention'], circular_padding=True,
                  parameters_per_network=sum(p.numel() for p in models[0].parameters()),
                  train_counts=[len(x) for x in X])
    return X, np.arange(n_marg)/(n_marg-1), models, sampler_rng, dict(z2_max_abs=lambda m, t, x: -m(t, -x)), config


def mnist(tr, s, data, dims, seed, device):
    """The MNIST paper runs: the digit loop 0 -> 1 -> ... -> 9 -> 0 with period one
    in t (digit d at t = d/10) and plain heads on a periodic clock. Returns what ising()
    returns."""
    pools = [torch.load(data/f'class_t{i}.pt', weights_only=True).to(device) for i in range(11)]
    # Anchors 0 and 10 are disjoint halves of the zeros; their union is the loop's one 0.
    # The aliases 0 and 1 at t = 1.0 and 1.1 make the windows 8,9,0 and 9,0,1 wrap.
    zero = torch.cat([pools[0], pools[10]])
    X = [zero, *pools[1:10], zero, pools[1]]
    # RNG order of the paper runs: a two-channel digit classifier, discarded here, initialises
    # from the seeded stream before the heads; the batches draw from a stream of their own.
    torch.nn.Sequential(torch.nn.Conv2d(2, 32, 3), torch.nn.Conv2d(32, 64, 3),
                        torch.nn.Linear(64*8*8, 128), torch.nn.Linear(128, 10))
    models = build_unets(tr, dims, s['channels'], s['attention'], device)
    seed_all(seed + 100000)
    config = dict(domain='mnist', dims=dims, seed=seed, steps=s['steps'], checkpoint_steps=CHECKPOINT,
                  n_steps_epoch=EPOCH, warmup_steps=LR_WARMUP, lr_min=LR_MIN, lr_max=LR_MAX, sigma=s['sigma'],
                  channels=s['channels'], window=WINDOW, symmetry='none', cyclic=True, chunks=s['chunks'],
                  spline='monotone cubic', batch_per_window=BATCH_PER_WINDOW, accumulation=s['accumulation'],
                  effective_batch=BATCH_PER_WINDOW*(len(X)-WINDOW)*s['accumulation'], anchors=11,
                  normalization='already 2*x-1', tf32_matmul=False, tf32_cudnn=False,
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), **UPSTREAM_COMMITS,
                  data=str(data.resolve()), digit_of_anchor=[i % 10 for i in range(11)], frame='resize22_pad5',
                  zero_endpoints='union of disjoint training zero pools, shared around cycle; heldout unchanged',
                  attention_resolutions=s['attention'], training_windows=len(X)-WINDOW,
                  training_control_digits=[i % 10 for i in range(len(X))],
                  time_embedding='period-one integer Fourier harmonics 1..channels/2',
                  parameters_per_network=sum(p.numel() for p in models[0].parameters()),
                  train_counts=[len(x) for x in X], training_rng_seed=seed + 100000)
    return X, np.arange(len(X))/10, models, rng_state(), dict(period_max_abs=lambda m, t, x: m(t + 1, x)), config


# held-out hearts, one per stage; training uses the other 27 (81 sections)
VALIDATION_HEARTS = ('T1_S1', 'T2_S6', 'T3_S5', 'T4_S3', 'T5_S2', 'T6_S3', 'T7_S2', 'T8_S3')


def heart_of(section_id):
    """'T1_C2_S1' -> 'T1_S1': the three chips of a section share a heart"""
    t, _chip, fish = section_id.split('_')
    return f'{t}_{fish}'


def hearts(tr, s, data, dims, seed, device):
    """The heart runs: the loop uninjured -> 6 hpa -> ... -> 28 dpa -> uninjured with period
    one in t (stage k at t = k/8), all 22 cohort channels generated jointly, plain heads on a
    periodic clock. Returns what ising() returns."""
    z = np.load(data/'hearts_48.npz', allow_pickle=False)   # composition is already native/2 - 1
    train = np.array([heart_of(str(i)) not in VALIDATION_HEARTS for i in z['section_id']])
    stage = np.array([C.TP_ORDER.index(str(t)) for t in z['timepoint']])
    pools = [torch.from_numpy(z['composition'][train & (stage == k)]).to(device) for k in range(C.N_STAGES)]
    # the aliases of stages 0 and 1 at t = 1 and 1.125 make the windows 6,7,0 and 7,0,1 wrap
    X = [*pools, pools[0], pools[1]]
    models = build_unets(tr, dims, s['channels'], s['attention'], device)
    config = dict(domain='hearts', dims=dims, seed=seed, steps=s['steps'], checkpoint_steps=CHECKPOINT,
                  n_steps_epoch=EPOCH, warmup_steps=LR_WARMUP, lr_min=LR_MIN, lr_max=LR_MAX, sigma=s['sigma'],
                  channels=s['channels'], window=WINDOW, symmetry='none', cyclic=True, chunks=s['chunks'],
                  spline='monotone cubic', batch_per_window=BATCH_PER_WINDOW, accumulation=s['accumulation'],
                  effective_batch=BATCH_PER_WINDOW*(len(X)-WINDOW)*s['accumulation'], anchors=len(X),
                  normalization='native cell units/2 - 1', tf32_matmul=True, tf32_cudnn=True,
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), **UPSTREAM_COMMITS,
                  data=str((data/'hearts_48.npz').resolve()), validation_hearts=list(VALIDATION_HEARTS),
                  attention_resolutions=s['attention'], training_windows=len(X)-WINDOW,
                  time_embedding='period-one integer Fourier harmonics 1..channels/2',
                  parameters_per_network=sum(p.numel() for p in models[0].parameters()),
                  train_counts=[len(x) for x in pools])
    return X, np.arange(len(X))/C.N_STAGES, models, rng_state(), dict(period_max_abs=lambda m, t, x: m(t + 1, x)), config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['ising', 'mnist', 'hearts'], default='ising')
    p.add_argument('--run-name', required=True, help='writes runs/<run-name>, which the cache or capture reads')
    p.add_argument('--data', type=resolve, help='marginals directory (default: data/ising_windows from '
                   '`python -m domains.ising.data windows`, data/mnist/digitloop from `python -m domains.mnist.data digitloop`, '
                   'data/hearts from `python -m domains.hearts.data_cohort`)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    s = SETTINGS[a.domain]
    data = a.data or WORK/'data'/s['data']
    out = WORK/'runs'/a.run_name
    started = time.monotonic()
    epochs = s['steps'] // EPOCH
    # before the import: multimarginal_cfm reads MMSFM_VECSPLINE at import
    os.environ.update(MMSFM_VECSPLINE='1', MMSFM_CHUNKS=str(s['chunks']), MMSFM_HALO=str(s['halo']))
    tr = upstream()
    torch.set_num_threads(4)
    if a.domain == 'mnist':   # fp32 throughout, upstream's attention
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    else:
        configure_execution()
    if a.domain != 'ising':   # the loops' clock is periodic
        from torchcfm.models.unet import unet
        unet.timestep_embedding = periodic_time_embedding
    if not torch.cuda.is_available():
        raise RuntimeError('This driver needs a CUDA GPU')
    out.mkdir(parents=True, exist_ok=True)
    # Prevent reruns from overwriting an existing trajectory.
    with (out/'started.json').open('x') as f:
        json.dump(dict(unix_time=time.time()), f)
    seed_all(a.seed)
    device = 'cuda'
    dims = (s['planes'], s['side'], s['side'])
    setup = dict(ising=ising, mnist=mnist, hearts=hearts)[a.domain]
    X, zt, models, sampler_rng, twins, config = setup(tr, s, data, dims, a.seed, device)
    flow, score = models
    # sm=True here and below: also train the score head
    fm = tr.build_FM(K=WINDOW, sm=True, zt=zt, sigma=s['sigma'], spline='cubic', monotonic=True, method='exact',
                     t_sampler='stratified', diff_ref='miniflow', device=device)
    json_write(out/'config.json', config)
    print('Checking network symmetries and compiled outputs/gradients', flush=True)
    json_write(out/'adaptation_checks.json', check_heads(models, dims, device, **twins))
    batch = tr.get_batch(fm, X, BATCH_PER_WINDOW, dims, zt, sm=True, K=WINDOW, device=device)
    for item in batch:
        assert torch.isfinite(item).all()
    compiled_flow, compiled_score = compile_parity(models, tuple(x[:4] for x in batch), fm, out)
    del batch
    restore_rng(sampler_rng)   # Validation draws must not alter the training RNG sequence.
    opt, sch = tr.build_optimizer_and_scheduler(a.domain, s['side'], flow, score, lrmin=LR_MIN, lrmax=LR_MAX,
                                                epochs=epochs, n_steps_epoch=EPOCH, total_iters_inc=LR_WARMUP)
    # upstream fills these in place; unused here
    flow_losses, score_losses, lrs = (np.zeros((epochs, EPOCH)) for _ in range(3))
    log = LossLog(out)
    # training runs the compiled heads; they share their parameters with flow and score, which are saved
    for epoch in range(epochs):
        tr.train_epoch(run=log, X=X, batch_size=BATCH_PER_WINDOW, accum_steps=s['accumulation'], dims=dims, zt=zt,
                       n_steps=EPOCH, model=compiled_flow, score_model=compiled_score, optimizer=opt,
                       scheduler=sch, K=WINDOW, FM=fm, sm=True, flow_losses=flow_losses, score_losses=score_losses,
                       lrs=lrs, epoch_num=epoch, outdir=str(out), pbar=QuietBar(), device=device)
        step = (epoch+1)*EPOCH
        if step % CHECKPOINT == 0:
            torch.save(dict(model_state_dict=flow.state_dict(), score_model_state_dict=score.state_dict(),
                            step=step, config=config), out/f'step_{step:06d}.pt')
            print(f'CHECKPOINT step={step}', flush=True)
    json_write(out/'status.json', dict(state='complete', step=s['steps'], elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
