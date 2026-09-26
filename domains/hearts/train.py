"""Train REGIS, or one of its two ablation arms, on the zebrafish cohort.

The target dynamics: an uninjured heart stays uninjured until it is cut, then
passes through the seven observed stages in order and returns to looking
uninjured, where it stays. Unlike the digit cycle of the MNIST chapter, the
uninjured state is a FIXED POINT rather than one more step round a loop -- the
only way to leave it is a new injury.

A pool holds 2048 slots, each a state and the stage it sits at. One iteration
samples 32 of them and advances each by one segment of `--steps-per-iter`
updates, so a slot traverses the arc over successive iterations rather than
within one, and the critic only ever compares a segment's endpoint against real
sections of the stage that endpoint should have reached.

  wound     a stage-0 slot carries a pending footprint and is cut with
            probability `--p-wound` per iteration. On heads it is stamped and
            the slot is asked to become 6 hpa; on tails it holds, with an L1
            self-target so holding is something the rule learns rather than
            something the schedule enforces.
  arc       stages 1..6 advance by one; stage 7 (28 dpa) returns to uninjured.
  critic    class-projection on the stage the segment LANDS on, reading the 21
            visible channels, the alive channel, and the damage reference the
            generator never sees.
  loss      R3GAN relativistic pairing with zero-centred penalties on both
            inputs, plus the six auxiliary terms of appendix C.
  eval      an exponential moving average of the rule (`ema`).

The three arms differ in `--g-arch` alone:

  python -m domains.hearts.train --g-arch regis   --seed 0 --run-name hearts_regis_seed0
  python -m domains.hearts.train --g-arch control --seed 0 --run-name hearts_control_nonlocal_seed0
  python -m domains.hearts.train --g-arch time    --seed 0 --run-name hearts_time_seed0

and `--holdout-heart T1_S1` drops one animal from training entirely.
The defaults are the configuration of the paper's runs. Writes
<workspace>/runs/<run-name>/{config.json, events.jsonl, ckpts/ckpt_NNNNNN.pt}.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from workspace import WORK
from models.train.logger import Logger
from models.train.r3gan import Discriminator, relativistic, zero_centered_gp
from domains.hearts import cohort as C
from domains.hearts import wounds as W
from domains.hearts.rule import HeartNCA, N_VISIBLE, DAMAGE_IDX, ALIVE_IDX
from domains.hearts.control import HeartControl


class Pool:
    """Replay slots. A slot is a state, the stage it sits at, its own per-bin
    capacity target, the wound site the critic is shown, and the footprint it
    will be cut with when its coin comes up."""

    def __init__(self, size, channels, h, w, device):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=device, dtype=dt)
        self.states = z(size, channels, h, w)
        self.masks = z(size, 1, h, w)
        self.bin_cap = z(size, 1, h, w)
        self.ref = z(size, 1, h, w)
        self.damage = z(size, 1, h, w, dt=torch.uint8)
        self.tp = z(size, dt=torch.long)
        self.age = z(size, dt=torch.long)
        self.pending = z(size, dt=torch.bool)
        self.size = size

    def sample(self, n):
        i = torch.randperm(self.size, device=self.states.device)[:n]
        return (i, self.states[i], self.masks[i], self.tp[i], self.age[i],
                self.bin_cap[i], self.ref[i], self.damage[i], self.pending[i])

    def write(self, i, states, masks, tp, damage, pending):
        self.states[i] = states.detach()
        self.masks[i] = masks.detach()
        self.tp[i] = tp
        self.age[i] = 0
        self.ref[i] = 0.0                    # a reseeded slot is a fresh animal
        self.bin_cap[i] = C.bin_capacity(states[:, :N_VISIBLE]).detach()
        self.damage[i] = damage.to(self.damage.dtype)
        self.pending[i] = pending

    def commit(self, i, states, tp, age, damage, ref, pending):
        self.states[i] = states.detach()
        self.tp[i] = tp
        self.age[i] = age
        self.damage[i] = damage
        self.ref[i] = ref.detach()
        self.pending[i] = pending

    def state_dict(self):
        return {k: getattr(self, k).cpu() for k in
                ("states", "masks", "bin_cap", "ref", "damage", "tp", "age", "pending")}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            setattr(self, k, v.to(self.states.device))


# ---- the auxiliary losses of appendix C ------------------------------------

def _edges(x):
    """L1 spatial gradient of a (B,1,H,W) field, zero-padded back to size."""
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    return dx.abs() + dy.abs()


def silhouette_loss(fake, real):
    """Hold the generated footprint to the section's. Heart size is a large
    technical bias across stages -- the uninjured cohort is some 50% larger
    than the 12 hpa one -- so the adversarial signal alone rewards shrinking or
    spreading the tissue towards whichever size is easiest to pass off. The
    edge term is weighted 10x because an outline is a thin target."""
    inter = (fake * real).sum(dim=(1, 2, 3))
    union = (fake + real - fake * real).sum(dim=(1, 2, 3))
    iou = (1.0 - (inter + 1e-6) / (union + 1e-6)).mean()
    return (F.smooth_l1_loss(fake, real)
            + 10.0 * F.smooth_l1_loss(_edges(fake), _edges(real)) + 2.0 * iou)


def per_heart(x, mask):
    """Mean of a per-pixel penalty over each section's own bins, then over the
    batch -- so a large heart does not outweigh a small one."""
    return (x.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)).mean()


# ---- the run ---------------------------------------------------------------

def build_rule(a, device):
    common = dict(hidden=a.hidden, fire_rate=a.fire_rate, step_size=a.step_size,
                  dx_clip=a.dx_clamp, alive_threshold=a.alive_threshold,
                  noise_channels=a.noise_channels, damage_ceiling=a.damage_ceiling,
                  n_channels=N_VISIBLE + 1 + a.n_hidden,
                  class_dim=C.N_STAGES if a.g_arch == "time" else 0)
    if a.g_arch == "control":
        return HeartControl(conv_base=a.conv_base,
                            conv_dilations=a.conv_dilations,
                            **common).to(device)
    return HeartNCA(**common).to(device)


# The configuration of every run of record. Only the arm, the seed, the run
# name, the length and the held-out heart are flags; config.json and every
# checkpoint record all of it.
RECORD = dict(
    hidden=128, n_hidden=6, noise_channels=3,
    fire_rate=0.5,
    alive_threshold=0.05,             # on a 3x3 neighbourhood max of the alive channel
    dx_clamp=10.0, step_size=0.1,     # so no channel moves by more than 1.0 per update
    damage_ceiling=4.0,               # a 48px bin is four 96px bins summed
    conv_base=28, conv_dilations=(1, 2, 5, 9),        # control only: 35-bin field
    d_base_channels=64, d_stages=4, d_blocks=2, d_cardinality=4, d_expansion=2,
    d_embed_dim=32,
    batch_size=32, lr_g=2e-4, lr_d=2e-4, beta1=0.0, beta2=0.99,
    gamma=1.0,                        # gradient-penalty weight
    grad_clip=1.0, ema_decay=0.999, ema_warmup=1000,
    pool_size=2048,
    steps_per_iter=15,                # updates per segment; a slot crosses the arc over many
    p_wound=0.3,                      # chance a stage-0 slot is cut this iteration
    reinject_prob=1.0,                # chance a slot closing the loop re-arms its wound
    proc_wound_sigma=0.5,
    max_iter_start=5, max_iter_end=20, max_iter_ramp_until=80_000,
    reseed_weights_early=(0.60, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.05),
    reseed_weights_late=(0.45, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06),
    reseed_ramp_iters=80_000,
    # the auxiliary losses of appendix C
    lam_shape=1.0, silhouette_saturation=0.05,
    lam_capacity=0.4,
    lam_heal=10.0,
    heal_stage_weights=(0.0, 0.0, 0.0, 0.2, 0.4, 0.7, 1.0, 1.0),   # not normalised
    lam_zero=5.0, zero_tol=1e-6,
    lam_homeo=1.0,
    norm_weight=1.0, r_max=1.0,       # hidden-norm hinge, and the radius it acts above
    log_every=50, ckpt_every=10_000,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train REGIS or an ablation arm on the zebrafish cohort.")
    p.add_argument("--g-arch", choices=["regis", "control", "time"], default="regis",
                   help="regis: the local, time-free rule. control: a dilated conv stack, "
                        "35-bin receptive field per update. time: the local rule handed the "
                        "stage label, so it can read a clock instead of building one")
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=100_000,
                   help="training iterations (shorten for a smoke run)")
    p.add_argument("--holdout-heart", action="append", default=[],
                   help="isolate id (T#_S#); every section of that fish is dropped from "
                        "both the critic's banks and the seed pool")
    p.add_argument("--smoke-test", action="store_true", help="tiny run for tests/test_smoke_hearts.py")
    a = argparse.Namespace(**RECORD, **vars(p.parse_args()))
    if a.smoke_test:
        a.iterations, a.batch_size, a.pool_size = 3, 4, 16
        a.steps_per_iter, a.log_every, a.ckpt_every, a.ema_warmup = 3, 1, 3, 2
    return a


def _weights(spec, n, device):
    w = torch.tensor(spec, dtype=torch.float32, device=device)
    assert w.numel() == n and (w >= 0).all() and w.sum() > 0, f"bad weights {spec!r}"
    return w / w.sum()


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = WORK / "runs" / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(a), indent=2, default=str))

    # ---- the cohort ----
    seeds, seed_ids, real_banks, ref_banks = C.load_cohort(C.PATH, device, a.holdout_heart)
    H, Wd = seeds[0].shape[-2:]
    n_total = N_VISIBLE + 1 + a.n_hidden
    zero_mask = C.zero_mask(real_banks, N_VISIBLE, device, a.zero_tol)
    real_banks = {tp: (C.expand(s, m, a.n_hidden), m) for tp, (s, m) in real_banks.items()}

    # only sections with a painted apex zone can seed a wounded trajectory
    usable, zone_grids, _tips = W.load_zones(seed_ids, H, Wd, device)
    seed_states = C.expand(*seeds, a.n_hidden)[usable]
    seed_masks = seeds[1][usable]
    shapes = W.load_wound_shapes(H, Wd)

    # ---- the models ----
    rule = build_rule(a, device)
    n = a.d_stages + 1
    disc = Discriminator(widths=[a.d_base_channels] * n, cardinalities=[a.d_cardinality] * n,
                         blocks_per_stage=[a.d_blocks] * n, expansion=a.d_expansion,
                         num_classes=C.N_STAGES, embed_dim=a.d_embed_dim,
                         in_channels=N_VISIBLE + 2, basis="avgpool").to(device)
    aug = C.Augment().to(device)
    print(f"{a.run_name} on {device}: {a.g_arch}, rule {sum(p.numel() for p in rule.parameters()):,} "
          f"parameters, critic {sum(p.numel() for p in disc.parameters()):,}; "
          f"{len(usable)} seed sections, {len(shapes)} wound footprints", flush=True)

    opt_g = torch.optim.Adam(rule.parameters(), lr=a.lr_g, betas=(a.beta1, a.beta2))
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.lr_d, betas=(a.beta1, a.beta2))

    def ema_avg(avg, p, num):
        decay = min(a.ema_decay, (1 + num) / (max(1, a.ema_warmup) + num))
        return decay * avg + (1.0 - decay) * p
    ema = torch.optim.swa_utils.AveragedModel(rule, avg_fn=ema_avg)

    # ---- the pool ----
    pool = Pool(a.pool_size, n_total, H, Wd, device)
    early = _weights(a.reseed_weights_early, C.N_STAGES, device)
    late = _weights(a.reseed_weights_late, C.N_STAGES, device)
    heal_w = torch.tensor(a.heal_stage_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(a.seed + 10_000)      # the wound stream, seeded apart

    def reseed(idx, weights):
        """Refill slots, drawing each slot's stage from `weights`. A stage-0
        slot also gets a footprint, stored but not yet cut."""
        if idx.numel() == 0:
            return
        k = idx.shape[0]
        tp = torch.multinomial(weights, num_samples=k, replacement=True)
        states = torch.empty(k, n_total, H, Wd, device=device)
        masks = torch.empty(k, 1, H, Wd, device=device)
        damage = torch.zeros(k, 1, H, Wd, dtype=torch.uint8, device=device)
        pending = torch.zeros(k, dtype=torch.bool, device=device)
        for s in range(k):
            t = int(tp[s])
            if t == 0:
                j = int(rng.integers(seed_states.shape[0]))
                states[s], masks[s] = seed_states[j], seed_masks[j]
                d = W.sample_shaped_wound(zone_grids[j, 0].cpu().numpy(),
                                          seed_masks[j, 0].cpu().numpy() > 0, rng, shapes)
                damage[s, 0] = torch.from_numpy(d).to(device)
                pending[s] = True
            else:
                bank, bank_m = real_banks[C.TP_ORDER[t]]
                j = int(rng.integers(bank.shape[0]))
                states[s], masks[s] = bank[j], bank_m[j]
        with torch.no_grad():                # geometry stays aligned: one draw
            for s in range(0, k, 64):
                e = min(s + 64, k)
                states[s:e], masks[s:e], damage_f = aug(
                    states[s:e], masks[s:e], damage=damage[s:e].float())
                damage[s:e] = damage_f.to(damage.dtype)
        pool.write(idx, states, masks, tp, damage, pending)

    with torch.no_grad():
        reseed(torch.arange(a.pool_size, device=device), early)
    logger = Logger(run_dir=run_dir)

    def save(step):
        out = run_dir / "ckpts"; out.mkdir(parents=True, exist_ok=True)
        torch.save({"step": step, "rule": rule.state_dict(), "disc": disc.state_dict(),
                    "ema": ema.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "pool": pool.state_dict(),
                    "args": vars(a), "hparams": rule.hparams,
                    "channels": C.CELL_TYPES + [C.DAMAGE_CHANNEL]},
                   out / f"ckpt_{step:06d}.pt")
        print(f"checkpoint at step {step}", flush=True)

    # ---- the loop ----
    for it in range(a.iterations):
        frac = 1.0 if a.reseed_ramp_iters <= 0 else min(1.0, max(0.0, it / a.reseed_ramp_iters))
        weights = ((1 - frac) * early + frac * late).clamp_min(0.0)

        idx, state, mask, tp, age, bin_cap, ref, damage, pending = pool.sample(a.batch_size)

        # cut the stage-0 slots whose coin comes up
        fired = (tp == 0) & pending & (torch.rand(a.batch_size, device=device) < a.p_wound)
        if fired.any():
            f = fired.view(-1, 1, 1, 1)
            state = W.stamp(state, mask, (damage.bool() & f).to(state.dtype),
                            n_visible=N_VISIBLE, damage_idx=DAMAGE_IDX, alive_idx=ALIVE_IDX,
                            sigma=a.proc_wound_sigma, alive_threshold=C.ALIVE_DAMAGE_THR)
            # the site the critic is shown is the STAMPED FIELD, not the 0/1
            # footprint: the real reference is densified and sum-pooled and
            # reaches 4.0, so a binary fake would be separable on scale alone,
            # on a channel the generator cannot touch.
            ref = torch.maximum(ref, state[:, DAMAGE_IDX:DAMAGE_IDX + 1] * f.to(state.dtype))

        # stage 0 + cut -> 1; 1..6 -> +1; 7 -> 0 (loop closure); else hold
        new_tp = tp.clone()
        new_tp[fired] = 1
        mid = (tp >= 1) & (tp <= C.N_STAGES - 2)
        new_tp[mid] = tp[mid] + 1
        terminal = tp == (C.N_STAGES - 1)
        new_tp[terminal] = 0
        hold = (tp == 0) & ~fired
        # the site is cleared across loop closure: the real uninjured bank has
        # an all-zero reference, so a looped slot that kept its site would hand
        # the critic a free separation
        ref = ref * (new_tp != 0).view(-1, 1, 1, 1).to(ref.dtype)
        cue = new_tp if a.g_arch == "time" else None

        # a real section of the stage this segment lands on
        real = torch.empty_like(state)
        real_m = torch.empty_like(mask)
        real_ref = torch.empty_like(ref)
        for i in range(a.batch_size):
            tp_name = C.TP_ORDER[int(new_tp[i])]
            bank, bank_m = real_banks[tp_name]
            j = int(rng.integers(bank.shape[0]))
            real[i], real_m[i], real_ref[i] = bank[j], bank_m[j], ref_banks[tp_name][j]
        with torch.no_grad():
            real, real_m, real_ref = aug(real, real_m, extra=real_ref)

        def for_critic(s, r):
            return torch.cat([s[:, :N_VISIBLE + 1], r], dim=1)
        real_in = for_critic(real, real_ref)
        source = state.detach().clone()

        # ---- critic step ----
        for p in rule.parameters():
            p.requires_grad_(False)
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = rule(state, n_steps=a.steps_per_iter, y=cue)
        r_in = real_in.detach().requires_grad_(True)
        f_in = for_critic(fake, ref).detach().requires_grad_(True)
        d_real, d_fake = disc(r_in, new_tp), disc(f_in, new_tp)
        adv_d = relativistic(d_real, d_fake)
        reg = 0.5 * a.gamma * (zero_centered_gp(r_in, d_real.float()).mean()
                               + zero_centered_gp(f_in, d_fake.float()).mean())
        (adv_d + reg).backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), a.grad_clip)
        opt_d.step()

        # ---- rule step: backprop through the whole segment ----
        for p in rule.parameters():
            p.requires_grad_(True)
        for p in disc.parameters():
            p.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        new_state = rule(state, n_steps=a.steps_per_iter, y=cue)
        g_adv = relativistic(disc(for_critic(new_state, ref), new_tp),
                             disc(real_in.detach(), new_tp))

        m = mask.detach()
        alive = new_state[:, ALIVE_IDX:ALIVE_IDX + 1]
        dmg = new_state[:, DAMAGE_IDX:DAMAGE_IDX + 1]
        sat = a.silhouette_saturation
        # a bin counts towards the silhouette once EITHER alive or damage
        # crosses the same threshold the alive gate uses, so damage decaying
        # during healing does not momentarily punch a hole in the outline
        g_shape = silhouette_loss(torch.maximum((alive / sat).clamp(0, 1),
                                                (dmg / sat).clamp(0, 1)), m)
        # the wound interior is not alive, so the gate freezes it and no
        # gradient reaches it: this drives the alive field back into the hole
        # and re-opens the path, loudly late and silently early
        g_heal = per_heart((m - alive).clamp_min(0) * heal_w[new_tp].view(-1, 1, 1, 1), m)
        # channels the cohort has at exactly zero at this stage
        zm = zero_mask[new_tp]
        g_zero = (((new_state[:, :N_VISIBLE].abs() * zm.view(-1, N_VISIBLE, 1, 1) * m)
                   .sum(dim=(1, 2, 3))
                   / (m.sum(dim=(1, 2, 3)).clamp_min(1.0) * zm.sum(dim=1).clamp_min(1.0)))
                  .mean())
        # each bin still sums to what it summed when the slot was reseeded
        g_cap = per_heart((new_state[:, :N_VISIBLE].sum(1, keepdim=True) - bin_cap).abs() * m, m)
        # holding is learned, not scheduled
        if hold.any():
            hb = hold.view(-1, 1, 1, 1)
            g_homeo = ((new_state[:, :N_VISIBLE + 1] - source[:, :N_VISIBLE + 1]).abs() * hb
                       ).sum() / (hb.sum() * (N_VISIBLE + 1) * H * Wd + 1e-8)
        else:
            g_homeo = torch.zeros((), device=device)
        # confine the hidden channels: free dynamics inside the ball, a wall at
        # the edge, so drift cannot accumulate off the visible channels
        g_hidden = (F.relu(new_state[:, N_VISIBLE + 1:].float().norm(dim=1) - a.r_max)
                    .pow(2).mean() if a.n_hidden > 0 else torch.zeros((), device=device))

        g_loss = (g_adv + a.lam_shape * g_shape + a.lam_heal * g_heal
                  + a.lam_zero * g_zero + a.lam_capacity * g_cap
                  + a.lam_homeo * g_homeo + a.norm_weight * g_hidden)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(rule.parameters(), a.grad_clip)
        opt_g.step()
        ema.update_parameters(rule)
        for p in disc.parameters():
            p.requires_grad_(True)

        # ---- the segment's endpoint returns to the pool ----
        new_pending = pending & ~fired & (new_tp == 0)
        if a.reinject_prob > 0:
            new_pending |= (terminal & (damage.sum(dim=(1, 2, 3)) > 0)
                            & (torch.rand(a.batch_size, device=device) < a.reinject_prob))
        pool.commit(idx, new_state, new_tp, age + 1, damage, ref, new_pending)

        cap = (a.max_iter_end if it >= a.max_iter_ramp_until else
               max(a.max_iter_start, min(a.max_iter_end, round(
                   a.max_iter_start + (a.max_iter_end - a.max_iter_start)
                   * it / a.max_iter_ramp_until))))
        done = (age + 1) >= cap
        if done.any():
            with torch.no_grad():
                reseed(idx[done], weights)

        if it % a.log_every == 0:
            logger.scalars({"loss/D_adv": adv_d.item(), "loss/D_reg": reg.item(),
                            "loss/G_adv": g_adv.item(), "loss/shape": g_shape.item(),
                            "loss/heal": g_heal.item(), "loss/zero": g_zero.item(),
                            "loss/capacity": g_cap.item(),
                            "loss/homeo": g_homeo.detach().item(),
                            "loss/hidden": g_hidden.detach().item(),
                            "pool/wounds_fired": int(fired.sum()),
                            "pool/max_age": cap}, step=it)
        if it % (20 * a.log_every) == 0:
            print(f"step {it:>6d}  D {adv_d.item():.3f}  G {g_adv.item():.3f}  "
                  f"shape {g_shape.item():.3f}  heal {g_heal.item():.4f}  "
                  f"zero {g_zero.item():.4f}  cap {g_cap.item():.3f}", flush=True)
        if (it + 1) % a.ckpt_every == 0:
            save(it + 1)
    if a.iterations % a.ckpt_every != 0:
        save(a.iterations)
    logger.close(status="completed")


if __name__ == "__main__":
    main()
