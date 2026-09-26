"""The one rollout driver: load a run, cut a heart, run the regeneration.

Every measurement in the chapter is this function under a different knockout,
so there is one implementation of the protocol and not one per assay.

    seed        a real uninjured section with a painted apex zone, expanded to
                the rule's state, cut with a standardised capsule
    arc         7 legs of `steps_per_leg` updates, landing on stages 1..7
    settle      `settle` further updates at stage 0 -- training closes the
                terminal leg 28 dpa -> uninjured, so 0 is the label the model
                actually saw past the arc, not a hold at 7
    knockout    a channel index zeroed after EVERY update, the in silico
                analogue of a loss-of-function: the seed is left intact, so the
                question is whether regeneration can proceed without that
                type's response, not what an animal missing it looks like

Scoring is at the settled state rather than at 28 dpa: the arc is rolled to its
end and the model is then given time to relax toward looking uninjured, which
it was trained to do. The question is whether it got home.

`seed_batch` and `roll` are seeded, so a measurement here is reproducible --
which the exported-graph path this replaces was not, having no way to seed its
noise channels or its firing mask.
"""
from __future__ import annotations

import numpy as np
import torch

from domains.hearts import cohort as C
from domains.hearts import wounds as W
from domains.hearts.rule import HeartNCA, N_VISIBLE, DAMAGE_IDX, ALIVE_IDX
from domains.hearts.control import HeartControl

STEPS_PER_LEG = 15
SETTLE_STEPS = 15
N_LEGS = C.N_STAGES - 1


def load_rule(ckpt_path, device="cpu"):
    """-> (rule, args). Always the EMA weights: evaluation reads nothing else.

    Reads runs written by this release and by the research tree, which spell the
    averaged state dict `ema` and `nca_ema` respectively and name the arms
    differently.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    rule = build_rule(a)
    key = "ema" if "ema" in ck else "nca_ema"
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in ck[key].items() if k != "n_averaged"}
    rule.load_state_dict(sd, strict=True)
    return rule.eval().to(device), a


def build_rule(a):
    """The untrained rule a run's settings describe."""
    # `or`, not a get() default: the slim checkpoints store g_arch as an
    # explicit None, which a default would never replace
    arch = a.get("g_arch") or "regis"
    is_control = arch in ("control", "dilated")
    class_dim = C.N_STAGES if (arch == "time" or a.get("g_class_cue")) else 0
    n_hidden = int(a.get("n_hidden", 6))
    common = dict(n_channels=N_VISIBLE + 1 + n_hidden, hidden=a["hidden"],
                  fire_rate=a["fire_rate"], step_size=a["step_size"],
                  dx_clip=a["dx_clamp"], alive_threshold=a["alive_threshold"],
                  noise_channels=a["noise_channels"],
                  damage_ceiling=a.get("damage_ceiling", 4.0), class_dim=class_dim)
    if is_control:
        dil = [int(x) for x in str(a.get("conv_dilations", "1,2,5,9")).split(",")]
        return HeartControl(conv_base=a["conv_base"], conv_dilations=dil, **common)
    return HeartNCA(**common)


def seed_batch(cohort_path, n, rng, device="cpu", n_hidden=6, wound=True,
               sigma=0.5, holdout_hearts=()):
    """-> (state, mask): `n` wounded uninjured hearts, sampled with replacement
    from the sections that carry an apex zone."""
    seeds, seed_ids, _banks, _refs = C.load_cohort(cohort_path, device, holdout_hearts)
    usable, zones, tips = W.load_zones(seed_ids, *seeds[0].shape[-2:], device)
    comp, mask = seeds[0][usable], seeds[1][usable]
    pick = [int(rng.integers(len(usable))) for _ in range(n)]
    state = C.expand(comp[pick], mask[pick], n_hidden)
    mask = mask[pick].contiguous()
    if not wound:
        return state, mask
    scale = state.shape[-1] / W.CANVAS
    dmg = torch.zeros(n, 1, *state.shape[-2:], device=device)
    for i, j in enumerate(pick):
        d = W.sample_capsule_wound(zones[j, 0].cpu().numpy(),
                                   mask[i, 0].cpu().numpy() > 0,
                                   tips[j].cpu().numpy(), rng, scale=scale)
        dmg[i, 0] = torch.from_numpy(d).to(device)
    state = W.stamp(state, mask, dmg, n_visible=N_VISIBLE, damage_idx=DAMAGE_IDX,
                    alive_idx=ALIVE_IDX, sigma=sigma,
                    alive_threshold=C.ALIVE_DAMAGE_THR)
    return state, mask


@torch.no_grad()
def roll(rule, state, knockout=None, ko_mode="no-increase", record=False,
         steps_per_leg=STEPS_PER_LEG, settle=SETTLE_STEPS, seed=0):
    """-> (settled, per_leg) or (settled, per_leg, every_step).

    `per_leg` is the state at the end of each leg, i.e. at each measured stage.
    `record=True` additionally returns the state before the first update and
    after every update of the arc, 1 + 7 * `steps_per_leg` of them, which is what
    the trajectory and cascade curves are made of.

    `knockout` is a channel index (or None) suppressed after every update:

      no-increase   clamped to its pre-step value, so the type can decay but can
                    never be produced or recruited. The closer analogue of a
                    loss-of-function -- block the programme, do not vaporise the
                    cells already there. The perturbation screen uses this.
      zero          forced to 0. A harder intervention; the cascade read-out
                    uses it, because there the question is whether a downstream
                    state can rise at all.
    """
    assert ko_mode in ("no-increase", "zero"), f"unknown ko_mode {ko_mode!r}"
    torch.manual_seed(seed)
    per_leg, every = [], ([state.clone()] if record else [])

    def suppress(x, prev):
        if knockout is None:
            return x
        if ko_mode == "zero":
            x[:, knockout] = 0.0
        else:
            x[:, knockout] = torch.minimum(x[:, knockout], prev[:, knockout])
        return x

    for leg in range(N_LEGS):
        cue = (torch.full((state.shape[0],), leg + 1, dtype=torch.long,
                          device=state.device) if rule.class_dim else None)
        for _ in range(steps_per_leg):
            prev = state
            state = suppress(rule.step(state, y=cue), prev)
            if record:
                every.append(state)
        per_leg.append(state.clone())
    cue = (torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
           if rule.class_dim else None)
    for _ in range(settle):
        prev = state
        state = suppress(rule.step(state, y=cue), prev)
    return (state, per_leg, every) if record else (state, per_leg)


def heart_means(states, mask):
    """(T, n_visible) or (T, N, n_visible): each visible channel's mean over the
    bins of its own heart. The quantity every curve in the chapter is built
    from -- a per-bin density, so a big section does not outweigh a small one."""
    m = (mask > 0).to(states[0].dtype)
    denom = m.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return torch.stack([((s[:, :N_VISIBLE] * m).sum(dim=(2, 3))
                         / denom[:, None]) for s in states])


def damage_over_heart(state, mask):
    """The run of record's own pre-registered criterion: mean damage over the
    heart's bins. It must fall to <= 0.01 by 28 dpa, with no decay supplied --
    recovery has to be something the rule learnt.

    Do NOT substitute a tissue fraction inside the post-wound mask: the wounded
    bins leave that mask by construction, so it reads 1.00 at every step
    whatever the model does. A quantity that cannot fail certifies nothing.
    """
    d = state[:, DAMAGE_IDX:DAMAGE_IDX + 1] * mask
    return (d.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0))
