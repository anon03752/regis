"""The heart rule: the REGIS update with the two constraints a tissue needs.

Same update as `models/nca.py` -- fixed identity/Sobel perception shared across
channels, a per-cell 3-layer MLP with a zero-initialised head, a clipped
increment, a per-cell Bernoulli firing mask, an alive mask from a 3x3
neighbourhood maximum. It is a separate file for one reason of substance and
two of bookkeeping.

THE SUBSTANTIVE ONE -- where the alive mask acts. `models/nca.py` offers both
placements; this domain needs the one that gates the UPDATE:

    z_{t+1} = z_t + A_t * M_t * eta * clip(dx)         here, and
    z_{t+1} = A_t * (z_t + M_t * eta * clip(dx))       cycling-MNIST.

The second form clears every channel of a cell outside an active
neighbourhood. A wounded bin is *by definition* not alive -- the alive channel
is `mask * (damage < 0.1)` -- so under that form the wound would have its own
damage channel zeroed on the first update and the injury would simply vanish.
Gating the increment instead freezes dead cells: the wound holds its value and
closes from the living rim inward, which is the process the chapter is about.
The Ising rule sets A = 1 and does not care.

THE BOOKKEEPING ONES, neither of which changes the function computed:
  - the MLP is `nn.Linear` on a channels-last permute, not 1x1 convolutions;
  - perception is blocked `[z, dz/dx, dz/dy]`, not interleaved per channel.
Keeping both spellings is what lets the checkpoints of record load unchanged.

On top of that the rule carries two clamps that state what heart tissue can
do, both applied after the increment (appendix C):
  - damage is non-increasing, d_{t+1} <= d_t, and bounded by the bin capacity.
    A wound is something the model can close, never deepen; a user-painted
    wound raises the ceiling for those bins and the model erodes from there.
  - the alive channel stays in [0, 1], the range the real field takes. It is
    fed to the critic, and "alive > 1" would be a free separating feature.

State layout, 28 channels: 0..19 the annotated cell types, 20 damage,
21 alive, 22..27 hidden (zero-seeded, written only by the model). The critic
additionally sees a damage-reference layer the rule never does; that is the
trainer's business, not this file's.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_VISIBLE = 21          # 20 cell types + damage: what the critic scores
DAMAGE_IDX = 20         # inside the visible block
ALIVE_IDX = 21          # the alive channel; hidden channels follow it
DAMAGE_CEILING = 4.0    # a 48px bin is four 96px bins summed, so capacity is 4
N_STAGES = 8            # uninjured, 6 hpa, 12 hpa, 1 dpa, 3 dpa, 7 dpa, 14, 28


class GradientSensor(nn.Module):
    """Perception: [identity, Sobel-x, Sobel-y] per channel, kernels shared.

    Depthwise and FIXED -- registered as parameters with `requires_grad=False`
    rather than buffers, which is how the checkpoints of record store them
    (and why `sum(p.numel())` reports 504 more than the trainable count).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_n = channels
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sy = sx.T.clone()
        self.sobel_x = nn.Parameter(sx[None, None].repeat(channels, 1, 1, 1), requires_grad=False)
        self.sobel_y = nn.Parameter(sy[None, None].repeat(channels, 1, 1, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding="same", groups=self.channel_n)
        gy = F.conv2d(x, self.sobel_y, padding="same", groups=self.channel_n)
        return torch.cat([x, gx, gy], dim=1)            # (B, 3C, H, W)


class HeartNCA(nn.Module):
    """The rule. `class_dim > 0` is the time-conditioned ablation arm, which is
    handed the stage the leg lands on as a spatially broadcast one-hot -- the
    same variable the critic is conditioned on -- and so no longer has to time
    itself from the hidden channels."""

    def __init__(self, n_channels: int = 28, hidden: int = 128, fire_rate: float = 0.5,
                 step_size: float = 0.1, dx_clip: float = 10.0,
                 alive_threshold: float = 0.05, noise_channels: int = 3,
                 n_visible: int = N_VISIBLE, alive_channel_idx: int = ALIVE_IDX,
                 damage_channel_idx: int = DAMAGE_IDX,
                 damage_ceiling: float = DAMAGE_CEILING, class_dim: int = 0) -> None:
        super().__init__()
        self.n_channels, self.hidden = n_channels, hidden
        self.fire_rate, self.step_size, self.dx_clip = fire_rate, step_size, dx_clip
        self.alive_threshold, self.noise_channels = alive_threshold, noise_channels
        self.n_visible = n_visible
        self.alive_channel_idx, self.damage_channel_idx = alive_channel_idx, damage_channel_idx
        self.damage_ceiling, self.class_dim = float(damage_ceiling), int(class_dim)

        self.sensor = GradientSensor(n_channels)
        self.fc1 = nn.Linear(3 * n_channels + noise_channels + self.class_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_channels)
        nn.init.zeros_(self.fc3.weight)          # start as the identity map
        nn.init.zeros_(self.fc3.bias)

    # ---- sub-steps ---------------------------------------------------------

    def alive(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) 0/1 mask: the alive channel exceeds threshold somewhere
        in the 3x3 neighbourhood. Read from the PRE-step state."""
        a = x[:, self.alive_channel_idx:self.alive_channel_idx + 1]
        a = F.max_pool2d(a, kernel_size=3, stride=1, padding=1)
        return (a > self.alive_threshold).to(x.dtype)

    def _dx(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """The raw increment, before clipping, firing and gating. Split out so
        the non-local control replaces only this (see control.py)."""
        b, _, h, w = x.shape
        f = self.sensor(x)
        if self.noise_channels > 0:
            f = torch.cat([f, torch.randn(b, self.noise_channels, h, w,
                                          device=x.device, dtype=x.dtype)], dim=1)
        f = f.permute(0, 2, 3, 1)                       # (B, H, W, F): MLP per cell
        if self.class_dim > 0:
            if y is None:
                raise ValueError("class_dim > 0 but no stage label passed to step()")
            oh = F.one_hot(y.reshape(-1).long(), self.class_dim).to(f.dtype)
            f = torch.cat([f, oh[:, None, None, :].expand(-1, h, w, -1)], dim=-1)
        f = F.relu(self.fc1(f))
        f = F.relu(self.fc2(f))
        return self.fc3(f).permute(0, 3, 1, 2)          # (B, C, H, W)

    # ---- the update --------------------------------------------------------

    def step(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        a = self.alive(x)
        dx = torch.clamp(self._dx(x, y), -self.dx_clip, self.dx_clip)
        if self.fire_rate < 1.0:
            fire = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                               device=x.device) < self.fire_rate).to(dx.dtype)
            dx = dx * fire
        dx = dx * a                                     # gate the UPDATE, not the state
        new_x = x + dx * self.step_size

        i = self.damage_channel_idx                     # damage can only close
        capped = torch.minimum(new_x[:, i:i + 1],
                               x[:, i:i + 1].clamp(0.0, self.damage_ceiling))
        new_x = torch.cat([new_x[:, :i], capped.clamp(0.0, self.damage_ceiling),
                           new_x[:, i + 1:]], dim=1)

        j = self.alive_channel_idx                      # alive stays in [0, 1]
        return torch.cat([new_x[:, :j], new_x[:, j:j + 1].clamp(0.0, 1.0),
                          new_x[:, j + 1:]], dim=1)

    def forward(self, x: torch.Tensor, n_steps: int = 1,
                y: torch.Tensor | None = None) -> torch.Tensor:
        for _ in range(n_steps):
            x = self.step(x, y=y)
        return x

    # ---- record and reload -------------------------------------------------

    @property
    def hparams(self) -> dict:
        return dict(n_channels=self.n_channels, hidden=self.hidden,
                    fire_rate=self.fire_rate, step_size=self.step_size,
                    dx_clip=self.dx_clip, alive_threshold=self.alive_threshold,
                    noise_channels=self.noise_channels, n_visible=self.n_visible,
                    alive_channel_idx=self.alive_channel_idx,
                    damage_channel_idx=self.damage_channel_idx,
                    damage_ceiling=self.damage_ceiling, class_dim=self.class_dim)

    @classmethod
    def from_checkpoint(cls, state_dict: dict, hparams: dict):
        m = cls(**hparams)
        m.load_state_dict(state_dict)
        return m
