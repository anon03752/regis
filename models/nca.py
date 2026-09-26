"""Neural cellular automaton and shared update operations for the non-local control.

NCA applies depthwise perception filters and a per-cell MLP to predict state
increments. Perception uses fixed identity/Sobel filters or learned kernels.
Optional constraints include spin-flip antisymmetry and 180-degree kernel
symmetry. Updates use clipping, Bernoulli firing or a two-part checkerboard
sweep, and an optional alive mask.

States have shape (B, C, H, W). Ising uses one visible channel, periodic
boundaries, and no alive mask. The MNIST wrapper adds hidden channels, a fixed
positional feature, zero padding, and an alive mask; its visible channel is last.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# alive_mask wipes every cell, hidden channels included, whose 3x3
# neighbourhood holds no visible value above this in the pre-step state
ALIVE_THRESHOLD = 0.1


def alive(state: torch.Tensor, v: int) -> torch.Tensor:
    """(B, 1, H, W) 0/1: does channel v exceed ALIVE_THRESHOLD anywhere in the
    cell's 3x3 neighbourhood?"""
    return (F.max_pool2d(state[:, v:v + 1], kernel_size=3, stride=1, padding=1)
            > ALIVE_THRESHOLD).float()


class UpdateRule(nn.Module):
    """Shared update operations around a subclass's `_net(state, z, noise)`.

    Handles clipping, the update penalty, spin-flip symmetry, firing masks,
    alive masking, and visible-channel readout."""

    # L_upd (appendix B): the Ising trainer switches it on with start_hinge(); each
    # network call then adds relu(|f| - 1)^2 over its raw branch
    hinge_sum = hinge_calls = None

    def __init__(self, *, fire_mode, fire_rate, update_clip, z2_sym, alive_mask, visible_index):
        super().__init__()
        self.fire_mode, self.fire_rate, self.update_clip = fire_mode, fire_rate, update_clip
        self.z2_sym, self.alive_mask, self.visible_index = z2_sym, alive_mask, visible_index

    def start_hinge(self, device):
        self.hinge_sum = torch.zeros((), device=device)
        self.hinge_calls = torch.zeros((), device=device)

    def _dx(self, state, z, noise):
        dx = self._net(state, z, noise)
        if self.hinge_sum is not None:
            # spins are 0/1 here, so a flip is a change of exactly 1 and only
            # |f| above 1 is penalised
            self.hinge_sum = self.hinge_sum + torch.relu(dx.abs() - 1.0).square().mean()
            self.hinge_calls = self.hinge_calls + 1
        return dx.clamp(-self.update_clip, self.update_clip)

    def _upd(self, state, z=None, noise=None):
        """Compute the increment, optionally antisymmetrised for 0/1 spins.

        With z2_sym, Delta = (f(s, xi) - f(1 - s, -xi)) / 2. Each branch
        contributes to the hinge penalty and is clipped before averaging."""
        if not self.z2_sym:
            return self._dx(state, z, noise)
        # Preserve graph construction order: it affects gradient accumulation
        # and floating-point rounding.
        flipped = 1.0 - state
        return 0.5 * (self._dx(state, z, noise) - self._dx(flipped, z, -noise))

    def masked_update(self, state, mask, z=None, noise=None):
        """s + m * Delta(s, xi): the cells where the mask is 1 take the update."""
        return state + mask * self._upd(state, z, noise)

    def step(self, state: torch.Tensor, z: torch.Tensor | None = None,
             noise: torch.Tensor | None = None,
             noise2: torch.Tensor | None = None) -> torch.Tensor:
        """One update. state: the NCA state z_t, (B, C, H, W). z: MNIST's
        persistent noise xi, (B, z_dim), fixed along a trajectory and broadcast
        to every cell. noise, noise2: Ising's per-cell noise xi,
        (B, noise_channels, H, W), one draw per half-step."""
        pre = state
        if self.fire_mode == "checker2":
            # a learned checkerboard sweep: sublattice A, then sublattice B on
            # the updated state, fresh noise each half. 4-neighbours never
            # co-update, so one step is one sweep.
            yy = torch.arange(state.shape[2], device=state.device)
            xx = torch.arange(state.shape[3], device=state.device)
            chk = ((yy[:, None] + xx[None, :]) % 2).to(state.dtype)
            state = self.masked_update(state, chk, z, noise)
            state = self.masked_update(state, 1.0 - chk, z, noise2)
        else:
            # one draw per cell, shared by its channels
            mask = (torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device)
                    < self.fire_rate).to(state.dtype)
            state = self.masked_update(state, mask, z, noise)
        if self.alive_mask:
            state = state * alive(pre, self.visible_index)
        return state

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W): the visible channel, raw; the critic handles its range."""
        v = self.visible_index
        return state[:, v:v + 1]


def positional_feature(h: int, w: int, device, dtype) -> torch.Tensor:
    """(1, 1, h, w): the fixed Gaussian exp(-(x^2 + y^2)/0.5) on coordinates in
    [-1, 1]. It keeps the digit from drifting over hundreds of transitions."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device, dtype=dtype),
                            torch.linspace(-1, 1, w, device=device, dtype=dtype), indexing="ij")
    return torch.exp(-(xx ** 2 + yy ** 2) / 0.5)[None, None]


def _perception_kernel(channels: int) -> torch.Tensor:
    """Grouped conv kernel: for each input channel emit [identity, sobel_x, sobel_y].
    Output channel ordering: c0_id, c0_sx, c0_sy, c1_id, c1_sx, c1_sy, ... .
    Shape: (channels * 3, 1, 3, 3)."""
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
    sy = sx.t().clone()
    ident = torch.zeros((3, 3))
    ident[1, 1] = 1.0
    k = torch.stack([ident, sx, sy])       # (3, 3, 3)
    k = k.unsqueeze(1)                     # (3, 1, 3, 3)
    k = k.repeat(channels, 1, 1, 1)        # (C*3, 1, 3, 3)
    return k


class NCA(UpdateRule):
    def __init__(
        self, *,
        channels: int,
        mlp_width: int,
        fire_rate: float,
        update_clip: float,
        z_dim: int = 0,
        alive_mask: bool = False,
        padding_mode: str = "circular",
        parity_even: bool = False,       # spatial kernels symmetric under 180-degree rotation
        z2_sym: bool = False,            # update antisymmetric under the spin flip (eq. ising_z2)
        noise_channels: int = 0,         # per-cell noise channels, drawn by the caller for every update
        perception_mode: str,            # "sobel": fixed identity/Sobel-x/Sobel-y; "trainable": learned from them
        perception_ksize: int = 3,       # trainable only: kernel side (3, 5, 7)
        fire_mode: str = "bernoulli",    # iid at fire_rate, or "checker2": two sublattice half-steps
        visible_index: int = 0,          # the visible channel (Ising 0, MNIST last)
        positional: bool = False,        # add the fixed positional feature (MNIST)
    ) -> None:
        super().__init__(fire_mode=fire_mode, fire_rate=fire_rate, update_clip=update_clip,
                         z2_sym=z2_sym, alive_mask=alive_mask, visible_index=visible_index)
        self.z_dim = z_dim
        self.positional = positional
        self.padding_mode = padding_mode
        self.parity_even = parity_even
        self.perception_mode = perception_mode

        self._pad = perception_ksize // 2          # symmetric pad so output keeps H,W
        if perception_mode == "trainable":
            # learned depthwise perception C -> 3C, initialised to identity/Sobel-x/Sobel-y (a KxK
            # kernel holds the 3x3 in its centre). Under parity_even only the 180-degree-symmetric
            # part (W + W rotated)/2 acts. The Sobel kernels are odd, so the Ising rule starts from
            # [identity, 0, 0]; the stored odd part gets no gradient and only shrinks under weight decay.
            self.perception = nn.Conv2d(channels, 3 * channels, kernel_size=perception_ksize,
                                        padding=0, groups=channels, bias=False)
            with torch.no_grad():
                w = torch.zeros_like(self.perception.weight)        # (3C, 1, K, K)
                c = self._pad
                w[:, :, c - 1:c + 2, c - 1:c + 2] = _perception_kernel(channels)
                self.perception.weight.copy_(w)
        else:
            self.register_buffer("perception_kernel", _perception_kernel(channels))

        in_c = 3 * channels + z_dim + noise_channels + int(positional)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_c, mlp_width, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_width, mlp_width, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_width, channels, kernel_size=1),
        )
        # Start with zero predicted increments; alive masking still applies.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @classmethod
    def from_hparams(cls, hp):
        """REGIS as the Ising trainer builds it, from its recorded hparams."""
        return cls(channels=hp["channels"], mlp_width=hp["mlp_width"],
                   fire_rate=hp["fire_rate"], fire_mode=hp["fire_mode"], update_clip=hp["update_clip"],
                   noise_channels=hp["noise_channels"], perception_mode="trainable",
                   perception_ksize=hp["perception_ksize"], parity_even=hp["parity_even"],
                   z2_sym=hp["z2_sym"], padding_mode="circular")    # the torus

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        w = self.perception.weight if self.perception_mode == "trainable" else self.perception_kernel
        if self.parity_even:
            w = 0.5 * (w + w.flip(-1, -2))      # the 180-degree symmetrisation
        p = self._pad
        return F.conv2d(F.pad(state, (p, p, p, p), mode=self.padding_mode), w, groups=state.shape[1])

    def _net(self, state, z, noise):
        p = self.perceive(state)
        inputs = [p]
        if self.positional:
            inputs.append(positional_feature(*p.shape[-2:], p.device, p.dtype).expand(p.shape[0], -1, -1, -1))
        if z is not None:
            inputs.append(z[:, :, None, None].expand(-1, -1, *state.shape[2:]))
        if noise is not None:
            inputs.append(noise)
        return self.mlp(torch.cat(inputs, 1))
