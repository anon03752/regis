"""The drift network: a time-conditioned U-Net.

diffusers' UNet2DModel in the configuration the MMtSBM authors use for MNIST
(blocks 32/64/128, two layers per block, sinusoidal time embedding). Its
blocks are convolutional; the mid block keeps diffusers' default single
self-attention layer, which has no positional encoding, so `sample_size` is
metadata and a drift trained on 192^2 windows runs at 512^2. The import is
local to build_drift, so this module imports without diffusers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Drift(nn.Module):
    """b(x, t) with t as (B, 1) absolute time; optionally symmetrised.

    `symmetrise` averages network outputs over a group during both training
    and inference:

        z2        b(x,t) := (b(x,t) - b(-x,t)) / 2                exactly odd
        rot180    b(x,t) := (b(x,t) + r b(r x,t)) / 2             parity-even
        z2rot180  the 4-element average of both                   both at once

    Disabled by default. Ising uses z2rot180; MNIST uses no symmetrisation.
    """

    def __init__(self, net: nn.Module, symmetrise: str = ""):
        super().__init__()
        assert symmetrise in ("", "z2", "rot180", "z2rot180"), symmetrise
        self.net, self.symmetrise = net, symmetrise

    def _raw(self, x, t):
        return self.net(x, t.reshape(-1), return_dict=False)[0]

    def _group(self, x):
        """The group elements as (transformed input, sign, needs-un-flipping)."""
        r = (-2, -1)
        g = [(x, 1.0, False)]
        if "rot180" in self.symmetrise:
            g.append((torch.flip(x, r), 1.0, True))
        if "z2" in self.symmetrise:
            g.append((-x, -1.0, False))
        if self.symmetrise == "z2rot180":
            g.append((-torch.flip(x, r), -1.0, True))
        return g

    def forward(self, x, t):
        """Average over symmetry transforms stacked as one batch.

        Separate calls round differently, so keep the batching used in training."""
        if not self.symmetrise:
            return self._raw(x, t)
        r = (-2, -1)
        g = self._group(x)
        y = self._raw(torch.cat([gx for gx, _, _ in g], 0), t.repeat(len(g), 1))
        out = 0
        for yi, (_, sign, unflip) in zip(y.chunk(len(g), 0), g):
            out = out + sign * (torch.flip(yi, r) if unflip else yi)
        return out / len(g)


def circularise(net: nn.Module) -> None:
    """Switch every k>1 convolution to circular padding, in place.

    Circular padding avoids zero-padding artifacts when applying a model
    trained on crops to periodic fields. It applies to input, residual,
    downsampling, and upsampling convolutions.

    `padding_mode` is absent from state_dict. Checkpoint loaders must restore
    the `circular` setting recorded in config.json.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size != (1, 1):
            m.padding_mode = "circular"


def build_drift(sample_size: int = 64, blocks=(32, 64, 128), layers_per_block: int = 2,
                channels: int = 1, symmetrise: str = "", circular: bool = False) -> Drift:
    from diffusers import UNet2DModel          # heavy; keep the module importable without it
    net = UNet2DModel(
        sample_size=sample_size, in_channels=channels, out_channels=channels,
        time_embedding_type="positional", layers_per_block=layers_per_block,
        block_out_channels=tuple(blocks),
        down_block_types=tuple("DownBlock2D" for _ in blocks),
        up_block_types=tuple("UpBlock2D" for _ in blocks))
    if circular:
        circularise(net)
    return Drift(net, symmetrise)

