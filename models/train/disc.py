"""The Ising critic: a no-norm conv stack, a global average pool and a linear
logit.

R3GAN: the zero-centred gradient penalty is the only regulariser, so there are
no normalisation layers. The critic is local: each final feature sees a window
of 2^(n_down+1) + 1 pixels a side (33 at n_down = 4, on 128^2 crops), and the
logit is the average of linear judgements of those windows.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv2d(3x3) -> LeakyReLU, no normalization."""

    def __init__(self, in_c: int, out_c: int, downsample: bool = False) -> None:
        super().__init__()
        s = 2 if downsample else 1
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=s, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Critic(nn.Module):
    """(B, in_channels, H, W) -> (B,) logits: a stride-1 block, then `n_down`
    stride-2 blocks (channels base, 2*base, 4*base, capped at 4*base).
    Input channels: [start,] x and one one-hot plane per marginal for the
    target time t_b (built in domains/ising/train.py)."""

    def __init__(self, in_channels: int, base: int, n_down: int) -> None:
        super().__init__()
        widths = [in_channels] + [base * min(2 ** i, 4) for i in range(n_down + 1)]
        blocks = [_ConvBlock(widths[0], widths[1], downsample=False)]
        for i in range(1, n_down + 1):
            blocks.append(_ConvBlock(widths[i], widths[i + 1], downsample=True))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.blocks(x)).flatten(1)).squeeze(-1)
