"""The non-local control: a dilated conv stack in place of the per-cell MLP.

The locality claim needs a control that differs from the rule in exactly one
way, the reach of the update network. This subclasses `HeartNCA` and overrides
`_dx` alone, so the alive gate, the increment clip, the firing mask, the
monotone damage cap and the alive clamp are inherited rather than
reimplemented -- there is no second copy of the update to drift.

Six convolutions at constant depth, no pooling: the first four are 3x3 with
dilations (1, 2, 5, 9) and the last two 1x1, then a zero-initialised 1x1 head.
Per-step receptive field 1 + 2*(1+2+5+9) = 35 bins against the rule's 3, at
28 channels = 31,528 parameters against the rule's 31,388 trainable (1.004x).
A dilated convolution is still a convolution, so the network stays exactly
translation-equivariant; a pooled U-Net would be equivariant only to even
shifts and would differ from the rule in two ways instead of one.

This is the same architecture as `models/control.py`'s `NonLocalControl`, which
plays the same role in the other two chapters. It is rebuilt here rather
than imported because the parameter names must stay as the checkpoints of
record spell them, and because zero padding is right for a heart on a canvas
where the torus padding of the Ising control is not.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from domains.hearts.rule import HeartNCA

DILATIONS = (1, 2, 5, 9)
DEPTH = 6


class HeartControl(HeartNCA):
    def __init__(self, *args, conv_base: int = 28, conv_dilations=DILATIONS,
                 conv_depth: int = DEPTH, **kw) -> None:
        super().__init__(*args, **kw)
        d = [int(x) for x in conv_dilations]
        assert 1 <= len(d) <= conv_depth, f"{len(d)} dilations > depth {conv_depth}"
        self.conv_dilations, self.conv_base = tuple(d), int(conv_base)
        self.conv_depth = int(conv_depth)
        self.receptive_field = 1 + 2 * sum(d)

        # The parent's MLP and perception are dead weight here: this update
        # reads the raw state, not Sobel features. Drop them so the parameter
        # count and the checkpoint describe what actually runs.
        del self.fc1, self.fc2, self.fc3, self.sensor

        cin = self.n_channels + self.noise_channels + self.class_dim
        b = self.conv_base
        self.conv = nn.ModuleList(
            [nn.Conv2d(cin if i == 0 else b, b, 3, padding=d[i], dilation=d[i])
             if i < len(d) else nn.Conv2d(cin if i == 0 else b, b, 1)
             for i in range(conv_depth)])
        self.head = nn.Conv2d(b, self.n_channels, 1)
        nn.init.zeros_(self.head.weight)         # start as the identity map
        nn.init.zeros_(self.head.bias)

    def _dx(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        b, _, h, w = x.shape
        feats = [x]
        if self.noise_channels > 0:
            feats.append(torch.randn(b, self.noise_channels, h, w,
                                     device=x.device, dtype=x.dtype))
        if self.class_dim > 0:
            if y is None:
                raise ValueError("class_dim > 0 but no stage label passed to step()")
            oh = F.one_hot(y.reshape(-1).long(), self.class_dim).to(x.dtype)
            feats.append(oh[:, :, None, None].expand(-1, -1, h, w))
        f = torch.cat(feats, dim=1)
        for conv in self.conv:
            f = F.relu(conv(f))
        return self.head(f)

    @property
    def hparams(self) -> dict:
        return dict(super().hparams, conv_base=self.conv_base,
                    conv_dilations=list(self.conv_dilations),
                    conv_depth=self.conv_depth)
