"""Non-local control with a dilated convolutional update network.

Four 3x3 convolutions at dilations (1, 2, 5, 9), two 1x1 layers, and a
zero-initialised output layer give a 35-pixel receptive field per half-step
(69 pixels per full checkerboard sweep). At width 38 the network has 43,169
parameters; the 3x3 Ising NCA has 17,436.

UpdateRule supplies the same noise inputs, firing schedule, clipping, and
update penalty used by REGIS. With parity_even, each spatial kernel is
symmetrised under 180-degree rotation. With z2_sym, the full increment is
antisymmetrised under spin inversion. Select with --g-arch control.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nca import UpdateRule, positional_feature


def _c3(cin, cout, dil, padding_mode):
    return nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil, padding_mode=padding_mode)


class NonLocalControl(UpdateRule):
    def __init__(self, *, channels, noise_channels, base, dilations=(1, 2, 5, 9),
                 fire_rate, update_clip,
                 parity_even=False, z2_sym=False, fire_mode="bernoulli",
                 z_dim=0,                      # MNIST's persistent noise xi, broadcast to every cell
                 visible_index=0,              # which channel is the observable
                 alive_mask=False,             # wipe cells with no ink in their 3x3 neighbourhood
                 padding_mode="circular", positional=False):
        super().__init__(fire_mode=fire_mode, fire_rate=fire_rate, update_clip=update_clip,
                         z2_sym=z2_sym, alive_mask=alive_mask, visible_index=visible_index)
        self.parity_even, self.z_dim, self.positional = parity_even, z_dim, positional
        # a dilation has no parameters, so without this buffer a checkpoint would
        # load cleanly into a stack with the wrong dilations
        self.register_buffer("_arch_dil_" + "_".join(map(str, dilations)), torch.zeros(0))
        cin = channels + noise_channels + int(positional) + z_dim
        self.flat = nn.ModuleList([_c3(cin if i == 0 else base, base, d, padding_mode)
                                   for i, d in enumerate(dilations)]
                                  + [nn.Conv2d(base, base, 1) for _ in range(2)])
        self.head = nn.Conv2d(base, channels, 1)
        nn.init.zeros_(self.head.weight)          # Start with zero predicted increments.
        nn.init.zeros_(self.head.bias)

    @classmethod
    def from_hparams(cls, hp):
        """Build from the Ising trainer's recorded hparams."""
        return cls(channels=hp["channels"], noise_channels=hp["noise_channels"], base=hp["conv_base"],
                   dilations=[int(d) for d in hp["conv_dilations"].split(",")], fire_rate=hp["fire_rate"],
                   fire_mode=hp["fire_mode"], update_clip=hp["update_clip"],
                   parity_even=hp["parity_even"], z2_sym=hp["z2_sym"])

    def _w(self, conv):
        w = conv.weight
        # A 1x1 kernel is already symmetric. Skip it to preserve gradient rounding.
        if self.parity_even and w.shape[-1] > 1:
            w = 0.5 * (w + w.flip(-1, -2))
        return w

    def _conv(self, conv, x):
        """nn.Conv2d.forward with the symmetrised weight"""
        return conv._conv_forward(x, self._w(conv), conv.bias)

    def _net(self, state, z, noise):
        x = [state] + ([noise] if noise is not None else [])
        if self.positional:
            x.append(positional_feature(*state.shape[2:], state.device, state.dtype)
                     .expand(state.shape[0], -1, -1, -1))
        if self.z_dim > 0:
            x.append(z[:, :, None, None].expand(-1, -1, *state.shape[2:]))
        x = torch.cat(x, 1)
        for conv in self.flat:
            x = F.relu(self._conv(conv, x))
        return self.head(x)
