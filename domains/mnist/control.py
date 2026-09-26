"""Non-local control for cycling MNIST.

Wraps models.control.NonLocalControl with the same state layout, latent
conditioning, alive mask, firing, positional feature, padding, and clipping
as CycleNCA. Both models use the same trainer.

Dilations (1, 2, 5, 9) give a 35-pixel receptive field per update on the
32-pixel canvas. The control has 82,625 parameters; REGIS has 87,825.
"""
from __future__ import annotations

from models.control import NonLocalControl


class CycleControl(NonLocalControl):
    def __init__(self, channel_n: int, base: int, z_dim: int, fire_rate: float, flat_dilations):
        # Use the same positional feature as CycleNCA.
        super().__init__(channels=channel_n, noise_channels=0, base=base,
                         dilations=flat_dilations, fire_rate=fire_rate, update_clip=1.0,
                         z_dim=z_dim, visible_index=channel_n - 1,
                         alive_mask=True, positional=True,
                         padding_mode="zeros")      # Match CycleNCA's padding.
        self.channel_n, self.base, self.flat_dilations = channel_n, base, flat_dilations

    @property
    def hparams(self) -> dict:
        return dict(channel_n=self.channel_n, base=self.base, z_dim=self.z_dim,
                    fire_rate=self.fire_rate, flat_dilations=self.flat_dilations)
