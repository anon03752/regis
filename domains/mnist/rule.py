"""MNIST configuration of models.nca.NCA.

Uses fixed identity/Sobel perception, zero padding, increments clipped to
[-1, 1], and the last state channel as the visible image. The alive mask
clears inactive cells, including their hidden channels. A fixed Gaussian
positional feature reduces spatial drift during long rollouts.

`z` denotes the persistent noise vector, shared across space and fixed along
a trajectory. The evolving NCA state is named `state`. Constructor settings
are stored in `hparams` so checkpoints can reconstruct the model.
"""
from models.nca import NCA


class CycleNCA(NCA):
    def __init__(self, channel_n: int, hidden_dim: int, z_dim: int, fire_rate: float):
        super().__init__(channels=channel_n, mlp_width=hidden_dim, z_dim=z_dim,
                         fire_rate=fire_rate, update_clip=1.0,
                         alive_mask=True, padding_mode="constant", perception_mode="sobel",
                         visible_index=channel_n - 1, positional=True)
        self.channel_n, self.hidden_dim = channel_n, hidden_dim

    @property
    def hparams(self) -> dict:
        return dict(channel_n=self.channel_n, hidden_dim=self.hidden_dim, z_dim=self.z_dim, fire_rate=self.fire_rate)
