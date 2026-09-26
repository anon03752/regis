"""Run the pretrained cycling-MNIST model and save an MP4. Run from the repository root."""
import argparse
from contextlib import closing
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from domains.mnist.rule import CycleNCA

CHECKPOINT = "checkpoints/mnist_regis_seed3.pt"
STARTS = "checkpoints/mnist_starts.npy"
DEVICE = "cpu"
SEED = 0
STEPS = 300
FPS = 30
SCALE = 8
INK = np.array([45, 84, 160], dtype=np.float32)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="MP4 output path")
OUTPUT = parser.parse_args().output

torch.set_num_threads(4)
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
hparams = checkpoint["args"]["hparams"]
model = CycleNCA(**hparams)
model.load_state_dict({
    key.removeprefix("module."): value
    for key, value in checkpoint["ema_G"].items() if key != "n_averaged"
})
model.to(DEVICE).eval()

torch.manual_seed(SEED)
digits = np.load(STARTS, allow_pickle=False).astype(np.float32) / 255
state = torch.zeros(4, hparams["channel_n"], 32, 32, device=DEVICE)
state[:, -1] = torch.from_numpy(digits).to(DEVICE)
latent = torch.randn(4, hparams["z_dim"], device=DEVICE)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode(), closing(imageio_ffmpeg.write_frames(
    str(OUTPUT), (64 * SCALE, 64 * SCALE), fps=FPS, quality=8,
)) as video:
    video.send(None)
    for step in range(STEPS + 1):
        digits = model.readout(state)[:, 0].cpu().numpy().clip(0, 1)
        # Arrange the four trajectories in a 2x2 grid.
        field = digits.reshape(2, 2, 32, 32).transpose(0, 2, 1, 3).reshape(64, 64)
        rgb = np.rint(255 + field[..., None] * (INK - 255)).astype(np.uint8)
        video.send(rgb.repeat(SCALE, axis=0).repeat(SCALE, axis=1))
        if step < STEPS:
            state = model.step(state, z=latent)

print(f"Saved {OUTPUT}")
