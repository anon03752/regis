"""Run the pretrained Ising model and save an MP4. Run from the repository root."""
import argparse
from contextlib import closing
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from models.nca import NCA

CHECKPOINT = "checkpoints/regis_k3_seed1.pt"
DEVICE = "cpu"
SEED = 0
SIZE = 256
STEPS = 1000
SAVE_EVERY = 4
FPS = 30
SCALE = 2
INK = np.array([45, 84, 160], dtype=np.float32)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="MP4 output path")
OUTPUT = parser.parse_args().output

torch.set_num_threads(4)
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
hparams = checkpoint["args"]["hparams"]
model = NCA.from_hparams(hparams)
model.load_state_dict({
    key.removeprefix("module."): value
    for key, value in checkpoint["ema_G"].items() if key != "n_averaged"
})
model.to(DEVICE).eval()

torch.manual_seed(SEED)
state = (torch.rand(1, 1, SIZE, SIZE, device=DEVICE) < 0.5).float()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode(), closing(imageio_ffmpeg.write_frames(
    str(OUTPUT), (SIZE * SCALE, SIZE * SCALE), fps=FPS, quality=8,
)) as video:
    video.send(None)
    for step in range(STEPS + 1):
        if step % SAVE_EVERY == 0 or step == STEPS:
            field = (model.readout(state)[0, 0].cpu().numpy() > 0.5).astype(np.float32)
            rgb = np.rint(255 + field[..., None] * (INK - 255)).astype(np.uint8)
            video.send(rgb.repeat(SCALE, axis=0).repeat(SCALE, axis=1))
        if step < STEPS:
            # The two checkerboard half-steps use independent noise draws.
            noise = torch.randn(1, hparams["noise_channels"], SIZE, SIZE, device=DEVICE)
            noise2 = noise.std() * torch.randn_like(noise)
            state = model.step(state, noise=noise, noise2=noise2)

print(f"Saved {OUTPUT}")
