"""Run the pretrained zebrafish-heart model on four wounded hearts and save an MP4. Run from the repository root."""
import argparse
from contextlib import closing
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from domains.hearts import cohort as C
from domains.hearts import rollout as R
from domains.hearts.results.style import CELL_COLOUR

CHECKPOINT = "checkpoints/hearts_regis_seed0.pt"
STARTS = "checkpoints/hearts_starts.npz"
SEED = 0
FPS = 10
SCALE = 6
PALETTE = np.array([[int(CELL_COLOUR[n].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
                    for n in C.CELL_TYPES + [C.DAMAGE_CHANNEL]], dtype=np.float32)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="MP4 output path")
OUTPUT = parser.parse_args().output

torch.set_num_threads(4)
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
rule = R.build_rule(checkpoint["args"])
rule.load_state_dict(checkpoint["ema"])
rule.eval()

starts = np.load(STARTS, allow_pickle=False)
state = torch.from_numpy(starts["state"])            # cell types, damage and alive
hidden = torch.zeros(len(state), checkpoint["args"]["n_hidden"], *state.shape[-2:])
state = torch.cat([state, hidden], dim=1)
mask = starts["mask"] > 0
# every update from the wound to 28 dpa, 15 per stage
_settled, _legs, every = R.roll(rule, state, record=True, seed=SEED)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
side = state.shape[-1]
with closing(imageio_ffmpeg.write_frames(
    str(OUTPUT), (2 * side * SCALE, 2 * side * SCALE), fps=FPS, quality=8,
)) as video:
    video.send(None)
    for frame in every:
        comp = frame[:, :R.N_VISIBLE].clamp_min(0).numpy()
        # the dominant cell type sets the hue, the bin's filling its strength over white
        density = (comp.sum(1) / 4.0).clip(0, 1)[..., None] ** 0.65
        rgb = 255 - density * (255 - PALETTE[comp.argmax(1)])
        rgb[~mask] = 255
        # Arrange the four hearts in a 2x2 grid.
        grid = rgb.reshape(2, 2, side, side, 3).transpose(0, 2, 1, 3, 4).reshape(2 * side, 2 * side, 3)
        video.send(np.rint(grid).astype(np.uint8).repeat(SCALE, axis=0).repeat(SCALE, axis=1))

print(f"Saved {OUTPUT}")
