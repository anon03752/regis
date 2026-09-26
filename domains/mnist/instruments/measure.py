"""Load EMA generators and construct shared starts from MNIST test digits."""
import numpy as np
import torch

from domains.mnist.classifier import SIDE
from domains.mnist.control import CycleControl
from domains.mnist.rule import CycleNCA

PER, CHUNK = 500, 1500      # starts per digit; trajectories per draw of test digits


def load_ema_generator(ckpt_path, device):
    """-> (G, args): the EMA generator of a training checkpoint, and the run's recorded args."""
    s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = s["args"]
    Rule = CycleControl if a["g_arch"] == "control" else CycleNCA
    G = Rule(**a["hparams"])
    G.load_state_dict({k.removeprefix("module."): v for k, v in s["ema_G"].items() if k != "n_averaged"})
    return G.eval().to(device), a


def start_states(classes, val_by_class, C, device, rng):
    """Start states: a real test digit of each requested class in the visible
    (last) channel, zeros in the hidden channels."""
    state = torch.zeros(len(classes), C, SIDE, SIDE, device=device)
    for i, c in enumerate(classes):
        pool = val_by_class[int(c)]
        state[i, -1:] = pool[int(rng.integers(len(pool)))]
    return state


def real_starts(val, channels, device):
    """-> (start, x0): the 5000 shared starts, 500 per digit, as start digit and
    start state. The test digit of each is drawn per CHUNK with numpy seed
    1000 + c, so CHUNK is part of the record."""
    start = np.repeat(np.arange(10), PER)
    x0 = torch.cat([start_states(start[c:c + CHUNK], val, channels, device, np.random.default_rng(1000 + c))
                    for c in range(0, len(start), CHUNK)])
    return start, x0
