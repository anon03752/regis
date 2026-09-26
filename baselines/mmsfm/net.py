"""MMSFM networks and sampling helpers.

The heads operate on [-1, 1] images. Ising uses spin-flip antisymmetry
and circular padding; MNIST and the hearts use a periodic time embedding.
Drift converts the outputs to the samplers' [0, 1] coordinates, and
one_transition integrates it."""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

UPSTREAM = Path(__file__).resolve().parent / 'upstream' / 'MMSFM'   # written by fetch_upstream.sh


def upstream():
    """The patched upstream's training module (images_train), with UPSTREAM on sys.path."""
    assert (UPSTREAM / 'scripts').exists(), 'patched upstream missing; run baselines/mmsfm/fetch_upstream.sh'
    sys.path[:0] = [str(UPSTREAM), str(UPSTREAM / 'conditional-flow-matching')]
    from scripts.images import images_train
    torch.autograd.set_detect_anomaly(False)   # images_train turns it on at import
    return images_train


class SymmetricHead(torch.nn.Module):
    """Spin-flip antisymmetric head: (base(t, x) - base(t, -x)) / 2."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, t, x):
        y, y_flipped = self.base(t.repeat(2), torch.cat([x, -x])).chunk(2)
        return (y - y_flipped) / 2


def build_unets(tr, dims, channels, attention_res, device):
    """Flow and score U-Nets: upstream's CIFAR-10 U-Net (depth 2, channel multipliers 1,2,2,2, fp32)
    at the given width and attention resolutions."""
    h = tr.get_hypers('cifar10', 64, dims)
    h.update(channels=channels, attention_res=attention_res)
    return tr.build_models(h, True, device)


def build_heads(tr, dims, channels, attention_res, device):
    """Ising flow and score U-Nets with spin-flip antisymmetry and circular padding."""
    heads = [SymmetricHead(m) for m in build_unets(tr, dims, channels, attention_res, device)]
    # Circular padding avoids zero-padding artifacts on periodic fields.
    # padding_mode is absent from state_dict, so checkpoint loaders must
    # construct the networks through this function too.
    for m in heads:
        for q in m.modules():
            if isinstance(q, torch.nn.Conv2d) and q.kernel_size != (1, 1):
                q.padding_mode = 'circular'
    return heads


def periodic_time_embedding(timesteps, dim):
    """The MNIST heads' clock: integer harmonics 1..dim/2 of period one in place of upstream's
    sinusoidal embedding, so t and t + 1 are the same point of the digit loop. Same width, no
    parameters; the phase is taken in float64 so the seam at t = 1 does not round."""
    phase = torch.remainder(timesteps.double(), 1.)
    harmonics = torch.arange(1, dim // 2 + 1, dtype=torch.float64, device=phase.device)
    args = 2 * math.pi * phase[:, None] * harmonics[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).float()


def configure_execution():
    """TF32 matmuls and convolutions; attention without activation recompute (upstream's
    AttentionBlock hard-codes checkpointing)."""
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    from torchcfm.models.unet.unet import AttentionBlock
    AttentionBlock.forward = lambda self, x: self._forward(x)


def sdpa_attention(dtype):
    """QKVAttentionLegacy.forward through SDPA at `dtype`: softmax(q k^T / sqrt(ch)) v with
    each head's q, k, v stored in turn; the output keeps the input's dtype."""
    def forward(self, qkv):
        b, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = (z.transpose(-1, -2).to(dtype).contiguous()
                   for z in qkv.reshape(b, self.n_heads, 3 * ch, length).split(ch, dim=2))
        return F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, -1, length).to(qkv.dtype)
    return forward


class Drift(torch.nn.Module):
    """dx/dt on the [0, 1] scale the samplers integrate: the heads take x*2-1 at clock
    t + phase, so their outputs (flow + score) are halved"""

    def __init__(self, flow, score):
        super().__init__()
        self.flow, self.score = flow, score

    def forward(self, t, x, phase):
        ts, native = (t + phase).float(), (x * 2 - 1).float()
        flow, score = self.flow(ts, native), self.score(ts, native)
        return flow.double() / 2 + score.double() / 2


def srk_schedule(dev, duration, steps):
    """(start, length) of the SRK steps over one transition, as torchsde accumulates them"""
    t, rows = 0., []
    while t < duration:
        end = min(t + duration / steps, duration)
        rows.append((t, end - t))
        t = end
    return torch.tensor(rows, dtype=torch.float64, device=dev)


def one_transition(drift, x, phase, schedule, sigma, gen, keep=None, tile=1):
    """One transition: SRA1 (torchsde's additive-noise SRK) with Gaussian increments W and
    space-time integrals U, term by term in torchsde's order (the order fixes the rounding).

    `keep`, a 0/1 channel mask per sample, holds the masked channels' drift and noise at zero
    at every internal stage: a knock-out that starts from a zeroed channel stays exactly zero.
    With `tile` > 1 the batch is that many copies of one ensemble and they share its noise, as
    if each copy were run alone from the same generator state: paired knock-outs in one pass."""
    z = torch.randn((len(schedule), 2, len(x) // tile, *x.shape[1:]), device=x.device, dtype=x.dtype,
                    generator=gen)
    dts = schedule[:, 1].reshape(-1, *[1] * x.ndim)
    W = dts.sqrt() * z[:, 0]
    U = dts ** 1.5 * (.5 * z[:, 0] + z[:, 1] / (12 ** .5))    # Var U = dt^3/3, Cov(W, U) = dt^2/2
    f = drift if keep is None else (lambda t, y, p: drift(t, y, p) * keep)
    for (t0, dt), dw, du in zip(schedule, W, U):
        if tile > 1:
            dw, du = dw.repeat(tile, *[1] * (x.ndim - 1)), du.repeat(tile, *[1] * (x.ndim - 1))
        if keep is not None:
            dw, du = dw * keep, du * keep
        rdt = 1 / dt
        f0 = f(t0 + 0 * dt, x, phase)
        y1 = x + (1 / 3) * f0 * dt + (sigma / 2) * (1 * dw + -1 * du * rdt)
        h1 = x + (3 / 4) * f0 * dt + (sigma / 2) * ((3 / 2) * du * rdt)
        f1 = f(t0 + (3 / 4) * dt, h1, phase)
        x = y1 + (2 / 3) * f1 * dt + (sigma / 2) * (0 * dw + 1 * du * rdt)
    return x
