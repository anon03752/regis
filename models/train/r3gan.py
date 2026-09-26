"""The R3GAN critic of the MNIST and hearts experiments, and the R3GAN losses.

A residual no-norm convolutional critic with projection conditioning on the
observation time (Miyato & Koyama, 2018): `len(widths)-1` strided stages of
`ResidualBlock`s plus a final basis stage, the logit read as <e(t), h>. The zero-centred
gradient penalty is the sole regulariser, so there are no normalisation layers.
The two experiments set their own width, stage count, blocks, input channels
and classes, and the final basis (see `DiscriminativeBasis`).

`domains/mnist/fast_gp.py` computes the same penalty's parameter gradient with
forward-mode AD, for speed; it is not a different objective.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


LRELU_SLOPE = 0.2
BIASED_ACT_GAIN = math.sqrt(2.0 / (1.0 + LRELU_SLOPE**2))
KERNEL_SIZE = 3                # of the grouped conv in every residual block


def msr_init(layer, activation_gain=1.0):
    """Conv2d or Linear: weights N(0, gain^2 / fan_in), bias zero."""
    with torch.no_grad():
        nn.init.normal_(layer.weight, std=activation_gain / math.sqrt(layer.weight[0].numel()))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class BiasedActivation(nn.Module):
    gain = BIASED_ACT_GAIN

    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return F.leaky_relu(x + self.bias.to(x.dtype).reshape(1, -1, 1, 1), negative_slope=LRELU_SLOPE)


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, groups=1, bias=False, activation_gain=1.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layer = msr_init(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding, groups=groups, bias=bias),
            activation_gain,
        )

    def forward(self, x):
        return self.layer(x)


class InterpolativeDownsampler(nn.Module):
    """2x downsampling by the normalised (1, 2, 1) x (1, 2, 1) tent filter."""

    def __init__(self):
        super().__init__()
        w = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("kernel", torch.outer(w, w) / 16)

    def forward(self, x):
        c = x.shape[1]
        weight = self.kernel.to(x.dtype).reshape(1, 1, *self.kernel.shape).repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, stride=2, padding=1, groups=c)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, cardinality, expansion, variance_scale_param):
        super().__init__()
        num_linear = 3
        expanded_ch = in_ch * expansion
        groups = max(1, math.gcd(expanded_ch, cardinality))
        act_gain = BiasedActivation.gain * (variance_scale_param ** (-1.0 / (2 * num_linear - 2)))

        self.conv1 = Conv2d(in_ch, expanded_ch, kernel_size=1, activation_gain=act_gain)
        self.act1 = BiasedActivation(expanded_ch)
        self.conv2 = Conv2d(expanded_ch, expanded_ch, kernel_size=KERNEL_SIZE, groups=groups, activation_gain=act_gain)
        self.act2 = BiasedActivation(expanded_ch)
        self.conv3 = Conv2d(expanded_ch, in_ch, kernel_size=1, activation_gain=0.0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(self.act1(y))
        y = self.conv3(self.act2(y))
        return x + y


class DownsampleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resampler = InterpolativeDownsampler()

    def forward(self, x):
        return self.resampler(x)


class DiscriminativeBasis(nn.Module):
    """Final stage: collapse the feature map to a vector.

    `depthwise` is a 4x4 depthwise conv, so it requires the map to BE 4x4 --
    true for MNIST (32 px through three stride-2 stages). The heart grid is
    48 px through four stages, which leaves 3x3, so the hearts critic pools
    adaptively instead.
    """

    def __init__(self, in_ch, out_dim, mode="depthwise"):
        super().__init__()
        assert mode in ("depthwise", "avgpool"), f"unknown basis mode {mode!r}"
        self.mode = mode
        if mode == "depthwise":
            self.depthwise = msr_init(nn.Conv2d(in_ch, in_ch, 4, stride=1, padding=0, groups=in_ch, bias=False))
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = msr_init(nn.Linear(in_ch, out_dim, bias=False))

    def forward(self, x):
        x = self.depthwise(x) if self.mode == "depthwise" else self.adaptive_pool(x)
        return self.linear(x.reshape(x.shape[0], -1))


class DiscriminatorStage(nn.Module):
    """Residual blocks, then a 2x downsample, or the basis if `out_dim` is given."""

    def __init__(self, in_ch, cardinality, num_blocks, expansion, variance_scale_param,
                 out_dim=None, basis="depthwise"):
        super().__init__()
        transition = DownsampleLayer() if out_dim is None else DiscriminativeBasis(in_ch, out_dim, basis)
        blocks = [ResidualBlock(in_ch, cardinality, expansion, variance_scale_param)
                  for _ in range(num_blocks)]
        blocks.append(transition)
        self.layers = nn.ModuleList(blocks)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    """The critic D(x, t): conditioned on the target observation time t, an
    integer index (MNIST's next digit, hearts' stage), by projection: logit
    <e(t), h>."""

    def __init__(
        self,
        widths=(32, 32, 32, 32),
        cardinalities=(4, 4, 4, 4),
        blocks_per_stage=(2, 2, 2, 2),
        expansion=2,
        num_classes=10,
        embed_dim=32,
        in_channels=1,
        basis="depthwise",
    ):
        super().__init__()
        variance_scale_param = sum(blocks_per_stage)
        self.from_rgb = Conv2d(in_channels, widths[0], kernel_size=1)

        layers = []
        for i in range(len(widths) - 1):
            layers.append(DiscriminatorStage(
                widths[i], cardinalities[i], blocks_per_stage[i], expansion, variance_scale_param))

        layers.append(DiscriminatorStage(
            widths[-1], cardinalities[-1], blocks_per_stage[-1], expansion, variance_scale_param,
            out_dim=embed_dim, basis=basis))
        self.main = nn.ModuleList(layers)
        self.embed = nn.Embedding(num_classes, embed_dim)

    def forward(self, x, t):
        x = self.from_rgb(x)
        for layer in self.main:
            x = layer(x)
        return (x * self.embed(t)).sum(dim=1)


# --- the losses -------------------------------------------------------------
# Relativistic pairing with zero-centred penalties on real AND generated inputs
# (R3GAN).

def zero_centered_gp(samples: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """||grad_x D(x)||^2 per sample: R1 on observed inputs, R2 on generated
    ones. `samples` must require grad."""
    grad, = torch.autograd.grad(logits.sum(), samples, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1)


def relativistic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """E softplus(b - a). L_D = relativistic(d_obs, d_gen); L_NCA = relativistic(d_gen, d_obs)."""
    return F.softplus(b - a).mean()
