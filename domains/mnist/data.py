"""Prepare MNIST class pools and the transport baselines' digit loop.

Images are resized to 22x22, padded to 32x32, and stored on the [-1, 1]
scale. Marginal i has class i % 10, for i = 0..10. The two zero marginals
use disjoint halves of the zero images; observations are unpaired.
class_t<i>.pt contains training images and eval_t<i>.pt contains test images.

    python -m domains.mnist.data digitloop

Writes data/mnist/digitloop/{class,eval}_t<i>.pt. Runs on CPU in about a minute.
"""
import argparse

import torch
from torchvision import datasets

from workspace import WORK
from domains.mnist.classifier import TRANSFORM

SEED = 13                     # the split of the zeros


def digits_by_class(train: bool, device="cpu") -> dict[int, torch.Tensor]:
    """The training or test split by class, in dataset order: {c: (n_c, 1, 32, 32)}."""
    by_class = {c: [] for c in range(10)}
    for img, lab in datasets.MNIST(WORK / "data", train=train, download=True, transform=TRANSFORM):
        by_class[lab].append(img)
    return {c: torch.stack(v).to(device) for c, v in by_class.items()}


def digitloop(out):
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(SEED)
    for name, train in (("class", True), ("eval", False)):
        pools = list(digits_by_class(train).values())
        half, perm = len(pools[0]) // 2, torch.randperm(len(pools[0]), generator=g)
        pools += [pools[0][perm[half:2 * half]]]
        pools[0] = pools[0][perm[:half]]
        for i, x in enumerate(pools):
            torch.save(x * 2 - 1, out / f"{name}_t{i}.pt")
        print(f"{name}_t0..10: " + " ".join(str(len(x)) for x in pools), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_subparsers(dest="kind", required=True).add_parser("digitloop")
    ap.parse_args()
    digitloop(WORK / "data" / "mnist" / "digitloop")
    print("done", flush=True)


if __name__ == "__main__":
    main()
