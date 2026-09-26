"""Train and load the frozen classifier used for MNIST evaluation.

SmallCNN predicts digit labels and provides 128-dimensional penultimate-layer
features for Frechet distances. The paper's classifier has 545K parameters
and 98.5% test accuracy. Training uses Adam at 1e-3, batch size 256, three
epochs, and seed 0. Images are resized to 22x22 and padded to 32x32.

The checkpoint is generated locally. Retraining can change classifications
and distances. The paper used the checkpoint with SHA-256
8b18f5d8566ec9db3be1bd5bff5e8b356f47d1af124ec72c3b4f25c338861689.

    python -m domains.mnist.classifier
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from workspace import WORK

SIDE = 32          # every digit is framed on a SIDE x SIDE canvas
TRANSFORM = transforms.Compose([
    transforms.Resize((22, 22)), transforms.Pad(5, fill=0), transforms.ToTensor()])
CKPT = WORK / "data" / "model-checkpoints" / "mnist-classifier.pth"


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        """(B, 1, 32, 32) -> (B, 128): the penultimate layer, before its ReLU."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc1(x.view(x.size(0), -1))

    def forward(self, x):
        """(B, 1, 32, 32) visible frames -> (B, 10) logits."""
        return self.fc2(F.relu(self.features(x)))


def load_classifier(device):
    """The frozen classifier: net(x) gives the logits, net.features(x) the Frechet features."""
    net = SmallCNN().to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device))
    return net.eval()


EPOCHS, BATCH, LR, SEED = 3, 256, 1e-3, 0       # the training recipe of the paper classifier


def main():
    from torch.utils.data import DataLoader
    from torchvision import datasets
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = datasets.MNIST(WORK / "data", train=True, download=True, transform=TRANSFORM)
    te = datasets.MNIST(WORK / "data", train=False, download=True, transform=TRANSFORM)
    trl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4)
    tel = DataLoader(te, batch_size=512, num_workers=4)
    torch.manual_seed(SEED)
    net = SmallCNN().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for ep in range(EPOCHS):
        net.train()
        for x, y in trl:
            opt.zero_grad()
            F.cross_entropy(net(x.to(dev)), y.to(dev)).backward()
            opt.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in tel:
                correct += (net(x.to(dev)).argmax(1).cpu() == y).sum().item()
                total += y.numel()
        print(f"epoch {ep + 1}/{EPOCHS}  test accuracy {correct / total:.4f}", flush=True)
    torch.save(net.state_dict(), CKPT)
    print(f"-> {CKPT}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
