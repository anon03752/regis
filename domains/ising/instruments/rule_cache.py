"""Cache rollouts of trained Ising update rules or the Glauber engine.

One rule update corresponds to one engine sweep. Learned rules use bf16
autocast on GPU. Engine chunk j uses seed ENGINE + 1000*j; learned-rule
noise continues the random stream used for that chunk's initial field.
See rollouts.py for initial states and the cache format."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from domains.ising.glauber import sweep_
from domains.ising.instruments.rollouts import READOUT_TIMES, chunks, quench, save
from models.nca import NCA
from models.control import NonLocalControl
from workspace import WORK

KEEP = set(READOUT_TIMES)


def load_rule(run, dev):
    """A trained rule's EMA generator (last checkpoint), its hparams and its step."""
    ck = torch.load(sorted((WORK / "runs" / run / "ckpts").glob("ckpt_*.pt"))[-1],
                    map_location=dev, weights_only=False)
    hp = ck["args"]["hparams"]
    G = (NonLocalControl if hp["g_arch"] == "control" else NCA).from_hparams(hp)
    G.load_state_dict({k.removeprefix("module."): v for k, v in ck["ema_G"].items() if k.startswith("module.")})
    return G.to(dev).eval(), hp, int(ck["step"])


def roll_engine(up, seed):
    s = torch.where(up, 1.0, -1.0)
    gen = torch.Generator(device=up.device).manual_seed(seed)
    out = []
    for t in range(1, READOUT_TIMES[-1] + 1):
        sweep_(s, 1.0, gen)
        if t in KEEP:
            out.append(np.packbits((s > 0).cpu().numpy()))
    return out


def roll_rule(G, hp, up):
    n, lat = len(up), up.shape[-1]
    st = up.float()[:, None]      # the state is the spin field alone: no hidden channels
    out = []
    with torch.autocast(up.device.type, dtype=torch.bfloat16, enabled=up.is_cuda):
        for t in range(1, READOUT_TIMES[-1] + 1):
            noise = torch.randn(n, hp["noise_channels"], lat, lat, device=up.device)
            # Preserve the noise scaling used in the original rollout caches.
            noise2 = noise.std() * torch.randn_like(noise)
            st = G.step(st, noise=noise, noise2=noise2)
            if t in KEEP:
                out.append(np.packbits((G.readout(st)[:, 0].float() > 0.5).cpu().numpy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="a trained rule's run directory name under runs/")
    src.add_argument("--engine", type=int, help="roll the engine instead, with this dynamics seed")
    ap.add_argument("--ns", type=int, default=512, help="number of fields")
    ap.add_argument("--lat", type=int, default=512)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.run:
        G, hp, step = load_rule(args.run, dev)
        meta = dict(run=args.run, step=step)
    else:
        meta = dict(engine_seed=args.engine)
    t0 = time.time()
    bits = [[] for _ in READOUT_TIMES]
    for j, n in chunks(args.ns):
        up = quench(j, n, args.lat, dev)
        frames = roll_rule(G, hp, up) if args.run else roll_engine(up, args.engine + 1000 * j)
        for b, f in zip(bits, frames):
            b.append(f)
        print(f"chunk {j}: {n} fields to t = {READOUT_TIMES[-1]} [{time.time() - t0:.0f}s]", flush=True)
    save(args.run or f"truth_seed{args.engine}", args.lat, args.ns, READOUT_TIMES, bits, **meta)


if __name__ == "__main__":
    main()
