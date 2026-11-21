# -*- coding: utf-8 -*-
"""Compute the validation L1 loss of every checkpoint of a run.

This is the quantity plotted next to the training curve: the same objective the
training minimises, evaluated on the held-out split instead of the training
patches. It is used only for the figure; the checkpoints reported in the paper
were selected on the reduced-resolution test split (Section IV-B of the paper).

Usage:
  python tools/val_loss.py --valid ../data/valid_wv3.h5 \
      --ckpts runs/wv3_wavelofix_500 --out data/wv3_val_loss.csv --device cpu
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.wavelet_mamba_fm import WaveletMamba
from model.wavelet_mamba_fm4 import WaveletMamba as WaveletMamba4
from model.ckpt_compat import load_compat


def epoch_of(path):
    m = re.search(r"ep(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--valid", required=True)
    p.add_argument("--ckpts", required=True, help="run directory holding ep*.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--arch", choices=["fm", "fm4"], default="fm")
    p.add_argument("--c-ms", type=int, default=8)
    p.add_argument("--c-conv", type=int, default=32)
    p.add_argument("--agg", default="fm")
    p.add_argument("--agg-d", type=int, default=84)
    p.add_argument("--agg-depth", type=int, default=0)
    p.add_argument("--variant", default="coupled")
    p.add_argument("--fuse", default="mamba")
    p.add_argument("--wave", default="lofix")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--maxv", type=float, default=2047.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0, help="use only the first N samples")
    args = p.parse_args()

    with h5py.File(args.valid, "r") as h:
        gt = np.asarray(h["gt"][...], dtype=np.float32) / args.maxv
        ms = np.asarray(h["ms"][...], dtype=np.float32) / args.maxv
        pan = np.asarray(h["pan"][...], dtype=np.float32) / args.maxv
    n = gt.shape[0] if not args.limit else min(args.limit, gt.shape[0])
    H, W = gt.shape[-2:]

    files = sorted(glob.glob(os.path.join(args.ckpts, "ep*.pt")), key=lambda f: epoch_of(f) or 0)
    if not files:
        sys.exit("no ep*.pt in " + args.ckpts)
    print("%d checkpoints, %d validation samples of %dx%d" % (len(files), n, H, W), flush=True)

    Net = WaveletMamba if args.arch == "fm" else WaveletMamba4
    model = Net(H=H, W=W, C_ms=args.c_ms, C_conv=args.c_conv, agg=args.agg,
                agg_d=args.agg_d, agg_depth=args.agg_depth, variant=args.variant,
                fuse=args.fuse, wave=args.wave).to(args.device)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("epoch,val_loss\n")
        with torch.no_grad():
            for path in files:
                ck = torch.load(path, map_location="cpu")
                sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
                if any(k.startswith("module.") for k in sd):
                    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                load_compat(model, sd)

                total, count = 0.0, 0
                for i in range(0, n, args.batch_size):
                    j = min(i + args.batch_size, n)
                    sr = model(torch.from_numpy(ms[i:j]).to(args.device),
                               torch.from_numpy(pan[i:j]).to(args.device))
                    ref = torch.from_numpy(gt[i:j]).to(args.device)
                    total += torch.abs(sr - ref).sum().item()
                    count += ref.numel()
                loss = total / count
                ep = epoch_of(path)
                fh.write("%d,%.8f\n" % (ep, loss))
                fh.flush()
                print("ep%03d  val L1 = %.6f" % (ep, loss), flush=True)


if __name__ == "__main__":
    main()
