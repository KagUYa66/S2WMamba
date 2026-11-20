# -*- coding: utf-8 -*-
"""Score training checkpoints on the official validation split.

Select the checkpoint on validation PSNR, never on the test set.

Usage:
  python eval_valid.py --valid ../data/valid_wv3.h5 \
      --ckpt-glob '../runs/fm84_step_seed1_500/ep*.pt'
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch

from model.wavelet_mamba_fm import WaveletMamba
from model.wavelet_mamba_fm4 import WaveletMamba as WaveletMamba4
from model.ckpt_compat import load_compat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--valid", required=True)
    p.add_argument("--ckpt-glob", required=True)
    p.add_argument("--arch", choices=["fm", "fm4"], default="fm")
    p.add_argument("--c-ms", type=int, default=8)
    p.add_argument("--c-conv", type=int, default=32)
    p.add_argument("--agg", choices=["msdg", "fm"], default="fm")
    p.add_argument("--agg-d", type=int, default=84)
    p.add_argument("--agg-depth", type=int, default=0)
    p.add_argument("--variant", choices=["full", "coupled", "no_spec", "no_spat", "rev"], default=None)
    p.add_argument("--fuse", choices=["mamba", "conv"], default=None)
    p.add_argument("--wave", choices=["haar", "split", "poly", "conv", "lofix", "ortho", "lofixo", "lofixn"], default=None)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    dev = "cuda"
    with h5py.File(args.valid, "r") as h:
        gt = torch.from_numpy(np.asarray(h["gt"][...], dtype=np.float32) / 2047.0).to(dev)
        pan = torch.from_numpy(np.asarray(h["pan"][...], dtype=np.float32) / 2047.0).to(dev)
        ms = torch.from_numpy(np.asarray(h["ms"][...], dtype=np.float32) / 2047.0).to(dev)
    print("valid samples:", gt.shape[0], flush=True)

    fuse = args.fuse
    if fuse is None:
        _cfg = os.path.join(os.path.dirname(os.path.abspath(args.ckpt_glob)), "config.json")
        fuse = "mamba"
        if os.path.exists(_cfg):
            try:
                with open(_cfg) as f:
                    fuse = json.load(f).get("fuse", "mamba")
            except Exception:
                pass
    wave = args.wave
    if wave is None:
        _cfg = os.path.join(os.path.dirname(os.path.abspath(args.ckpt_glob)), "config.json")
        wave = "haar"
        if os.path.exists(_cfg):
            try:
                with open(_cfg) as f:
                    wave = json.load(f).get("wave", "haar")
            except Exception:
                pass
    variant = args.variant
    if variant is None:
        cfg = os.path.join(os.path.dirname(os.path.abspath(args.ckpt_glob)), "config.json")
        variant = "coupled"
        if os.path.exists(cfg):
            try:
                with open(cfg) as f:
                    variant = json.load(f).get("variant", "coupled")
            except Exception:
                pass
    Net = WaveletMamba if args.arch == "fm" else WaveletMamba4
    model = Net(C_ms=args.c_ms, H=64, W=64, Scale=4, C_conv=args.c_conv,
                agg=args.agg, agg_d=args.agg_d,
                agg_depth=args.agg_depth, variant=variant, fuse=fuse, wave=wave).to(dev)
    files = sorted(glob.glob(args.ckpt_glob))
    if not files:
        raise SystemExit("no checkpoints match " + args.ckpt_glob)

    print("ckpt,epoch,psnr,l1", flush=True)
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        ep = ck.get("epoch", -1) if isinstance(ck, dict) else -1
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        load_compat(model, sd)
        model.eval()
        psnrs, l1s = [], []
        with torch.no_grad():
            for i in range(0, gt.shape[0], args.batch_size):
                g = gt[i:i + args.batch_size]
                pn = pan[i:i + args.batch_size]
                m = ms[i:i + args.batch_size]
                sr = model(m, pn)
                l1s.append(torch.mean(torch.abs(sr - g)).item())
                for j in range(g.shape[0]):
                    mse = torch.mean((sr[j] - g[j]) ** 2).item()
                    psnrs.append(10.0 * np.log10(1.0 / mse))
        print("%s,%d,%.4f,%.6f" % (os.path.basename(f), ep,
                                   float(np.mean(psnrs)), float(np.mean(l1s))),
              flush=True)


if __name__ == "__main__":
    main()
