# -*- coding: utf-8 -*-
"""Inference for the FM-aggregator model (same tiling protocol as the original)."""
import argparse
import json
import os

import numpy as np
import scipy.io as sio
import torch

from model.wavelet_mamba_fm import WaveletMamba
from model.wavelet_mamba_fm4 import WaveletMamba as WaveletMamba4
from model.ckpt_compat import load_compat
from utils.load_test_data import load_h5py_with_hp


def resolve_cfg(weight, key, default):
    cfg = os.path.join(os.path.dirname(os.path.abspath(weight)), "config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg) as f:
                return json.load(f).get(key, default)
        except Exception:
            pass
    return default


def resolve_variant(variant, weight):
    """Explicit --variant wins. Otherwise take it from config.json next to the
    weights; a config without the key predates the split (i.e. 'coupled')."""
    if variant:
        return variant
    cfg = os.path.join(os.path.dirname(os.path.abspath(weight)), "config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg) as f:
                return json.load(f).get("variant", "coupled")
        except Exception:
            pass
    return "coupled"


def main(args):
    cut_size, pad = args.cut_size, args.pad
    Net = WaveletMamba if args.arch == "fm" else WaveletMamba4
    model = Net(H=cut_size + 2 * pad, W=cut_size + 2 * pad,
                C_ms=args.c_ms, C_conv=args.c_conv, agg=args.agg,
                agg_d=args.agg_d, agg_depth=args.agg_depth,
                variant=args.variant, fuse=args.fuse, wave=args.wave).to(args.device)
    weight = torch.load(args.weight, map_location="cpu")
    if isinstance(weight, dict) and "model" in weight and "epoch" in weight:
        weight = weight["model"]
    if any(k.startswith("module.") for k in weight):
        weight = {k.replace("module.", "", 1): v for k, v in weight.items()}
    load_compat(model, weight)
    model.eval()

    ms, _, pan, _ = load_h5py_with_hp(args.file_path)
    image_num, C, h, w = ms.shape
    _, _, H, W = pan.shape
    ms_size = cut_size // 4
    edge_H = (cut_size - (H - (H // cut_size) * cut_size)) % cut_size
    edge_W = (cut_size - (W - (W // cut_size) * cut_size)) % cut_size

    os.makedirs(args.save_dir, exist_ok=True)
    for k in range(image_num):
        print("Processing the %dth sample..." % (k + 1), flush=True)
        with torch.no_grad():
            x1 = ms[k].unsqueeze(0).float().to(args.device)
            x2 = pan[k, 0].unsqueeze(0).unsqueeze(0).float().to(args.device)
            x1_pad = torch.zeros(1, C, h + pad // 2 + edge_H // 4,
                                 w + pad // 2 + edge_W // 4).to(args.device)
            x2_pad = torch.zeros(1, 1, H + pad * 2 + edge_H, W + pad * 2 + edge_W).to(args.device)
            x1 = torch.nn.functional.pad(x1, (pad // 4, pad // 4, pad // 4, pad // 4),
                                         args.pad_mode)
            x2 = torch.nn.functional.pad(x2, (pad, pad, pad, pad), args.pad_mode)
            x1_pad[:, :, :h + pad // 2, :w + pad // 2] = x1
            x2_pad[:, :, :H + pad * 2, :W + pad * 2] = x2
            output = torch.zeros(1, C, H + edge_H, W + edge_W).to(args.device)

            for i in range((H + edge_H) // cut_size):
                for j in range((W + edge_W) // cut_size):
                    MS = x1_pad[:, :, i * ms_size:(i + 1) * ms_size + pad // 2,
                           j * ms_size:(j + 1) * ms_size + pad // 2]
                    PAN = x2_pad[:, :, i * cut_size:(i + 1) * cut_size + 2 * pad,
                           j * cut_size:(j + 1) * cut_size + 2 * pad]
                    if args.tta == "flips":
                        tf = [(False, fh, fv) for fh in (False, True)
                              for fv in (False, True)]
                    elif args.tta == "d4":
                        tf = [(tr, fh, fv) for tr in (False, True)
                              for fh in (False, True)
                              for fv in (False, True)]
                    else:
                        tf = None
                    if tf is not None:
                        outs = []
                        for tr, fh, fv in tf:
                            x, y = MS, PAN
                            if tr:
                                x = x.transpose(2, 3).contiguous()
                                y = y.transpose(2, 3).contiguous()
                            if fh:
                                x = torch.flip(x, [3]); y = torch.flip(y, [3])
                            if fv:
                                x = torch.flip(x, [2]); y = torch.flip(y, [2])
                            o = model(x, y)
                            if fv:
                                o = torch.flip(o, [2])
                            if fh:
                                o = torch.flip(o, [3])
                            if tr:
                                o = o.transpose(2, 3).contiguous()
                            outs.append(o)
                        sr = sum(outs) / len(outs)
                    else:
                        sr = model(MS, PAN)
                    sr = torch.clamp(sr, 0, 1)
                    output[:, :, i * cut_size:(i + 1) * cut_size,
                           j * cut_size:(j + 1) * cut_size] = \
                        sr[:, :, pad:cut_size + pad, pad:cut_size + pad] * 2047.

            output = output[:, :, :H, :W].squeeze().permute(1, 2, 0).cpu().numpy()
        sio.savemat(os.path.join(args.save_dir, "output_mulExm_%d.mat" % k),
                    {"sr": output})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file_path", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--cut_size", type=int, default=256)
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--arch", choices=["fm", "fm4"], default="fm")
    p.add_argument("--c-ms", type=int, default=8)
    p.add_argument("--c-conv", type=int, default=32)
    p.add_argument("--variant", choices=["full", "coupled", "no_spec", "no_spat", "rev"], default=None)
    p.add_argument("--fuse", choices=["mamba", "conv"], default=None)
    p.add_argument("--wave", choices=["haar", "split", "poly", "conv", "lofix", "ortho", "lofixo", "lofixn"], default=None)
    p.add_argument("--agg", choices=["msdg", "fm", "add"], default="fm")
    p.add_argument("--agg-d", type=int, default=84)
    p.add_argument("--agg-depth", type=int, default=0)
    p.add_argument("--tta", choices=["none", "flips", "d4"], default="none")
    p.add_argument("--pad-mode", choices=["reflect", "replicate", "constant"],
                   default="reflect")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    a.variant = resolve_variant(a.variant, a.weight)
    a.fuse = a.fuse or resolve_cfg(a.weight, 'fuse', 'mamba')
    a.wave = a.wave or resolve_cfg(a.weight, 'wave', 'haar')
    main(a)
