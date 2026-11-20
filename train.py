# -*- coding: utf-8 -*-
"""Clean, reproducible trainer for S2WMamba / TGRS.

Default recipe matches the original WV3 run:
    AdamW lr=4e-4, weight_decay=0, L1 loss, batch 32, 460 epochs,
    StepLR(step_size=100, gamma=0.7).
Note: the historical train.py defined randtrans() but never called it, so the
original run used NO augmentation. --aug defaults to "none" to match that;
use --aug flips only for a deliberate new experiment.
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.wavelet_mamba_fm import WaveletMamba
from model.wavelet_mamba_fm4 import WaveletMamba as WaveletMamba4
from utils.data import PansharpeningH5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augment(gt, pan, ms, gen):
    """Random flips, same transform for all three tensors."""
    idx = torch.randint(0, 4, (gt.shape[0],), generator=gen, device=gen.device)
    for k in range(4):
        m = idx == k
        if not m.any():
            continue
        if k == 1:
            gt[m] = torch.flip(gt[m], [3]); pan[m] = torch.flip(pan[m], [3]); ms[m] = torch.flip(ms[m], [3])
        elif k == 2:
            gt[m] = torch.flip(gt[m], [2]); pan[m] = torch.flip(pan[m], [2]); ms[m] = torch.flip(ms[m], [2])
        elif k == 3:
            gt[m] = torch.flip(gt[m], [2, 3]); pan[m] = torch.flip(pan[m], [2, 3]); ms[m] = torch.flip(ms[m], [2, 3])
    return gt, pan, ms


def build_scheduler(opt, args):
    if args.sched == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    if args.sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)
    if args.sched == "exp":
        gamma = (args.lr_end / args.lr) ** (1.0 / args.epochs)
        print("exp decay: %.3g -> %.3g over %d epochs (gamma=%.6f)"
              % (args.lr, args.lr_end, args.epochs, gamma), flush=True)
        return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    raise ValueError(args.sched)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=460)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optim", choices=["adamw", "adam"], default="adamw")
    p.add_argument("--sched", choices=["step", "cosine", "exp"], default="step")
    p.add_argument("--lr-end", type=float, default=5e-5,
                   help="final LR for --sched exp (exponential decay from --lr)")
    p.add_argument("--step-size", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--arch", choices=["fm", "fm4"], default="fm",
                   help="fm = 8-band WV3 (3 levels); fm4 = 4-band GF2/QB (2 levels)")
    p.add_argument("--c-ms", type=int, default=8)
    p.add_argument("--c-conv", type=int, default=32)
    p.add_argument("--wave", choices=["haar", "split", "poly", "conv", "lofix", "ortho", "lofixo", "lofixn"], default="haar",
                   help="wavelet bank: haar (paper) | split (1D polyphase) | poly (2D polyphase) | conv (learned 2x2)")
    p.add_argument("--fuse", choices=["mamba", "conv"], default="mamba",
                   help="fusion block: FusionMamba (paper) or its conv replacement (Conv-Fuse ablation)")
    p.add_argument("--variant", choices=["full", "coupled", "no_spec", "no_spat", "rev"], default="coupled",
                   help="ablation switch: drop the MS (2D-wavelet) or PAN (1D-channel-wavelet) branch")
    p.add_argument("--agg", choices=["msdg", "fm", "add"], default="fm")
    p.add_argument("--agg-d", type=int, default=84)
    p.add_argument("--agg-depth", type=int, default=0)
    p.add_argument("--mamba-init", choices=["default", "gaussian"], default="default",
                   help="default = keep Mamba's own init; gaussian = old blanket N(0,0.01)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay for a weight-averaged copy; 0 disables")
    p.add_argument("--aug", choices=["none", "flips"], default="none")
    p.add_argument("--gpu-data", type=int, default=1,
                   help="1 = keep the whole training set on the GPU (removes DataLoader overhead)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--max-iters", type=int, default=0, help="stop after N iterations (smoke test)")
    p.add_argument("--resume", default="")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    dev = "cuda"

    if args.gpu_data:
        ds = PansharpeningH5(args.data)
        gt_all = torch.from_numpy(ds.gt).to(dev)
        pan_all = torch.from_numpy(ds.pan).to(dev)
        ms_all = torch.from_numpy(ds.ms).to(dev)
        del ds
        torch.cuda.empty_cache()
        n = gt_all.shape[0]
        idx_gen = torch.Generator(device=dev)
        idx_gen.manual_seed(args.seed)
        loader = None
    else:
        dataset = PansharpeningH5(args.data)
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=True, generator=g)

    Net = WaveletMamba if args.arch == "fm" else WaveletMamba4
    model = Net(C_ms=args.c_ms, H=64, W=64, Scale=4, C_conv=args.c_conv,
                agg=args.agg, agg_d=args.agg_d,
                agg_depth=args.agg_depth, variant=args.variant, fuse=args.fuse, wave=args.wave,
                mamba_init=args.mamba_init).to(dev)
    n_params = sum(q.numel() for q in model.parameters())
    print("params: %d" % n_params, flush=True)

    ema = None
    if args.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
        ema_params = list(ema.parameters())
        model_params = list(model.parameters())
        print("EMA enabled, decay %.4f" % args.ema, flush=True)

    opt_cls = torch.optim.AdamW if args.optim == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args)
    criterion = nn.L1Loss()

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"]
        print("resumed from %s at epoch %d" % (args.resume, start_epoch), flush=True)

    aug_gen = torch.Generator(device=dev)
    aug_gen.manual_seed(args.seed * 1000 + 7)

    log_path = os.path.join(args.out, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["epoch", "lr", "loss", "sec"])

    iters = 0
    ema_steps = 0
    t_all = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        if args.gpu_data:
            perm = torch.randperm(n, generator=idx_gen, device=dev)
            bs = args.batch_size
            batches = ((gt_all[perm[i:i + bs]], pan_all[perm[i:i + bs]], ms_all[perm[i:i + bs]])
                       for i in range(0, n - n % bs, bs))
        else:
            batches = loader
        for gt, pan, ms in batches:
            gt = gt.to(dev, non_blocking=True)
            pan = pan.to(dev, non_blocking=True)
            ms = ms.to(dev, non_blocking=True)
            if args.aug == "flips":
                gt, pan, ms = augment(gt, pan, ms, aug_gen)

            sr = model(ms, pan)
            loss = criterion(sr, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:
                ema_steps += 1
                d = min(args.ema, (1.0 + ema_steps) / (10.0 + ema_steps))
                with torch.no_grad():
                    torch._foreach_mul_(ema_params, d)
                    torch._foreach_add_(ema_params, model_params, alpha=1.0 - d)
            losses.append(loss.item())
            iters += 1
            if args.max_iters and iters >= args.max_iters:
                break

        sched.step()
        dt = time.time() - t0
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        lr_now = opt.param_groups[0]["lr"]
        writer.writerow([epoch + 1, "%.6g" % lr_now, "%.6f" % mean_loss, "%.1f" % dt])
        log_f.flush()
        if (epoch + 1) % args.log_every == 0:
            print("epoch %d/%d  loss %.5f  lr %.3g  %.1fs  elapsed %.1fmin"
                  % (epoch + 1, args.epochs, mean_loss, lr_now, dt,
                     (time.time() - t_all) / 60), flush=True)

        if args.max_iters and iters >= args.max_iters:
            print("max-iters reached", flush=True)
            break

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "epoch": epoch + 1}
            torch.save(ck, os.path.join(args.out, "last.pt"))
            torch.save(ck, os.path.join(args.out, "ep%03d.pt" % (epoch + 1)))
            if ema is not None:
                torch.save({"model": ema.state_dict(), "epoch": epoch + 1,
                            "ema": args.ema}, os.path.join(args.out, "ema.pt"))

    log_f.close()
    print("done in %.1f min" % ((time.time() - t_all) / 60), flush=True)


if __name__ == "__main__":
    main()
