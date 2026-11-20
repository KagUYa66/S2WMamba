# -*- coding: utf-8 -*-
"""Uniform weight averaging of several checkpoints (model soup).

Usage: python avg_ckpt.py --out avg.pt --ckpts ep460.pt ep480.pt ep500.pt
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    a = p.parse_args()

    sds = []
    for f in a.ckpts:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        sds.append(sd)

    keys = list(sds[0].keys())
    for sd in sds[1:]:
        assert set(sd.keys()) == set(keys), "checkpoint key mismatch"
    avg = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in keys}
    torch.save({"model": avg, "epoch": -1}, a.out)
    print("saved %s from %d checkpoints (%d tensors)" % (a.out, len(sds), len(avg)))


if __name__ == "__main__":
    main()
