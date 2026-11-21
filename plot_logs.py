# -*- coding: utf-8 -*-
"""Plot the WorldView-3 training curve: training loss, and validation loss if present.

Usage:  python plot_logs.py [--runs runs] [--out figures]

`runs/<run>/log.csv` holds  epoch,lr,loss,sec  where `loss` is the mean L1 training
loss of that epoch.  If `data/wv3_val_loss.csv` (epoch,val_loss) exists, the
validation curve is drawn on the same axes; it is produced by `tools/val_loss.py`
on the machine that holds the checkpoints and the validation split.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_C = "#2a78d6"      # categorical slot 1
VAL_C = "#eb6834"        # categorical slot 2
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
RUN = "wv3_wavelofix_500"


def read_two_col(path, xname, yname):
    xs, ys = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get(xname):
                continue
            xs.append(float(row[xname]))
            ys.append(float(row[yname]))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--val", default="data/wv3_val_loss.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = os.path.join(args.runs, RUN, "log.csv")
    ep_t, loss_t = read_two_col(train, "epoch", "loss")

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(ep_t, loss_t, color=TRAIN_C, linewidth=2.0, label="training",
            solid_capstyle="round")
    ax.annotate("training", xy=(ep_t[-1], loss_t[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK_2)

    if os.path.exists(args.val):
        ep_v, loss_v = read_two_col(args.val, "epoch", "val_loss")
        ax.plot(ep_v, loss_v, color=VAL_C, linewidth=2.0, label="validation",
                solid_capstyle="round")
        ax.annotate("validation", xy=(ep_v[-1], loss_v[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK_2)

    ax.set_yscale("log")
    ax.set_xlabel("epoch", fontsize=9, color=INK_2)
    ax.set_ylabel("mean $\\ell_1$ loss (log scale)", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(ep_t) * 1.10)
    # the data spans less than one decade, so name the ticks explicitly
    ticks = [0.009, 0.010, 0.012, 0.015, 0.020]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["%.3f" % t for t in ticks])
    ax.minorticks_off()
    ax.tick_params(labelsize=8, colors=INK_2, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.grid(axis="y", which="major", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(args.out, "wv3_training_curve." + ext)
        fig.savefig(out, facecolor=SURFACE)
        print("wrote", out)


def plot_val_psnr(out_dir, csv_path, selected=460):
    """Validation PSNR against epoch, with the reported checkpoint marked."""
    if not os.path.exists(csv_path):
        print("missing", csv_path)
        return
    ep, psnr = [], []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            ep.append(int(row["epoch"]))
            psnr.append(float(row["psnr"]))

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(ep, psnr, color=TRAIN_C, linewidth=2.0, solid_capstyle="round")
    if selected in ep:
        i = ep.index(selected)
        ax.plot([ep[i]], [psnr[i]], marker="o", markersize=8, color=TRAIN_C,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate("reported checkpoint\n(epoch %d)" % selected,
                    xy=(ep[i], psnr[i]), xytext=(-14, -46),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=INK_2,
                    arrowprops=dict(arrowstyle="-", color="#b8b7b2", linewidth=0.8,
                                    shrinkA=2, shrinkB=4))

    ax.set_xlabel("epoch", fontsize=9, color=INK_2)
    ax.set_ylabel("validation PSNR (dB)", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(ep) * 1.05)
    ax.tick_params(labelsize=8, colors=INK_2, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, "wv3_val_psnr." + ext)
        fig.savefig(path, facecolor=SURFACE)
        print("wrote", path)


if __name__ == "__main__":
    main()
    plot_val_psnr("figures", os.path.join("data", "wv3_val_psnr.csv"))
