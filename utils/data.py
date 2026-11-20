# -*- coding: utf-8 -*-
"""Minimal PanCollection reader: ms, pan, gt only."""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PansharpeningH5(Dataset):
    def __init__(self, path, maxv=2047.0):
        with h5py.File(path, "r") as h:
            self.ms = np.asarray(h["ms"][...], dtype=np.float32) / maxv
            self.pan = np.asarray(h["pan"][...], dtype=np.float32) / maxv
            self.gt = np.asarray(h["gt"][...], dtype=np.float32) / maxv
        assert self.ms.shape[0] == self.pan.shape[0] == self.gt.shape[0]

    def __len__(self):
        return self.gt.shape[0]

    def __getitem__(self, i):
        return (torch.from_numpy(self.gt[i]),
                torch.from_numpy(self.pan[i]),
                torch.from_numpy(self.ms[i]))
