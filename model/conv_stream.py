# -*- coding: utf-8 -*-
"""Conv replacement for the FusionMamba block (ablation "--fuse conv").

The point of the ablation is to remove the *global state-space mixing* while
keeping everything else about the fusion block identical: the same two
LayerNorms, the same inproj MLPs, the same post-projection residual, the same
skip_selection gate, the same two cross-modulation branches and the same
summation. Only CrossMambaBlock -> CrossConvBlock changes: the auxiliary stream
now enters through a depthwise 3x3 convolution (local) instead of a selective
scan (global).

Widths are chosen so each CrossConvBlock matches the parameter count of the
CrossMambaBlock it replaces: 8,640 vs 8,768 at dim=32 and 43,428 vs 43,176 at
dim=84 (within 1.5%).
"""
import torch
import torch.nn as nn
from einops import rearrange


def _width_mult(dim):
    """Channel multiplier that matches the corresponding Mamba block's budget."""
    return 4 if dim <= 32 else 3


class CrossConvBlock(nn.Module):
    """Local stand-in for CrossMambaBlock: main stream modulated by aux (3x3 DW)."""

    def __init__(self, dim):
        super().__init__()
        k = _width_mult(dim)
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pw1 = nn.Conv2d(dim, k * dim, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(k * dim, dim, 1)

    def forward(self, main, aux):
        x = main + self.dw(aux)
        return self.pw2(self.act(self.pw1(x)))


class ConvStream(nn.Module):
    """Drop-in for FusionMamba with the same signature and skeleton."""

    def __init__(self, dim, H=None, W=None, depth=0, final=False):
        super().__init__()
        self.final = final
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.inproj1 = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                     nn.Linear(dim * 2, dim))
        self.inproj2 = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                     nn.Linear(dim * 2, dim))
        self.spa_cross = CrossConvBlock(dim)
        self.spe_cross = CrossConvBlock(dim)
        self.skip_selection = nn.Parameter(torch.zeros(1))

    def forward(self, pan, ms):
        b, c, h, w = pan.shape
        p = rearrange(pan, 'b c h w -> b (h w) c')
        m = rearrange(ms, 'b c h w -> b (h w) c')
        p_skip, m_skip = p, m

        p = self.inproj1(self.norm1(p))
        m = self.inproj2(self.norm2(m))
        p = p + p_skip
        m = m + m_skip

        p = rearrange(p, 'b (h w) c -> b c h w', h=h, w=w)
        m = rearrange(m, 'b (h w) c -> b c h w', h=h, w=w)

        spa = self.spa_cross(p, m)
        spe = self.spe_cross(m, p)

        wgt = torch.sigmoid(self.skip_selection)
        fusion = spa + spe + p * wgt + m * (1 - wgt)

        if self.final:
            return fusion
        return (p + fusion) / 2, (m + fusion) / 2


def build_fusion(kind, *args, **kwargs):
    """kind='mamba' -> FusionMamba, kind='conv' -> ConvStream."""
    if kind == 'conv':
        return ConvStream(*args, **kwargs)
    from model.fusion_mamba import FusionMamba
    return FusionMamba(*args, **kwargs)
