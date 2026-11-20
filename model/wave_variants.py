# -*- coding: utf-8 -*-
"""Wavelet-transform variants for the ablation study (--wave {haar,split,poly,conv}).

Every call site gets its OWN module. For 'conv' that means one separately
learnable filter bank per decomposition/reconstruction site (the paper's fixed
Haar shares a basis everywhere; a single global learned kernel would handicap
the learned variant, so it is deliberately not done).

- haar  : the fixed orthonormal Haar wavelet (paper).
- split : 1D only -- take every other channel instead of filtering the pair.
          A permutation: zero parameters, exactly invertible.
- poly  : 2D only -- 2x2 polyphase subsampling without filtering.
          A permutation: zero parameters, exactly invertible.
- conv  : learnable 2x2 analysis/synthesis bank with the SAME support and the
          SAME output shapes as Haar, so only "fixed basis" vs "learned basis"
          differs (1D: 2-tap conv along channels; 2D: 2x2 stride-2 conv).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 2D variants
class Poly2D(nn.Module):
    """2x2 polyphase decomposition: pure subsampling, no filtering."""

    def forward(self, x):
        return torch.cat([x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2],
                          x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]], dim=1)


class IDWTPoly2D(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        q = c // 4
        out = x.new_zeros(b, q, h * 2, w * 2)
        out[:, :, 0::2, 0::2] = x[:, 0 * q:1 * q]
        out[:, :, 0::2, 1::2] = x[:, 1 * q:2 * q]
        out[:, :, 1::2, 0::2] = x[:, 2 * q:3 * q]
        out[:, :, 1::2, 1::2] = x[:, 3 * q:4 * q]
        return out


class ConvDWT2D(nn.Module):
    """Learnable 2x2 stride-2 analysis bank (same support/shape as Haar)."""

    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(C, 4 * C, 2, 2, bias=False)

    def forward(self, x):
        return self.conv(x)


class ConvIDWT2D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv = nn.ConvTranspose2d(4 * C, C, 2, 2, bias=False)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------- 1D variants
class Split1D(nn.Module):
    """Take every other channel instead of filtering the pair (permutation)."""

    def forward(self, x):
        return torch.cat([x[:, 0::2], x[:, 1::2]], dim=1)


class IDWTSplit1D(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        q = c // 2
        out = x.new_empty(b, c, h, w)
        out[:, 0::2] = x[:, :q]
        out[:, 1::2] = x[:, q:]
        return out


def _to_seq(x):
    """Move the channel axis (length C) to the sequence axis, as Haar1D does."""
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b * h * w, 1, c), (b, c, h, w)


def _from_seq(x, shape):
    b, c, h, w = shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class ConvDWT1D(nn.Module):
    """Learnable 2-tap bank along the channel axis, laid out exactly like Haar1D.

    Two learned filters replace (1,1)/sqrt2 and (1,-1)/sqrt2; output is
    [low(C/2) | high(C/2)] so analyze_dwt_1d splits it the same way.
    """

    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv1d(1, 2, 2, 2, bias=False)

    def forward(self, x):
        s, shape = _to_seq(x)               # (B*H*W, 1, C)
        s = self.conv(s)                    # (B*H*W, 2, C/2)
        s = s.reshape(shape[0] * shape[2] * shape[3], -1)
        return _from_seq(s, shape)          # [low | high] on the channel axis


class ConvIDWT1D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv = nn.ConvTranspose1d(2, 1, 2, 2, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        s = x.permute(0, 2, 3, 1).reshape(b * h * w, 2, c // 2)
        s = self.conv(s).reshape(b, h, w, c)
        return s.permute(0, 3, 1, 2)


# ------------------------------------------------------- fixed-low, learned-high
class LoFix2D(nn.Module):
    """LL fixed at the Haar 2x2 average; the three detail bands are learnable."""

    def __init__(self, C):
        super().__init__()
        self.hf = nn.Conv2d(C, 3 * C, 2, 2, bias=False)

    def forward(self, x):
        ll = F.avg_pool2d(x, 2) * 2.0          # matches Haar's 0.5*(a+b+c+d)
        return torch.cat([ll, self.hf(x)], dim=1)


class LoFixIDWT2D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.hf = nn.ConvTranspose2d(3 * C, C, 2, 2, bias=False)

    def forward(self, x):
        c = x.shape[1] // 4
        ll, hf = x[:, :c], x[:, c:]
        return F.interpolate(ll, scale_factor=2, mode='nearest') * 0.5 + self.hf(hf)


class LoFix1D(nn.Module):
    """Low group fixed at the Haar low-pass over channel pairs; high is learnable."""

    def __init__(self, C):
        super().__init__()
        self.hf = nn.Conv1d(1, 1, 2, 2, bias=False)
        self.register_buffer('w_lo', torch.tensor([[[0.70710678, 0.70710678]]]))

    def forward(self, x):
        s, shape = _to_seq(x)                  # (B*H*W, 1, C)
        low = F.conv1d(s, self.w_lo, stride=2)
        high = self.hf(s)
        out = torch.cat([low, high], dim=1).reshape(shape[0] * shape[2] * shape[3], -1)
        return _from_seq(out, shape)           # [low(C/2) | high(C/2)]


class LoFixIDWT1D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.hf = nn.ConvTranspose1d(1, 1, 2, 2, bias=False)
        self.register_buffer('w_lo', torch.tensor([[[0.70710678, 0.70710678]]]))

    def forward(self, x):
        b, c, h, w = x.shape
        s = x.permute(0, 2, 3, 1).reshape(b * h * w, 2, c // 2)
        low, high = s[:, 0:1], s[:, 1:2]
        out = F.conv_transpose1d(low, self.w_lo, stride=2) + self.hf(high)
        return out.reshape(b, h, w, c).permute(0, 3, 1, 2)




# ---------------------------------------------- learnable-but-orthogonal basis
def _skew4(th):
    """Batched 4x4 skew-symmetric matrix from 6 params per item."""
    n = th.shape[0]
    S = th.new_zeros(n, 4, 4)
    iu = torch.triu_indices(4, 4, offset=1, device=th.device)
    S[:, iu[0], iu[1]] = th
    return S - S.transpose(1, 2)


def _cayley2(angle):
    """2x2 rotation from one angle."""
    c, s_ = torch.cos(angle), torch.sin(angle)
    return torch.stack([torch.stack([c, -s_], -1), torch.stack([s_, c], -1)], -2)


class OrthoDWT2D(nn.Module):
    """Haar's 2x2 basis replaced by a per-channel learnable rotation.

    R = expm(S) @ H0 with S skew-symmetric, so R is orthogonal by construction
    and init (S=0) reproduces Haar exactly. Same shapes/layout as DWT_2D.
    """

    def __init__(self, C):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(C, 6))
        H = torch.tensor([[[.5, .5], [.5, .5]],
                          [[.5, .5], [-.5, -.5]],
                          [[.5, -.5], [.5, -.5]],
                          [[.5, -.5], [-.5, .5]]]).reshape(4, 4)
        self.register_buffer('H0', H)

    def _R(self):
        return torch.matrix_exp(_skew4(self.theta)) @ self.H0

    def forward(self, x):
        B, C, H, W = x.shape
        blk = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H // 2, W // 2, 4)
        out = torch.einsum('bchwj,ckj->bchwk', blk, self._R())
        return out.permute(0, 4, 1, 2, 3).reshape(B, 4 * C, H // 2, W // 2)


class OrthoIDWT2D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(C, 6))
        H = torch.tensor([[[.5, .5], [.5, .5]],
                          [[.5, .5], [-.5, -.5]],
                          [[.5, -.5], [.5, -.5]],
                          [[.5, -.5], [-.5, .5]]]).reshape(4, 4)
        self.register_buffer('H0', H)

    def _R(self):
        return torch.matrix_exp(_skew4(self.theta)) @ self.H0

    def forward(self, x):
        B, C4, h, w = x.shape
        C = C4 // 4
        sub = x.view(B, 4, C, h, w).permute(0, 2, 3, 4, 1)
        out = torch.einsum('bchwk,ckj->bchwj', sub, self._R())
        return out.reshape(B, C, h, w, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, 2 * h, 2 * w)


class OrthoDWT1D(nn.Module):
    """Per-pair 2x2 rotation over channel pairs; init reproduces Haar 1D."""

    def __init__(self, C):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(C // 2))
        self.register_buffer('M0', torch.tensor([[0.70710678, 0.70710678],
                                                 [0.70710678, -0.70710678]]))

    def forward(self, x):
        B, C, H, W = x.shape
        pr = x.reshape(B, C // 2, 2, H, W).permute(0, 1, 3, 4, 2)
        M = _cayley2(self.theta) @ self.M0
        out = torch.einsum('bphwj,pkj->bphwk', pr, M)
        return out.permute(0, 4, 1, 2, 3).reshape(B, C, H, W)


class OrthoIDWT1D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(C // 2))
        self.register_buffer('M0', torch.tensor([[0.70710678, 0.70710678],
                                                 [0.70710678, -0.70710678]]))

    def forward(self, x):
        B, C, H, W = x.shape
        pr = x.reshape(B, 2, C // 2, H, W).permute(0, 2, 3, 4, 1)
        M = _cayley2(self.theta) @ self.M0
        out = torch.einsum('bphwk,pkj->bphwj', pr, M)
        return out.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)




# ----------------------------- fixed LL + orthogonal HF (2D only; 1D stays lofix)
def _skew3(th):
    n = th.shape[0]
    S = th.new_zeros(n, 3, 3)
    iu = torch.triu_indices(3, 3, offset=1, device=th.device)
    S[:, iu[0], iu[1]] = th
    return S - S.transpose(1, 2)


class LoFixOrtho2D(nn.Module):
    """LL fixed at the Haar average; the three detail rows rotate inside the
    orthogonal complement of LL (SO(3)), so the basis stays orthonormal."""

    def __init__(self, C):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(C, 3))
        H = torch.tensor([[[.5, .5], [.5, .5]],
                          [[.5, .5], [-.5, -.5]],
                          [[.5, -.5], [.5, -.5]],
                          [[.5, -.5], [-.5, .5]]]).reshape(4, 4)
        self.register_buffer('H0', H)

    def _Rhf(self):
        return torch.matrix_exp(_skew3(self.theta)) @ self.H0[1:]

    def forward(self, x):
        B, C, H, W = x.shape
        ll = F.avg_pool2d(x, 2) * 2.0
        blk = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H // 2, W // 2, 4)
        hf = torch.einsum('bchwj,ckj->bchwk', blk, self._Rhf())
        hf = hf.permute(0, 4, 1, 2, 3).reshape(B, 3 * C, H // 2, W // 2)
        return torch.cat([ll, hf], dim=1)


class LoFixOrthoIDWT2D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(C, 3))
        H = torch.tensor([[[.5, .5], [.5, .5]],
                          [[.5, .5], [-.5, -.5]],
                          [[.5, -.5], [.5, -.5]],
                          [[.5, -.5], [-.5, .5]]]).reshape(4, 4)
        self.register_buffer('H0', H)

    def _Rhf(self):
        return torch.matrix_exp(_skew3(self.theta)) @ self.H0[1:]

    def forward(self, x):
        B, C4, h, w = x.shape
        C = C4 // 4
        ll, hf = x[:, :C], x[:, C:]
        hf = hf.view(B, 3, C, h, w).permute(0, 2, 3, 4, 1)
        blk = torch.einsum('bchwk,ckj->bchwj', hf, self._Rhf())
        blk = blk.reshape(B, C, h, w, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, 2 * h, 2 * w)
        return F.interpolate(ll, scale_factor=2, mode='nearest') * 0.5 + blk




# --------------------- fixed LL + HIGH-FREQUENCY FILTERS OF UNIT L2 NORM
class LoFix2DNorm(nn.Module):
    """Like lofix, but every learned detail filter is renormalised to unit
    L2 norm before use, so the analysis keeps Haar's per-filter energy."""

    def __init__(self, C):
        super().__init__()
        self.hf = nn.Conv2d(C, 3 * C, 2, 2, bias=False)

    def forward(self, x):
        w = self.hf.weight
        w = w / (w.reshape(w.shape[0], -1).norm(dim=1).reshape(-1, 1, 1, 1) + 1e-8)
        hf = F.conv2d(x, w, stride=2)
        ll = F.avg_pool2d(x, 2) * 2.0
        return torch.cat([ll, hf], dim=1)


class LoFixIDWT2DNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.hf = nn.ConvTranspose2d(3 * C, C, 2, 2, bias=False)

    def forward(self, x):
        c = x.shape[1] // 4
        ll, hf = x[:, :c], x[:, c:]
        w = self.hf.weight
        w = w / (w.permute(1, 0, 2, 3).reshape(w.shape[1], -1).norm(dim=1)
                 .reshape(1, -1, 1, 1) + 1e-8)
        return F.interpolate(ll, scale_factor=2, mode='nearest') * 0.5             + F.conv_transpose2d(hf, w, stride=2)


class LoFix1DNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.hf = nn.Conv1d(1, 1, 2, 2, bias=False)
        self.register_buffer('w_lo', torch.tensor([[[0.70710678, 0.70710678]]]))

    def forward(self, x):
        s, shape = _to_seq(x)
        w = self.hf.weight
        w = w / (w.norm() + 1e-8)
        low = F.conv1d(s, self.w_lo, stride=2)
        high = F.conv1d(s, w, stride=2)
        out = torch.cat([low, high], dim=1).reshape(shape[0] * shape[2] * shape[3], -1)
        return _from_seq(out, shape)


class LoFixIDWT1DNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.hf = nn.ConvTranspose1d(1, 1, 2, 2, bias=False)
        self.register_buffer('w_lo', torch.tensor([[[0.70710678, 0.70710678]]]))

    def forward(self, x):
        b, c, h, w = x.shape
        s = x.permute(0, 2, 3, 1).reshape(b * h * w, 2, c // 2)
        low, high = s[:, 0:1], s[:, 1:2]
        wf = self.hf.weight
        wf = wf / (wf.norm() + 1e-8)
        out = F.conv_transpose1d(low, self.w_lo, stride=2) + F.conv_transpose1d(high, wf, stride=2)
        return out.reshape(b, h, w, c).permute(0, 3, 1, 2)


# ---------------------------------------------------------------- factories
def make_dwt2d(kind, C):
    if kind == 'haar':
        from model.torch_wavelets import DWT_2D
        return DWT_2D(wave='haar')
    if kind == 'poly':
        return Poly2D()
    if kind == 'conv':
        return ConvDWT2D(C)
    if kind == 'lofix':
        return LoFix2D(C)
    if kind == 'ortho':
        return OrthoDWT2D(C)
    if kind == 'lofixo':
        return LoFixOrtho2D(C)
    if kind == 'lofixn':
        return LoFix2DNorm(C)
    raise ValueError('unknown 2D wavelet variant: %s' % kind)


def make_idwt2d(kind, C):
    if kind == 'haar':
        from model.torch_wavelets import IDWT_2D
        return IDWT_2D(wave='haar')
    if kind == 'poly':
        return IDWTPoly2D()
    if kind == 'conv':
        return ConvIDWT2D(C)
    if kind == 'lofix':
        return LoFixIDWT2D(C)
    if kind == 'ortho':
        return OrthoIDWT2D(C)
    if kind == 'lofixo':
        return LoFixOrthoIDWT2D(C)
    if kind == 'lofixn':
        return LoFixIDWT2DNorm(C)
    raise ValueError('unknown 2D wavelet variant: %s' % kind)


def make_dwt1d(kind, C):
    if kind == 'haar':
        from model.torch_wavelets import DWT_1D
        return DWT_1D(wave='haar')
    if kind == 'split':
        return Split1D()
    if kind == 'conv':
        return ConvDWT1D(C)
    if kind == 'lofix':
        return LoFix1D(C)
    if kind == 'ortho':
        return OrthoDWT1D(C)
    if kind == 'lofixo':
        return LoFix1D(C)
    if kind == 'lofixn':
        return LoFix1DNorm(C)
    raise ValueError('unknown 1D wavelet variant: %s' % kind)


def make_idwt1d(kind, C):
    if kind == 'haar':
        from model.torch_wavelets import IDWT_1D
        return IDWT_1D(wave='haar')
    if kind == 'split':
        return IDWTSplit1D()
    if kind == 'conv':
        return ConvIDWT1D(C)
    if kind == 'lofix':
        return LoFixIDWT1D(C)
    if kind == 'ortho':
        return OrthoIDWT1D(C)
    if kind == 'lofixo':
        return LoFixIDWT1D(C)
    if kind == 'lofixn':
        return LoFixIDWT1DNorm(C)
    raise ValueError('unknown 1D wavelet variant: %s' % kind)


def family(kind):
    """Map --wave to (2d_kind, 1d_kind). 'split' touches 1D only, 'poly' 2D only."""
    return {'haar': ('haar', 'haar'),
            'split': ('haar', 'split'),
            'poly': ('poly', 'haar'),
            'conv': ('conv', 'conv'),
            'lofix': ('lofix', 'lofix'),
            'ortho': ('ortho', 'ortho'),
            'lofixo': ('lofixo', 'lofix'),
            'lofixn': ('lofixn', 'lofixn')}[kind]
