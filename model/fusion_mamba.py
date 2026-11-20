import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None,None,None

class SingleMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=32, bimamba_type='v6', 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        # output = self.norm1(output)
        return output + skip


class CrossMambaBlock(nn.Module):
    def __init__(self, dim, H, W, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v7', 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        # output = self.norm2(output)
        if self.use_skip:
            output = output + skip
        return output


class FusionMamba(nn.Module):
    def __init__(self, dim, H, W, depth=0, final=False):
        super().__init__()
        self.final = final

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.spa_mamba_layers = nn.ModuleList([])
        self.spe_mamba_layers = nn.ModuleList([])
        for _ in range(depth):
            self.spa_mamba_layers.append(SingleMambaBlock(dim, H, W))
            self.spe_mamba_layers.append(SingleMambaBlock(dim, H, W))
        self.spa_cross_mamba = CrossMambaBlock(dim, H, W)
        self.spe_cross_mamba = CrossMambaBlock(dim, H, W)

        self.inproj1 = nn.Sequential(
            nn.Linear(dim ,dim*2),
            # nn.LeakyReLU(0.1),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )

        self.inproj2 = nn.Sequential(
            nn.Linear(dim ,dim*2),
            # nn.LeakyReLU(0.1),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )

        # self.out_proj = nn.Sequential(
        #     nn.Linear(dim*2, dim*4),
        #     nn.SiLU(),
        #     nn.Linear(dim*4, dim)
        # )

        # self.out_proj = NimabiProduct(dim)
        # self.out_proj1 = DynamicGate(dim)
        # self.out_proj2 = DynamicGate(dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.outproj = MixGate(dim)
        self.skip_selection = nn.Parameter(torch.zeros(1))

    def forward(self, pan, ms):
        b, c, h, w = pan.shape
        pan = rearrange(pan, 'b c h w -> b (h w) c', h=h, w=w)
        ms = rearrange(ms, 'b c h w -> b (h w) c', h=h, w=w)

        pan_skip = pan
        ms_skip = ms

        pan = self.norm1(pan)
        ms = self.norm2(ms)

        pan = self.inproj1(pan)
        ms = self.inproj2(ms)

        for spa_layer, spe_layer in zip(self.spa_mamba_layers, self.spe_mamba_layers):
            pan = spa_layer(pan)
            ms = spe_layer(ms)

        pan = pan+pan_skip
        ms = ms+ms_skip
        
        spa_fusion = self.spa_cross_mamba(pan, ms)
        spe_fusion = self.spe_cross_mamba(ms, pan)

        skip_weight = self.sigmoid(self.skip_selection)
        skip = pan * skip_weight + ms * (1 - skip_weight)
        fusion = spa_fusion + spe_fusion + skip
        pan = rearrange(pan, 'b (h w) c -> b c h w', h=h, w=w)
        ms = rearrange(ms, 'b (h w) c -> b c h w', h=h, w=w)
        output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
        if self.final:
            return output
        else:
            return (pan + output) / 2, (ms + output) /2
