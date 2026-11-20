# BRANCH NAMING -- read before touching this file.
# The module names below are the OPPOSITE of the paper, the figures and the
# previous submission. The paper names a branch after the wavelet it uses, not
# after the modality it carries:
#
#   self.spectral_branch   MS features + 2-D wavelet of PAN, inverse 2-D transform
#                          -> the paper's **Spatial Branch** (output F_sp)
#   self.spatial_branch    PAN features + 1-D channel wavelet of the cue,
#                          inverse 1-D transform
#                          -> the paper's **Spectral Branch** (output F_se)
#
# Hence --variant no_spec removes the paper's SPATIAL branch and --variant no_spat
# removes the paper's SPECTRAL branch. The names are frozen because renaming them
# would invalidate every archived checkpoint; see code/NAMING.md. Do not "fix"
# one side only.
import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from einops import rearrange
# from model.fusion_mamba import FusionMamba
try:
    from fusion_mamba import FusionMamba
except:
    from model.fusion_mamba import FusionMamba
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from torch_wavelets import DWT_2D,IDWT_2D,DWT_1D,IDWT_1D
except:
    from model.torch_wavelets import DWT_2D,IDWT_2D,DWT_1D,IDWT_1D
try:
    from conv_stream import build_fusion
except:
    from model.conv_stream import build_fusion
try:
    from wave_variants import make_dwt2d, make_idwt2d, make_dwt1d, make_idwt1d, family
except:
    from model.wave_variants import make_dwt2d, make_idwt2d, make_dwt1d, make_idwt1d, family


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SCALE_FACTOR = 4
H_DEFAULT = 64
W_DEFAULT = 64
C_MS_DEFAULT = 8
C_CONV_DEFAULT = 32
DEPTH = 0

def analyze_dwt_2d(dwted):
    # 获取变换后的各个分量
    B, C, H, W = dwted.shape
    dwted = dwted.view(B, 4, -1, H, W)
    ll = dwted[:, 0]  # 低频分量
    lh = dwted[:, 1]  # 水平高频
    hl = dwted[:, 2]  # 垂直高频
    hh = dwted[:, 3]  # 对角高频

    # 组合成一个元组
    subbands = (ll, lh, hl, hh)
    return subbands

def analyze_dwt_1d(dwted):
    # get shape of the input tensor
    B, C, H, W = dwted.shape

    assert C % 2 == 0, "Channel dimension must be even!"

    l = dwted[:, 0:C//2, :, :]
    h = dwted[:, C//2:, :, :]

    # cat as a tuple
    subbands = (l, h)
    return subbands

def synthesize_dwt_2d(subbands):
    # ll, lh, hl, hh = subbands
    ll = subbands[0]
    lh = subbands[1]
    hl = subbands[2]
    hh = subbands[3]

    B, C, H, W = ll.shape

    dwted = torch.stack((ll, lh, hl, hh), dim=1)  # [B, 4, C, H, W]
    dwted = dwted.view(B, 4*C, H, W)  # [B, 4*C, H, W]

    return dwted

def synthesize_dwt_1d(subbands):
    # l, h = subbands
    dwted = torch.cat(subbands, dim=1)
    return dwted


class SpectralBranch_Stage(nn.Module):
    def __init__(self, k, H=H_DEFAULT, W=W_DEFAULT, C_conv=C_CONV_DEFAULT, depth=DEPTH, fuse='mamba'):
        super().__init__()
        self.fmamba1 = build_fusion(fuse, C_conv, H//k, W//k, depth=depth, final=True)
        self.fmamba2 = build_fusion(fuse, C_conv, H//k, W//k, depth=depth, final=True)
        self.fmamba3 = build_fusion(fuse, C_conv, H//k, W//k, depth=depth, final=True)
        self.fmamba4 = build_fusion(fuse, C_conv, H//k, W//k, depth=depth, final=True)

    def forward(self, lrms, ddwted):
        # print("ddwted shape:", ddwted.shape)
        ddwted_tuple = analyze_dwt_2d(ddwted)
        ll, lh, hl, hh = ddwted_tuple
        # print("ll shape:", ll.shape)
        # print("lrms shape:", lrms.shape)
        ll_fusioned = self.fmamba1(lrms, ll)
        lh_fusioned = self.fmamba2(lrms, lh)
        hl_fusioned = self.fmamba3(lrms, hl)
        hh_fusioned = self.fmamba4(lrms, hh)

        output = synthesize_dwt_2d((ll_fusioned, lh_fusioned, hl_fusioned, hh_fusioned))
        return output

# class SpectralBranch_Stage2(nn.Module):
#     # H = 64
#     # W = 64
#     def __init__(self, H=64, W=64):
#         self.fmamba1 = build_fusion(fuse, H/2, W/2, depth=1, final=True)
#         self.fmamba2 = build_fusion(fuse, H/2, W/2, depth=1, final=True)
#         self.fmamba3 = build_fusion(fuse, H/2, W/2, depth=1, final=True)
#         self.fmamba4 = build_fusion(fuse, H/2, W/2, depth=1, final=True)

#     def forward(self, temp, dwted):
#         dwted_tuple = analyze_dwt_2d(dwted)
#         ll, lh, hl, hh = dwted_tuple
#         # print("ll shape:", ll.shape)
#         ll_fusioned = self.fmamba1(temp, ll)
#         lh_fusioned = self.fmamba2(temp, lh)
#         hl_fusioned = self.fmamba3(temp, hl)
#         hh_fusioned = self.fmamba4(temp, hh)

#         output = synthesize_dwt_2d((ll_fusioned, lh_fusioned, hl_fusioned, hh_fusioned))
#         return output

class SpectralBranch(nn.Module):
    def __init__(self, H=H_DEFAULT, W=W_DEFAULT, C_ms=C_MS_DEFAULT, C_conv=C_CONV_DEFAULT, depth=DEPTH, fuse='mamba', wave='haar'):
        super().__init__()
        self.conv1 = nn.Conv2d(C_ms, C_conv, 3, 1, 1)
        self.stage1 = SpectralBranch_Stage(k=4, H=H, W=W, C_conv=C_conv, depth=depth, fuse=fuse)
        self.idwt_a = make_idwt2d(wave, C_conv)
        self.idwt_b = make_idwt2d(wave, C_conv)
        self.stage2 = SpectralBranch_Stage(k=2, H=H, W=W, C_conv=C_conv, depth=depth, fuse=fuse)

    def forward(self, ms, ddwted, dwted):
        # ms: (B, C, H/4, W/4)
        # ddwted: (B, C, H/4, W/4)
        # dwted: (B, C, H/2, W/2)
        # print("ms shape:", ms.shape)

        ms = self.conv1(ms)
        temp = self.stage1(ms, ddwted)
        temp = self.idwt_a(temp)
        # print("temp shape:", temp.shape)
        output1 = self.stage2(temp, dwted)
        output1 = self.idwt_b(output1)
        return output1
        # output1: (B, C, H, W)

# The code for the spatial branch is as follows.

class SpacialBranch_Stage(nn.Module):
    # ATTENTION:C represents to the channel of L OR H OR Pan but NOT L cat H.
    def __init__(self, C, H=H_DEFAULT, W=W_DEFAULT, C_conv=C_CONV_DEFAULT, depth=DEPTH, fuse='mamba', wave='haar'):
        super().__init__()
        self.convpan = nn.Sequential(
            nn.Conv2d(C, C_conv, 3, 1, 1),
            # nn.SiLU()
        )
        self.convl = nn.Sequential(
            nn.Conv2d(C, C_conv, 3, 1, 1),
            # nn.SiLU()
        )
        self.convh = nn.Sequential(
            nn.Conv2d(C, C_conv, 3, 1, 1),
            # nn.SiLU()
        )

        self.procpan = nn.Sequential(
            nn.Conv2d(C_conv, C_conv*2, 3, 1, 1),
            # nn.GroupNorm(1, C_conv*2),
            nn.GELU(),
            nn.Conv2d(C_conv*2, C_conv, 1)
        )

        self.procl = nn.Sequential(
            nn.Conv2d(C_conv, C_conv*2, 3, 1, 1),
            # nn.GroupNorm(1, C_conv*2),
            nn.GELU(),
            nn.Conv2d(C_conv*2, C_conv, 1)
        )

        self.proch = nn.Sequential(
            nn.Conv2d(C_conv, C_conv*2, 3, 1, 1),
            # nn.GroupNorm(1, C_conv*2),
            nn.GELU(),
            nn.Conv2d(C_conv*2, C_conv, 1)
        )


        self.fmamba1 = build_fusion(fuse, C_conv, H, W, depth=depth, final=True)
        self.fmamba2 = build_fusion(fuse, C_conv, H, W, depth=depth, final=True)

        # self.doubleconv = nn.Sequential(
        #     nn.Conv2d(C_CONV_DEFAULT*2, C_CONV_DEFAULT, 3, 1, 1),
        #     nn.Conv2d(C_CONV_DEFAULT, C*2, 3, 1, 1)
        # )
        self.convout1 = nn.Conv2d(C_conv, C, 3, 1, 1)
        self.convout2 = nn.Conv2d(C_conv, C, 3, 1, 1)
        self.idwt_1d = make_idwt1d(wave, 2 * C)

    def forward(self, pan, landh):
        # pan: (B, 1, H, W)
        # l: (B, C, H, W)
        # h: (B, C, H, W)
        # print("pan shape:", pan.shape)
        # print("l shape:", l.shape)
        # print("h shape:", h.shape)
        landh_tuple = analyze_dwt_1d(landh)
        l = landh_tuple[0]
        h = landh_tuple[1]

        pan = self.convpan(pan)
        l = self.convl(l)
        h = self.convh(h)

        pan = pan + self.procpan(pan)
        l = l + self.procl(l)
        h = h + self.proch(h)

        output1 = self.fmamba1(pan, l)
        output2 = self.fmamba2(pan, h)

        output1 = self.convout1(output1)
        output2 = self.convout2(output2)

        output = synthesize_dwt_1d((output1, output2))
        output = self.idwt_1d(output)

        return output

class SpacialBranch(nn.Module):
    def __init__(self, H=H_DEFAULT, W=W_DEFAULT, C_conv=C_CONV_DEFAULT, depth=DEPTH, fuse='mamba', wave='haar'):
        super().__init__()
        # self.conv1 = nn.Conv2d(C_pan, C_CONV_DEFAULT, 3, 1, 1)
        self.stage1 = SpacialBranch_Stage(C=1, H=H, W=W, C_conv=C_conv, depth=depth, fuse=fuse, wave=wave)
        self.stage2 = SpacialBranch_Stage(C=2, H=H, W=W, C_conv=C_conv, depth=depth, fuse=fuse, wave=wave)
        self.stage3 = SpacialBranch_Stage(C=4, H=H, W=W, C_conv=C_conv, depth=depth, fuse=fuse)

    def forward(self, pan, subbands1d1, subbands1d2, subbands1d3):

        output1 = self.stage1(pan, subbands1d3)
        output2 = self.stage2(output1, subbands1d2)
        output3 = self.stage3(output2, subbands1d1)

        return output3

class InteractiveEnhanceBlock(nn.Module):
    def __init__(self, C, expansion=4):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(C*2, C * expansion, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(C * expansion, C*2, 1)
        )

    def forward(self, x, y):
        # x: (B, C, H, W)
        # y: (B, C, H, W)
        b, c, h, w = x.shape

        x_skip = x
        y_skip = y
        xy = torch.cat([x, y], dim=1)  
        xy = self.convblock(xy)
        x, y = xy.chunk(2, dim=1)
        x = x + x_skip
        y = y + y_skip

        return x, y


class MultiScaleDynamicGate(nn.Module):
    def __init__(self, C, expansion=12, feature=6):
        super().__init__()

        groups = 1 
        scales = [1, 3]

        self.globalpool = nn.AdaptiveAvgPool2d(1)

        self.pre_norm = nn.GroupNorm(groups, C*2)

        self.scale_conv = nn.ModuleList([
            nn.Conv2d(C*2, C*expansion, k, 1, k//2, groups=groups)
            for k in scales
        ])

        self.scale_process = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(groups, C*expansion),
                nn.GELU(),
                nn.Conv2d(C*expansion, C*feature, 1, groups=groups),
                # nn.GroupNorm(groups, C*feature),
                nn.GELU()
            ) for _ in range(len(scales))
        ])

        self.global_process = nn.Sequential(
                # nn.GroupNorm(groups, C*2),
                # nn.GELU(),
                nn.Conv2d(C*2, C*feature, 1, groups=groups),
                # nn.GroupNorm(groups, C*feature),
                nn.GELU()
        )

        self.y_process = nn.Sequential(  # NOT USED
            nn.Conv2d(C, C*2, 1),
            # nn.InstanceNorm2d(C, affine=True),
            nn.GELU(),
            nn.Conv2d(C*2, C, 1)
        )

        self.aggregate = nn.Sequential(
            # nn.Conv2d(C*6, C*4, 1),
            # nn.LayerNorm([C*4]),
            # nn.GELU(),
            nn.Conv2d(C*feature, C*3, 1)
        )

        self.intenchance = InteractiveEnhanceBlock(C, expansion=4)

        self.tanh = nn.Tanh()
        self.norm_mult = nn.GroupNorm(groups, C)
        self.norm_dec = nn.GroupNorm(groups, C)

    def forward(self, x, y):
        b, c, h, w = x.shape

        x, y = self.intenchance(x, y)

        # 输入拼接和归一化
        xy = torch.cat([x, y], dim=1)
        xy = self.pre_norm(xy)

        global_feat = self.globalpool(xy)
        global_feat = self.global_process(global_feat)
        all_feats = global_feat

        for conv, process in zip(self.scale_conv, self.scale_process):
            feat = conv(xy)
            feat = process(feat)
            all_feats = all_feats + feat

        gates = self.aggregate(all_feats)

        mult_gate, decorate_gate, add_gate = gates.chunk(3, dim=1)
        mult_gate = self.norm_mult(mult_gate)
        decorate_gate = self.norm_dec(decorate_gate)

        mult_gate = self.tanh(mult_gate)
        decorate_gate = self.tanh(decorate_gate) * 0.2

        output = x * (1 + decorate_gate + mult_gate * y) + add_gate

        return output

# Below is the code to construct the entire model which is named WaveletMamba.

class FMAggregator(nn.Module):
    """FMamba aggregator. Both branch outputs pass through one shared 1x1 projection."""

    def __init__(self, C, H, W, D=64, depth=0, fuse='mamba'):
        super().__init__()
        self.pin = nn.Conv2d(C, D, 1)
        self.fm = build_fusion(fuse, D, H, W, depth=depth, final=True)
        self.pout = nn.Conv2d(D, C, 1)

    def forward(self, x, y):
        return self.pout(self.fm(self.pin(x), self.pin(y)))


class ConvFusion(nn.Module):
    """Drop-in replacement for an ablated branch.

    Takes the *same two modalities* the branch consumes (MS-side features and the
    PAN image, concatenated), so the ablation removes the branch's structure --
    wavelet decomposition + FusionMamba -- rather than access to a modality.
    Structure: 1x1 projection -> pre-activation residual block (conv-ReLU-conv)
    -> 1x1 output. C_mid = 2 * C_conv.
    """

    def __init__(self, C_in, C_out, C_mid=None):
        super().__init__()
        C_mid = C_mid or 2 * C_CONV_DEFAULT
        self.proj = nn.Conv2d(C_in, C_mid, 1)
        self.body = nn.Sequential(
            nn.Conv2d(C_mid, C_mid, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C_mid, C_mid, 3, 1, 1),
        )
        self.out = nn.Conv2d(C_mid, C_out, 1)

    def forward(self, ms_feat, pan):
        h = self.proj(torch.cat([ms_feat, pan], dim=1))
        h = h + self.body(h)
        return self.out(h)


class WaveletMamba(nn.Module):
    def __init__(self, C_ms=C_MS_DEFAULT, H=H_DEFAULT, W=W_DEFAULT, Scale=SCALE_FACTOR, C_conv=C_CONV_DEFAULT, depth=DEPTH, agg='msdg', agg_d=64, agg_depth=0, mamba_init='default', variant='coupled', fuse='mamba', wave='haar'):
        super().__init__()
        # ablation switches: 'full' | 'no_spec' (drop the MS/2D-wavelet branch)
        # | 'no_spat' (drop the PAN/1D-channel-wavelet branch)
        self.variant = variant
        # ablations are defined on top of the coupled main model; only
        # variant='full' is the parallel (branch-independent) design
        # variant='rev' reverses the cascade: the spatial branch runs FIRST on
        # ms_skip, and its output conditions the spectral branch's MS input
        self.rev = variant == 'rev'
        self.coupled = variant in ('coupled', 'no_spec', 'no_spat')
        self.use_spec = variant in ('full', 'coupled', 'no_spat', 'rev')
        self.use_spat = variant in ('full', 'coupled', 'no_spec', 'rev')

        self.scale = Scale
        self.upsample4 = nn.Upsample(scale_factor=Scale, mode='bilinear', align_corners=False)

        if self.use_spec:
            w2, w1 = family(wave)
            self.convpan1 = nn.Conv2d(1, C_conv, 3, 1, 1)
            self.dwt_2d_l1 = make_dwt2d(w2, C_conv)
            self.dwt_2d_l2 = make_dwt2d(w2, C_conv)
            self.spectral_branch = SpectralBranch(H=H, W=W, C_ms=C_ms, C_conv=C_conv, depth=depth, fuse=fuse, wave=w2)
            self.convo1 = nn.Conv2d(C_conv, C_ms, 3, 1, 1)
        else:
            self.no_spec = ConvFusion(C_ms + 1, C_ms, 2 * C_conv)

        if self.use_spat:
            w2, w1 = family(wave)
            self.dwt_1d_l1 = make_dwt1d(w1, C_ms)
            self.dwt_1d_l2 = make_dwt1d(w1, C_ms // 2)
            self.dwt_1d_l3 = make_dwt1d(w1, C_ms // 4)
            self.spatial_branch = SpacialBranch(H=H, W=W, C_conv=C_conv, depth=depth, fuse=fuse, wave=w1)
        else:
            self.no_spat = ConvFusion(C_ms + 1, C_ms, 2 * C_conv)

        self.agg_type = agg
        if agg == 'msdg':
            self.gate = MultiScaleDynamicGate(C_ms)
        elif agg == 'fm':
            self.agg = FMAggregator(C_ms, H, W, agg_d, agg_depth, fuse=fuse)
        elif agg == 'add':
            self.agg = None          # ablation: res = output1 + output2
        else:
            raise ValueError('unknown aggregator: %s' % agg)

        self._init_weights(mamba_init)

    def _init_weights(self, mamba_init='default'):
        """Per-module initialization.

        Mamba blocks keep their own initialization (dt_proj's inv_dt schedule,
        in_proj/out_proj defaults) unless mamba_init='gaussian', which reproduces
        the old blanket N(0, 0.01) recipe used for the archived checkpoints.
        """
        if mamba_init == 'gaussian':
            # exact reproduction of the archived recipe: every Conv2d/Linear
            # (Mamba included) gets N(0, 0.01) and zero bias
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if m.weight is not None:
                        nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            return

        mamba_ids = set()
        for m in self.modules():
            if isinstance(m, Mamba):
                for sub in m.modules():
                    mamba_ids.add(id(sub))

        for name, m in self.named_modules():
            if id(m) in mamba_ids:
                continue
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    if name in ('convo1',) or name.endswith('.pout'):
                        # keep the residual branch small at init without zeroing
                        # its gradient path
                        nn.init.normal_(m.weight, 0, 1e-3)
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                                nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _aggregate(self, x, y):
        if self.agg_type == 'msdg':
            return self.gate(x, y)
        if self.agg_type == 'add':
            return x + y
        return self.agg(x, y)

    def _spec(self, ms_in, pan):
        """Spectral branch: MS features refined by the 2D-wavelet bands of PAN."""
        pan_conved = self.convpan1(pan)
        subbands2d1 = self.dwt_2d_l1(pan_conved)
        dwted_tuple = analyze_dwt_2d(subbands2d1)
        ll1 = dwted_tuple[0]
        subbands2d2 = self.dwt_2d_l2(ll1)
        return self.convo1(self.spectral_branch(ms_in, subbands2d2, subbands2d1))

    def _spat(self, pan, cue):
        """Spatial branch: PAN features refined by the 1D-wavelet bands of `cue`."""
        subbands1d1 = self.dwt_1d_l1(cue)
        landh1_tuple = analyze_dwt_1d(subbands1d1)
        l1 = landh1_tuple[0]
        subbands1d2 = self.dwt_1d_l2(l1)
        landh2_tuple = analyze_dwt_1d(subbands1d2)
        l2 = landh2_tuple[0]
        subbands1d3 = self.dwt_1d_l3(l2)
        return self.spatial_branch(pan, subbands1d1, subbands1d2, subbands1d3)

    # def forward(self, pan, ms):
    def forward(self, ms, pan):
        # pan: (B, 1, H, W)
        # ms: (B, C_ms, H/4, W/4)

        ms_skip = self.upsample4(ms)

        if self.rev:
            # reverse cascade: spatial branch first (cue = ms_skip, no
            # cross-modulation), then the spectral branch sees the spatial
            # output added to its own MS input -- the exact mirror of
            # cue = output1 + ms_skip in the forward direction
            output2 = self._spat(pan, ms_skip)
            ms_c = ms + F.interpolate(output2, scale_factor=1.0 / self.scale,
                                      mode='bilinear', align_corners=False)
            output1 = self._spec(ms_c, pan)
        else:
            # DWT2D part
            if self.use_spec:
                output1 = self._spec(ms, pan)
            else:
                output1 = self.no_spec(ms_skip, pan)

            if self.use_spat:
                cue = (output1 + ms_skip) if self.coupled else ms_skip
                output2 = self._spat(pan, cue)
            else:
                output2 = self.no_spat(output1 + ms_skip, pan)

        res = self._aggregate(output1, output2)

        final_output = ms_skip+res

        return final_output


def main():
    Pan = torch.rand(16, 1, 64, 64).to(DEVICE)
    ms = torch.rand(16, 8, 16, 16).to(DEVICE)
    print("input shapes:\nPan:", Pan.shape, "\nms(lrms):", ms.shape)
    model = WaveletMamba()
    model.to(DEVICE)
    print("testing...")
    # output = model(Pan, ms)
    output = model(ms, Pan)
    print("output(HRMS) shape:", output.shape)

if __name__ == "__main__":
    main()
    # summary(model, input_size=(16, 1, 64, 64), device='cuda')