import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

NET_VERSION = "lite_m_clean_keep_gate2_d256_depth6_ssm2_mlp3_drop003_basc_tail3_a005"

try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, getCSM
    from .csms6s import CrossScan, CrossMerge
    from .csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, \
        CrossMerge_Ab_2direction
    from .csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, getCSM
    from csms6s import CrossScan, CrossMerge
    from csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, \
        CrossMerge_Ab_2direction
    from csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x

class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError

class SpectralSE(nn.Module):
    """Lightweight spectral channel attention for HSI."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        w = self.pool(x)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

# =====================================================
# =====================================================
# =====================================================
# APAT tokenizer + semantic-aware dynamic gate
# =====================================================

class MultiScaleSpectralConvBranch(nn.Module):
    """Learnable multi-scale spectral convolution branch for HSI."""

    def __init__(self, out_channels, kernel_size, hidden_channels=24):
        super().__init__()
        padding = kernel_size // 2
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding,
                      groups=hidden_channels, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, C, H, W], where C is the spectral dimension
        B, C, H, W = x.shape
        spectral_seq = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, 1, C)
        feat = self.spectral_conv(spectral_seq)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        feat = feat.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return self.proj(feat)

class HSITokenizer(nn.Module):
    """
    Multi-scale spectral convolution tokenizer for HSI.

    Compared with the original fixed spectral shift aggregation, this version uses
    learnable 1D convolutions along the spectral axis at multiple receptive fields.
    It is still anchored by a raw spectral projection and complemented by a spatial branch.
    """

    def __init__(self, in_chans, embed_dim, patch_size, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.channel_first = channel_first

        dim_small = embed_dim // 3
        dim_mid = embed_dim // 3
        dim_large = embed_dim - dim_small - dim_mid
        hidden_channels = max(embed_dim // 16, 16)

        self.raw_anchor = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.spec_small = MultiScaleSpectralConvBranch(dim_small, kernel_size=3, hidden_channels=hidden_channels)
        self.spec_mid = MultiScaleSpectralConvBranch(dim_mid, kernel_size=5, hidden_channels=hidden_channels)
        self.spec_large = MultiScaleSpectralConvBranch(dim_large, kernel_size=7, hidden_channels=hidden_channels)

        self.spectral_mix = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.detail_dw = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False)
        self.detail_pw = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.detail_bn = nn.BatchNorm2d(embed_dim)

        self.branch_spatial = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, groups=in_chans, bias=False),
            nn.BatchNorm2d(in_chans),
            nn.GELU(),
            SpectralSE(in_chans, reduction=16),
            nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        hidden = max(embed_dim // 2, 32)
        self.anchor_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = norm_layer(embed_dim) if patch_norm else nn.Identity()

    def _spectral_branch(self, x):
        f_small = self.spec_small(x)
        f_mid = self.spec_mid(x)
        f_large = self.spec_large(x)

        x_spec = torch.cat([f_small, f_mid, f_large], dim=1)
        x_spec = self.spectral_mix(x_spec)

        local_detail = x_spec - F.avg_pool2d(x_spec, kernel_size=3, stride=1, padding=1)
        local_detail = self.detail_dw(local_detail)
        local_detail = self.detail_pw(local_detail)
        local_detail = F.gelu(self.detail_bn(local_detail))

        return x_spec + 0.2 * local_detail

    def forward(self, x):
        x_raw = self.raw_anchor(x)
        x_spec = self._spectral_branch(x)

        anchor = self.anchor_gate(torch.cat([x_raw, x_spec], dim=1))
        x_spec = x_raw + anchor * x_spec

        x_spa = self.branch_spatial(x)

        fuse = self.fuse_gate(torch.cat([x_spec, x_spa], dim=1))
        x = x_spec + fuse * x_spa

        x = self.proj(x)
        if not self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x

class CenterContextReadout(nn.Module):
    """
    Explicit center-pixel readout for patch-based classification.

    The classifier no longer produces dense logits for the whole patch. Instead it
    extracts the center token and fuses it with local and global context before
    predicting the label of the center pixel.
    """

    def __init__(self, dim, num_classes, channel_first=False, local_kernel=3, drop=0.0):
        super().__init__()
        self.channel_first = channel_first
        self.local_kernel = local_kernel
        self.norm = nn.LayerNorm(dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 3, dim, bias=False),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        if self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()

        x = self.norm(x)
        B, H, W, C = x.shape
        center_h = H // 2
        center_w = W // 2

        center_feat = x[:, center_h, center_w, :]
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        local_map = F.avg_pool2d(x_nchw, kernel_size=self.local_kernel, stride=1,
                                 padding=self.local_kernel // 2)
        local_feat = local_map[:, :, center_h, center_w]
        global_feat = x_nchw.mean(dim=(2, 3))

        feat = self.fuse(torch.cat([center_feat, local_feat, global_feat], dim=1))
        return self.head(feat)

class LiDARTokenizer(nn.Module):
    """LiDAR tokenizer: multi-scale geometric branches plus morphology branch."""

    def __init__(self, in_chans, embed_dim, patch_size, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        dim_split = embed_dim // 4
        dim_last = embed_dim - 3 * dim_split
        self.branch1 = nn.Conv2d(in_chans, dim_split, kernel_size=1, bias=False)
        self.branch3 = nn.Conv2d(in_chans, dim_split, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_chans, dim_split, kernel_size=5, padding=2, bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_morph = nn.Conv2d(in_chans, dim_last, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = norm_layer(embed_dim) if patch_norm else nn.Identity()

    def forward(self, x):
        x1 = self.branch1(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x_morph = self.branch_morph(self.max_pool(x) - self.avg_pool(x))
        x = torch.cat([x1, x3, x5, x_morph], dim=1)
        x = F.gelu(x)
        x = self.proj(x)
        if not self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x

class DecoupledDynamicGate(nn.Module):
    """
    Global-conditioned local spatial gate (refined).

    Refinements over the previous version:
    1) remove x*y from the global condition to reduce over-amplification of dominant
       co-activated regions;
    2) keep global prior only as a weak bounded bias, so global context mainly
       modulates local evidence instead of directly overriding it;
    3) replace the old balance-style regularization with a softer entropy-oriented
       regularization plus mild spatial smoothness.
    """

    def __init__(self, d_inner):
        super().__init__()
        hidden = max(d_inner // 2, 16)

        self.semantic = nn.Sequential(
            nn.Conv2d(d_inner * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_context = nn.Sequential(
            nn.Conv2d(d_inner * 3, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.gamma_head = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)
        self.beta_head = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)

        self.prior_head = nn.Conv2d(hidden, 2, kernel_size=1, bias=True)
        self.prior_scale = nn.Parameter(torch.tensor(0.10))

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1, bias=True),
        )

        self.last_masks = None

        with torch.no_grad():
            self.prior_head.bias.zero_()
            self.mask_head[-1].bias.zero_()

    def forward(self, x, y):
        diff = torch.abs(x - y)

        # local evidence
        local_feat = self.semantic(torch.cat([x, y, diff], dim=1))

        # global condition without x*y
        gx = self.global_pool(x)
        gy = self.global_pool(y)
        gd = self.global_pool(diff)
        global_feat = self.global_context(torch.cat([gx, gy, gd], dim=1))

        gamma = 1.0 + 0.35 * torch.tanh(self.gamma_head(global_feat))
        beta = 0.35 * torch.tanh(self.beta_head(global_feat))

        conditioned = local_feat * gamma + beta
        conditioned = conditioned * self.channel_gate(conditioned)

        local_logits = self.mask_head(conditioned)

        # weak bounded global prior: assist but not dominate
        weak_prior = torch.tanh(self.prior_head(global_feat))
        mask_logits = local_logits + self.prior_scale * weak_prior

        masks = torch.sigmoid(mask_logits)
        self.last_masks = masks
        return masks

    def regularization_loss(self):
        if self.last_masks is None:
            return None

        masks = self.last_masks.clamp(1e-6, 1 - 1e-6)

        entropy = -(masks * torch.log(masks) + (1 - masks) * torch.log(1 - masks)).mean()
        entropy_reg = torch.log(torch.tensor(2.0, device=masks.device)) - entropy

        smooth_h = (masks[:, :, 1:, :] - masks[:, :, :-1, :]).abs().mean()
        smooth_w = (masks[:, :, :, 1:] - masks[:, :, :, :-1]).abs().mean()

        return 0.25 * entropy_reg + 0.25 * (smooth_h + smooth_w)

class SharedFeatureConstructor(nn.Module):
    """Learnable shared branch constructor for z with disagreement-aware correction."""

    def __init__(self, dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        if channel_first:
            self.proj = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
            self.disagree_proj = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
            self.balance_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(dim * 2, dim, bias=False),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
            self.disagree_proj = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
            self.balance_gate = nn.Sequential(
                nn.Linear(dim * 2, dim, bias=False),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x, y):
        dim = 1 if self.channel_first else -1
        joint = torch.cat([x, y], dim=dim)
        shared = self.proj(joint)

        avg_feat = 0.5 * (x + y)
        disagree = self.disagree_proj(torch.abs(x - y))
        gate = self.balance_gate(joint)

        return shared + gate * avg_feat + (1.0 - gate) * disagree

class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

class SS2Dv2_add:
    def __initv2add__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            use_direction_gate=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
                          ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
                          ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                           cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        # 找到这部分代码 (大约在 600 多行)
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        if self.forward_core is None:
            raise ValueError(f"Unsupported forward_type for Lite-M clean-keep-gate2 SS2D: {forward_type}")
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

# support: v01-v05; v051d,v052d,v052dc;
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    def __initv2__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v2",
            channel_first=False,
            use_direction_gate=False,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forward_go
        self.use_direction_gate = use_direction_gate

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
                          ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
                          ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                           cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        # tags for forward_type 结束之后...
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        if self.forward_core is None:
            raise ValueError(f"Unsupported forward_type for Lite-M clean-keep-gate2 SS2D: {forward_type}")
        k_group = 4

        # 实例化全新的双向解耦门控
        if self.use_direction_gate:
            self.dynamic_gate = DecoupledDynamicGate(d_inner=d_inner)

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj_hsi = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_lidar = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight_hsi = nn.Parameter(torch.stack([t.weight for t in self.x_proj_hsi], dim=0))  # (K, N, inner)
        del self.x_proj_hsi
        self.x_proj_weight_lidar = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj_lidar], dim=0))  # (K, N, inner)
        del self.x_proj_lidar

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs_hsi = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_lidar = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight_hsi = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs_hsi], dim=0))  # (K, inner, rank)
            self.dt_projs_bias_hsi = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_hsi], dim=0))  # (K, inner)
            self.dt_projs_weight_lidar = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs_lidar], dim=0))  # (K, inner, rank)
            self.dt_projs_bias_lidar = nn.Parameter(
                torch.stack([t.bias for t in self.dt_projs_lidar], dim=0))  # (K, inner)
            del self.dt_projs_hsi
            del self.dt_projs_lidar

            # A, D =======================================
            self.A_logs_hsi = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds_hsi = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
            self.A_logs_lidar = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds_lidar = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            y: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        x_proj_weight = self.x_proj_weight_hsi
        y_proj_weight = self.x_proj_weight_lidar
        x_proj_bias = getattr(self, "x_proj_bias", None)
        y_proj_bias = getattr(self, "y_proj_bias", None)
        dt_projs_weight_hsi = self.dt_projs_weight_hsi
        dt_projs_bias_hsi = self.dt_projs_bias_hsi
        dt_projs_weight_lidar = self.dt_projs_weight_lidar
        dt_projs_bias_lidar = self.dt_projs_bias_lidar
        A_logs_hsi = self.A_logs_hsi
        A_logs_lidar = self.A_logs_lidar
        Ds_hsi = self.Ds_hsi
        Ds_lidar = self.Ds_lidar
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs_hsi.shape
        K, D, R = dt_projs_weight_hsi.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        x_xs = CrossScan.apply(x)
        y_xs = CrossScan.apply(y)
        if no_einsum:
            # x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            # dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_xs, x_proj_weight)
            y_dbl = torch.einsum("b k d l, k c d -> b k c l", y_xs, y_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            if y_proj_bias is not None:
                y_dbl = y_dbl + y_proj_bias.view(1, K, -1, 1)
            x_dts, x_Bs, x_Cs = torch.split(x_dbl, [R, N, N], dim=2)
            # Context/shared branch keeps the original full [dt, B, C] projection,
            # but only B is used for one-way compensation.
            _, y_Bs, _ = torch.split(y_dbl, [R, N, N], dim=2)
            x_dts = torch.einsum("b k r l, k d r -> b k d l", x_dts, dt_projs_weight_hsi)

        x_xs = x_xs.view(B, -1, L)

        # Only target/x is scanned. Context/shared branch contributes A and B.
        x_dts = x_dts.contiguous().view(B, -1, L)

        x_As = -torch.exp(A_logs_hsi.to(torch.float))
        y_As = -torch.exp(A_logs_lidar.to(torch.float))
        x_Bs = x_Bs.contiguous().view(B, K, N, L)
        x_Cs = x_Cs.contiguous().view(B, K, N, L)
        x_Ds = Ds_hsi.to(torch.float)
        y_Bs = y_Bs.contiguous().view(B, K, N, L)
        x_delta_bias = dt_projs_bias_hsi.view(-1).to(torch.float)

        if force_fp32:
            x_xs, x_dts, x_Bs, x_Cs, y_Bs = to_fp32(x_xs, x_dts, x_Bs, x_Cs, y_Bs)

        xy_As = (x_As + y_As) / 2

        # 💡 核心逻辑：双向完全解耦的动态特征融合
        if getattr(self, "use_direction_gate", False) and x is not None and y is not None:
            # Keep two-channel gate output. mask_y is not used for forward z_out,
            # but remains in self.dynamic_gate.last_masks for gate regularization.
            gate_masks = self.dynamic_gate(x, y)  # [B, 2, H, W]
            mask_x = gate_masks[:, 0:1, :, :].view(B, 1, 1, L)

            # Dynamic B exchange for one-way target update.
            x_Bs_fused = mask_x * y_Bs + (1.0 - mask_x) * x_Bs
        else:
            # Hard exchange: directly use context/shared B to update target/x.
            x_Bs_fused = y_Bs

        # Only compute target/x output. Reverse context/z scan is removed.
        x_ys: torch.Tensor = selective_scan(
            x_xs, x_dts, xy_As, x_Bs_fused, x_Cs, x_Ds, x_delta_bias, delta_softplus
        ).view(B, K, -1, H, W)

        x_y: torch.Tensor = CrossMerge.apply(x_ys)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__datahsi__", dict(
                A_logs=A_logs_hsi, ctx_A_logs=A_logs_lidar,
                Bs=x_Bs, ctx_Bs=y_Bs, Bs_fused=x_Bs_fused, Cs=x_Cs, Ds=x_Ds,
                us=x_xs, dts=x_dts, delta_bias=x_delta_bias,
                ys=x_ys, y=x_y,
            ))

        x_y = x_y.view(B, -1, H, W)
        if not channel_first:
            x_y = x_y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        x_y = out_norm(x_y)

        return x_y.to(x.dtype) if to_dtype else x_y

    def _context_in_proj(self, y: torch.Tensor):
        """Use only the effective first half of in_proj for the context branch."""
        if self.disable_z:
            return self.in_proj(y)

        out_dim = self.in_proj.weight.shape[0] // 2
        weight = self.in_proj.weight[:out_dim]
        bias = self.in_proj.bias[:out_dim] if self.in_proj.bias is not None else None
        if self.channel_first:
            return F.conv2d(y, weight[:, :, None, None], bias)
        return F.linear(y, weight, bias)

    def forwardv2(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        y = self._context_in_proj(y)
        if not self.disable_z:
            x, x_z = x.chunk(2, dim=(1 if self.channel_first else -1))  # target branch
            if not self.disable_z_act:
                x_z = self.act(x_z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
            y = y.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
            y = self.conv2d(y)  # (b, d, h, w)
        x = self.act(x)
        y = self.act(y)
        x = self.forward_core(x, y)
        x = self.out_act(x)
        if not self.disable_z:
            x = x * x_z
        out_x = self.dropout(self.out_proj(x))
        return out_x

    # =============================================================================================#
    def forward_corev2_none(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=True,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanCore,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2_none(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_corev2_none(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    def forward_go(self, input_x: torch.Tensor, input_y=None):
        if input_y is None:
            return self.forwardv2_none(input_x)
        else:
            return self.forwardv2(input_x, input_y)

class SS2D_add(nn.Module, mamba_init, SS2Dv2_add):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,

        )
        self.__initv2add__(**kwargs)

class SS2D(nn.Module, mamba_init, SS2Dv2):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            use_direction_gate=False,
            # ======================
            **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
            use_direction_gate=use_direction_gate,

        )
        self.__initv2__(**kwargs)

class VSSBlock_add(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            use_direction_gate=False,
            # =============================
            mlp_ratio=3.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D_add(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )
        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# =====================================================
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            use_direction_gate=False,
            # =============================
            mlp_ratio=3.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
                use_direction_gate=use_direction_gate,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input_x: torch.Tensor, input_y: torch.Tensor):
        x = input_x
        y = input_y
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x, y)))
            else:
                x = self.op(self.norm(x), self.norm(y))
                x_mid = input_x + self.drop_path(x)
        else:
            x_mid = input_x
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x_mid + self.drop_path(self.mlp(self.norm2(x_mid)))  # FFN
        return x

    def _forward_none(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input_x: torch.Tensor, input_y=None):
        if self.use_checkpoint and input_y is not None:
            return checkpoint.checkpoint(self._forward, input_x, input_y)
        elif input_y is None:
            return self._forward_none(input_x)
        else:
            return self._forward(input_x, input_y)

class VSSM(nn.Module):
    def __init__(
            self,
            input_shape=(32, 32),
            patch_size=2,
            in_chans_hsi=144,
            in_chans_lidar=1,
            num_classes=1000,
            depths=[6],
            dims=256,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            use_direction_gate=False,
            mlp_ratio=3.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            drop_path_rate=0.03,
            patch_norm=True,
            norm_layer="LN",
            downsample_version: str = "v_my",
            patchembed_version: str = "v1",
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_features = dims
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.patch_embed_hsi = HSITokenizer(in_chans_hsi, dims, patch_size, patch_norm, norm_layer,
                                            channel_first=self.channel_first)
        self.patch_embed_lidar = LiDARTokenizer(in_chans_lidar, dims, patch_size, patch_norm, norm_layer,
                                                channel_first=self.channel_first)
        self.shared_constructor = SharedFeatureConstructor(dims, channel_first=self.channel_first)

        self.layers_x = nn.ModuleList()
        self.layers_y = nn.ModuleList()
        self.layers_z = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers_x.append(self._make_layer(
                dim=self.dims,
                drop_path=dpr[0:sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=None,
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                use_direction_gate=use_direction_gate,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))
            self.layers_y.append(self._make_layer(
                dim=self.dims,
                drop_path=dpr[0:sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=None,
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                use_direction_gate=False,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))
            self.layers_z.append(self._make_layer_add(
                dim=self.dims,
                drop_path=dpr[0:sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=None,
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        self.classifier_hsi = CenterContextReadout(
            self.num_features, num_classes, channel_first=self.channel_first, local_kernel=3, drop=mlp_drop_rate
        )
        self.classifier_lidar = CenterContextReadout(
            self.num_features, num_classes, channel_first=self.channel_first, local_kernel=3, drop=mlp_drop_rate
        )
        self.classifier_add = CenterContextReadout(
            self.num_features, num_classes, channel_first=self.channel_first, local_kernel=3, drop=mlp_drop_rate
        )

        # Tail BASC-R: Block-Adaptive Shared Constructor Residual for the last blocks only.
        # The first blocks keep the original clean shared_constructor unchanged to protect
        # low-level boundary/geometry cues. Only the last 3 VSS blocks get a small
        # block-specific correction, with alpha bounded to (0, 0.05) and initialized
        # to 0.025. Save/restore CPU RNG so adding these modules does not perturb the
        # initialization sequence of the original clean model modules.
        rng_state = torch.get_rng_state()
        self.total_depth = sum(depths)
        self.basc_active_blocks = min(3, self.total_depth)
        self.basc_start_block = self.total_depth - self.basc_active_blocks
        self.shared_constructors = nn.ModuleList([
            SharedFeatureConstructor(dims, channel_first=self.channel_first)
            for _ in range(self.basc_active_blocks)
        ])
        self.shared_alpha_logits = nn.Parameter(torch.zeros(self.basc_active_blocks))
        torch.set_rng_state(rng_state)

        self.last_gate_reg = None
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_layer_add(
            dim=96, drop_path=[0.1, 0.1], use_checkpoint=False, norm_layer=nn.LayerNorm,
            downsample=nn.Identity(), channel_first=False, ssm_d_state=16, ssm_ratio=2.0,
            ssm_dt_rank="auto", ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2", use_direction_gate=False,
            mlp_ratio=3.0, mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0, gmlp=False, **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock_add(
                hidden_dim=dim, drop_path=drop_path[d], norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer, ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate, ssm_init=ssm_init, forward_type=forward_type,
                mlp_ratio=mlp_ratio, mlp_act_layer=mlp_act_layer, mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp, use_checkpoint=use_checkpoint,
            ))
        return nn.ModuleList(blocks)

    @staticmethod
    def _make_layer(
            dim=96, drop_path=[0.1, 0.1], use_checkpoint=False, norm_layer=nn.LayerNorm,
            downsample=nn.Identity(), channel_first=False, ssm_d_state=16, ssm_ratio=2.0,
            ssm_dt_rank="auto", ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2", use_direction_gate=False,
            mlp_ratio=3.0, mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0, gmlp=False, **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, drop_path=drop_path[d], norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer, ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate, ssm_init=ssm_init, forward_type=forward_type,
                use_direction_gate=use_direction_gate,
                mlp_ratio=mlp_ratio, mlp_act_layer=mlp_act_layer, mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp, use_checkpoint=use_checkpoint,
            ))
        return nn.ModuleList(blocks)

    def _construct_block_shared(self, x, y, block_idx):
        """Construct z with a tail-only block-adaptive correction.

        Early blocks use the original clean shared constructor without any extra
        correction. The last few blocks use a small residual interpolation toward
        a block-specific constructor. alpha is bounded to (0, 0.05) and initialized
        to 0.025, so the new path cannot strongly overwrite the original clean fusion path.
        """
        z_base = self.shared_constructor(x, y)
        if block_idx < self.basc_start_block or self.basc_active_blocks == 0:
            return z_base

        local_idx = block_idx - self.basc_start_block
        z_layer = self.shared_constructors[local_idx](x, y)
        alpha = 0.05 * torch.sigmoid(self.shared_alpha_logits[local_idx])
        return z_base + alpha * (z_layer - z_base)

    def _collect_gate_reg(self):
        regs = []
        for layer in self.layers_x:
            for block in layer:
                op = getattr(block, 'op', None)
                if op is not None and getattr(op, 'dynamic_gate', None) is not None:
                    reg = op.dynamic_gate.regularization_loss()
                    if reg is not None:
                        regs.append(reg)
        if regs:
            self.last_gate_reg = torch.stack(regs).mean()
        else:
            self.last_gate_reg = None

    def get_gate_regularization_loss(self):
        return self.last_gate_reg

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.patch_embed_hsi(x)
        y = self.patch_embed_lidar(y)
        z = self.shared_constructor(x, y)

        block_idx = 0
        for (layer_x, layer_y, layer_z) in zip(self.layers_x, self.layers_y, self.layers_z):
            for block_x, block_y, block_z in zip(layer_x, layer_y, layer_z):
                x = block_x(x, z)
                y = block_y(y, z)
                z = block_z(self._construct_block_shared(x, y, block_idx))
                block_idx += 1

        self._collect_gate_reg()
        x = self.classifier_hsi(x)
        y = self.classifier_lidar(y)
        z = self.classifier_add(z)
        return x, y, z

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)

