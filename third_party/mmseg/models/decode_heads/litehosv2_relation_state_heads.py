# Copyright (c) OpenMMLab. All rights reserved.
"""LiteHOSV2 V0 relation-state slow/fast/gate heads."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MessageHub
from torch import Tensor

from mmseg.registry import MODELS
from ..utils.relation_state import (
    LiteHOSRelationState,
    compute_slot_prototypes,
    relation_violation_score,
)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DWConvBNAct(nn.Sequential):
    """Depthwise-separable conv block for high-resolution relation features."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1):
        padding = (kernel_size // 2) * int(dilation)
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding=padding,
                dilation=int(dilation),
                groups=in_channels,
                bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class CoarseContactGuidedAggregator(nn.Module):
    """Use coarse/contact cues to gate relation feature aggregation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.contact_head = nn.Sequential(
            ConvBNAct(channels + 1, channels, 3),
            nn.Conv2d(channels, 1, 1),
        )
        self.gate = nn.Sequential(
            ConvBNAct(channels + 2, channels, 3),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvBNAct(channels + 2, channels, 3),
            ConvBNAct(channels, channels, 3),
        )

    def forward(self, pixel_feat: Tensor,
                coarse_prob: Tensor) -> Tuple[Tensor, Tensor]:
        contact_logits = self.contact_head(torch.cat([pixel_feat, coarse_prob], dim=1))
        contact_prob = contact_logits.sigmoid()
        guidance = torch.cat([pixel_feat, coarse_prob, contact_prob], dim=1)
        gate = self.gate(guidance)
        relation_feat = self.refine(
            torch.cat([pixel_feat * (1.0 + gate), coarse_prob, contact_prob], dim=1))
        return relation_feat, contact_logits


class CoarseReliabilityHead(nn.Module):
    """Predict whether the coarse concept prior is locally trustworthy."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(channels + 2, channels, 3),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, pixel_feat: Tensor, coarse_prob: Tensor,
                uncertainty: Tensor) -> Tensor:
        return self.head(torch.cat([pixel_feat, coarse_prob, uncertainty], dim=1))


class CoarseReliabilityGuidedAggregator(nn.Module):
    """Efficiently aggregate relation features with trusted/uncertain priors.

    Dense 3x3 convolutions at stride-4 resolution are expensive. This block keeps
    the trusted and uncertain coarse streams, but uses depthwise-separable
    dilated convolutions so the prior can see wider context with a much smaller
    inference cost.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.base = DWConvBNAct(channels, channels, 3, dilation=1)
        self.trusted = DWConvBNAct(channels + 1, channels, 3, dilation=2)
        self.uncertain = DWConvBNAct(channels + 1, channels, 3, dilation=3)
        self.gate = nn.Sequential(
            DWConvBNAct(channels + 3, channels, 3, dilation=2),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.fuse = DWConvBNAct(channels + 3, channels, 3, dilation=2)

    def forward(self,
                pixel_feat: Tensor,
                coarse_prob: Tensor,
                uncertainty: Tensor,
                reliability: Tensor) -> Tensor:
        trust = (coarse_prob * reliability).clamp(0.0, 1.0)
        unknown = (coarse_prob * uncertainty).clamp(0.0, 1.0)
        base = self.base(pixel_feat)
        trusted = self.trusted(torch.cat([pixel_feat * trust, trust], dim=1))
        uncertain = self.uncertain(
            torch.cat([pixel_feat * unknown, unknown], dim=1))
        gate = self.gate(
            torch.cat([pixel_feat, coarse_prob, uncertainty, reliability], dim=1))
        relation_seed = base + gate * trusted + (1.0 - gate) * uncertain
        return self.fuse(
            torch.cat(
                [relation_seed, coarse_prob, uncertainty, reliability],
                dim=1))


class LegacyDenseCoarseReliabilityGuidedAggregator(nn.Module):
    """Original dense-conv CRGA kept for controlled latency/accuracy comparison."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.base = ConvBNAct(channels, channels, 3)
        self.trusted = nn.Sequential(
            ConvBNAct(channels + 1, channels, 3),
            ConvBNAct(channels, channels, 3),
        )
        self.uncertain = nn.Sequential(
            ConvBNAct(channels + 1, channels, 3),
            ConvBNAct(channels, channels, 3),
        )
        self.gate = nn.Sequential(
            ConvBNAct(channels + 3, channels, 3),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(channels + 3, channels, 3),
            ConvBNAct(channels, channels, 3),
        )

    def forward(self,
                pixel_feat: Tensor,
                coarse_prob: Tensor,
                uncertainty: Tensor,
                reliability: Tensor) -> Tensor:
        trust = (coarse_prob * reliability).clamp(0.0, 1.0)
        unknown = (coarse_prob * uncertainty).clamp(0.0, 1.0)
        base = self.base(pixel_feat)
        trusted = self.trusted(torch.cat([pixel_feat * trust, trust], dim=1))
        uncertain = self.uncertain(
            torch.cat([pixel_feat * unknown, unknown], dim=1))
        gate = self.gate(
            torch.cat([pixel_feat, coarse_prob, uncertainty, reliability], dim=1))
        relation_seed = base + gate * (trusted + uncertain)
        return self.fuse(
            torch.cat(
                [relation_seed, coarse_prob, uncertainty, reliability],
                dim=1))


class SCSEBlock(nn.Module):
    """Lightweight spatial and channel squeeze-excitation block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        hidden = max(self.channels // int(reduction), 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, self.channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.channel_gate(x) + x * self.spatial_gate(x)


class ECABlock(nn.Module):
    """Efficient channel attention with a tiny 1D channel filter."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        weights = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        weights = self.conv(weights).transpose(1, 2).unsqueeze(-1).sigmoid()
        return x * weights


class SpatialGateBlock(nn.Module):
    """CBAM-style spatial gate without channel MLP."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        del channels
        padding = int(kernel_size) // 2
        self.conv = nn.Conv2d(2, 1, int(kernel_size), padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        return x * self.conv(torch.cat([avg_map, max_map], dim=1)).sigmoid()


class CBAMLiteBlock(nn.Module):
    """Small serial channel+spatial attention block."""

    def __init__(self,
                 channels: int,
                 reduction: int = 16,
                 kernel_size: int = 7) -> None:
        super().__init__()
        channels = int(channels)
        hidden = max(channels // int(reduction), 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial_gate = SpatialGateBlock(channels, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        avg_weights = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
        max_weights = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        x = x * (avg_weights + max_weights).sigmoid()
        return self.spatial_gate(x)


class CoordAttLiteBlock(nn.Module):
    """Coordinate attention using height/width pooled context."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        channels = int(channels)
        hidden = max(8, channels // int(reduction))
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.attn_h = nn.Conv2d(hidden, channels, 1, bias=False)
        self.attn_w = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        pooled_h = x.mean(dim=3, keepdim=True)
        pooled_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        context = self.shared(torch.cat([pooled_h, pooled_w], dim=2))
        context_h, context_w = torch.split(context, [h, w], dim=2)
        attn_h = self.attn_h(context_h).sigmoid()
        attn_w = self.attn_w(context_w.transpose(2, 3)).sigmoid()
        return x * attn_h * attn_w


class SimAMBlock(nn.Module):
    """Parameter-free spatial-channel energy attention."""

    def __init__(self, channels: int, eps: float = 1e-4) -> None:
        super().__init__()
        del channels
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        centered = (x - mean).pow(2)
        denom = 4.0 * (
            centered.sum(dim=(2, 3), keepdim=True) /
            max(x.shape[-2] * x.shape[-1] - 1, 1) + self.eps)
        return x * torch.sigmoid(centered / denom + 0.5)


class PIDLiteBoundaryMixer(nn.Module):
    """PIDNet-inspired local/context/boundary feature mixer.

    P branch preserves local detail, I branch injects pooled context, and D
    branch predicts an edge attention map used by a Light-Bag style fusion.
    """

    def __init__(self, channels: int, context_kernel: int = 5) -> None:
        super().__init__()
        channels = int(channels)
        self.p_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.i_reduce = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=int(context_kernel), stride=2,
                padding=int(context_kernel) // 2),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        hidden = max(channels // 4, 16)
        self.d_branch = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.conv_p = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        p_feat = self.p_branch(x)
        i_feat = self.i_reduce(x)
        if i_feat.shape[-2:] != x.shape[-2:]:
            i_feat = F.interpolate(
                i_feat, size=x.shape[-2:], mode="bilinear",
                align_corners=False)
        edge_att = self.d_branch(x).sigmoid()
        p_add = self.conv_p((1.0 - edge_att) * i_feat + p_feat)
        i_add = self.conv_i(i_feat + edge_att * p_feat)
        return self.out_act(x + p_add + i_add)


class PID3LightBagAggregator(nn.Module):
    """PIDNet-style three-branch aggregation for relation features.

    P preserves local detail from the fused pixel feature, I carries the
    relation/context feature, and D predicts an edge attention map that gates a
    Light-Bag fusion before slot decoding.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.p_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.i_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        hidden = max(channels // 4, 16)
        self.d_branch = nn.Sequential(
            ConvBNAct(channels + 1, hidden, 3),
            nn.Conv2d(hidden, 1, 1),
        )
        self.conv_p = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_act = nn.GELU()

    def forward(self,
                pixel_feat: Tensor,
                relation_feat: Tensor,
                coarse_prob: Tensor) -> Tuple[Tensor, Tensor]:
        p_feat = self.p_branch(pixel_feat)
        i_feat = self.i_branch(relation_feat)
        if i_feat.shape[-2:] != relation_feat.shape[-2:]:
            i_feat = F.interpolate(
                i_feat,
                size=relation_feat.shape[-2:],
                mode="bilinear",
                align_corners=False)
        edge_logits = self.d_branch(torch.cat([relation_feat, coarse_prob], dim=1))
        edge_att = edge_logits.sigmoid()
        p_add = self.conv_p((1.0 - edge_att) * i_feat + p_feat)
        i_add = self.conv_i(i_feat + edge_att * p_feat)
        return self.out_act(relation_feat + p_add + i_add), edge_logits


def build_lite_attention(name: Optional[str], channels: int) -> Optional[nn.Module]:
    if name is None:
        return None
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized in ("", "none", "identity"):
        return None
    if normalized == "eca":
        return ECABlock(channels)
    if normalized in ("cbam", "cbam_lite"):
        return CBAMLiteBlock(channels)
    if normalized in ("coordatt", "coordatt_lite", "coordinate"):
        return CoordAttLiteBlock(channels)
    if normalized == "simam":
        return SimAMBlock(channels)
    if normalized in ("pid_lite", "pidlite", "pid_boundary", "pid_boundary_lite"):
        return PIDLiteBoundaryMixer(channels)
    if normalized in ("spatial", "spatial_gate", "spatialgate"):
        return SpatialGateBlock(channels)
    raise ValueError(f"Unknown lite attention module: {name}")


class WindowPixelSelfAttention(nn.Module):
    """Local pixel self-attention over non-overlapping windows.

    This is intentionally independent of the coarse mask: it refines pixel
    features with local context without using coarse as a hard attention guide.
    """

    def __init__(self,
                 channels: int,
                 num_heads: int = 4,
                 window_size: int = 7,
                 mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.norm1 = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(
            self.channels, int(num_heads), dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(self.channels)
        hidden = int(round(self.channels * float(mlp_ratio)))
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        ws = max(int(self.window_size), 1)
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        tokens = x.permute(0, 2, 3, 1).reshape(
            b, hp // ws, ws, wp // ws, ws, c)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
        attn_in = self.norm1(tokens)
        tokens = tokens + self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        out = tokens.reshape(b, hp // ws, wp // ws, ws, ws, c)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
        out = out[:, :h, :w].permute(0, 3, 1, 2).contiguous()
        return out


class SimpleGate(nn.Module):
    """Split-channel gate used by lightweight ConvNeXt-style blocks."""

    def forward(self, x: Tensor) -> Tensor:
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class CoarsePixelInteractionMixer(nn.Module):
    """Let coarse probability interact with pixel features as a residual cue."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.channels = int(channels)
        padding = int(kernel_size) // 2
        self.mix = nn.Sequential(
            nn.Conv2d(self.channels + 1, self.channels * 2, 1, bias=False),
            nn.BatchNorm2d(self.channels * 2),
            nn.GELU(),
            nn.Conv2d(
                self.channels * 2,
                self.channels * 2,
                int(kernel_size),
                padding=padding,
                groups=self.channels * 2,
                bias=False),
            nn.BatchNorm2d(self.channels * 2),
            SimpleGate(),
            nn.Conv2d(self.channels, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
        )

    def forward(self, pixel_feat: Tensor, coarse_prob: Tensor) -> Tensor:
        return pixel_feat + self.mix(torch.cat([pixel_feat, coarse_prob], dim=1))


def _fixed5_topology_allowed(pattern: str) -> Tensor:
    """Return the frozen six-token adjacency used by the TCSVT controls."""
    pattern = str(pattern).strip().lower()
    if pattern not in ("matched", "crossed"):
        raise ValueError(
            "topology_pattern must be 'matched' or 'crossed', "
            f"got {pattern!r}")
    # token order: LH, RH, LO, RO, TO, G.  The crossed control changes only
    # the hand-object correspondence and therefore preserves graph density.
    paired_token = (2, 3, 0, 1) if pattern == "matched" else (3, 2, 1, 0)
    edges = {
        0: (0, paired_token[0], 4, 5),
        1: (1, paired_token[1], 4, 5),
        2: (2, paired_token[2], 4, 5),
        3: (3, paired_token[3], 4, 5),
        4: (0, 1, 2, 3, 4, 5),
        5: (0, 1, 2, 3, 4, 5),
    }
    allowed = torch.zeros(6, 6, dtype=torch.bool)
    for src, dsts in edges.items():
        allowed[src, list(dsts)] = True
    return allowed


class HardTopologyTokenMixer(nn.Module):
    """Fixed semantic-token topology for LH/RH/LO/RO/TO plus a global token."""

    def __init__(self,
                 channels: int,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 topology_pattern: str = "matched") -> None:
        super().__init__()
        self.channels = int(channels)
        self.global_token = nn.Parameter(torch.zeros(1, 1, self.channels))
        nn.init.normal_(self.global_token, std=0.02)
        self.norm1 = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(
            self.channels, int(num_heads), dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(self.channels)
        hidden = int(round(self.channels * float(mlp_ratio)))
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
        )
        allowed = _fixed5_topology_allowed(topology_pattern)
        self.register_buffer("attn_mask", ~allowed, persistent=False)

    def forward(self, slot_tokens: Tensor) -> Tensor:
        batch_size = slot_tokens.shape[0]
        global_token = self.global_token.expand(batch_size, -1, -1)
        tokens = torch.cat([slot_tokens, global_token], dim=1)
        normed = self.norm1(tokens)
        tokens = tokens + self.attn(
            normed,
            normed,
            normed,
            attn_mask=self.attn_mask,
            need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens[:, :slot_tokens.shape[1]]


class EdgeGatedTopologyTokenMixer(nn.Module):
    """Fixed relation topology with learnable edge biases."""

    def __init__(self,
                 channels: int,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 topology_pattern: str = "matched") -> None:
        super().__init__()
        self.channels = int(channels)
        self.global_token = nn.Parameter(torch.zeros(1, 1, self.channels))
        nn.init.normal_(self.global_token, std=0.02)
        self.norm1 = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(
            self.channels, int(num_heads), dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(self.channels)
        hidden = int(round(self.channels * float(mlp_ratio)))
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
        )
        allowed = _fixed5_topology_allowed(topology_pattern)
        self.register_buffer("disallowed_mask", ~allowed, persistent=False)
        self.edge_bias = nn.Parameter(torch.zeros(6, 6))

    def forward(self, slot_tokens: Tensor) -> Tensor:
        batch_size = slot_tokens.shape[0]
        global_token = self.global_token.expand(batch_size, -1, -1)
        tokens = torch.cat([slot_tokens, global_token], dim=1)
        normed = self.norm1(tokens)
        attn_mask = self.edge_bias.to(dtype=normed.dtype, device=normed.device)
        attn_mask = attn_mask.masked_fill(
            self.disallowed_mask.to(device=normed.device), -1e4)
        tokens = tokens + self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens[:, :slot_tokens.shape[1]]


class SemanticGroupTokenMixer(nn.Module):
    """Soft fixed5 group context mixer for hand/object/TO relation slots."""

    def __init__(self, channels: int, num_slots: int = 5) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_slots = int(num_slots)
        self.norm = nn.LayerNorm(self.channels * 3)
        self.update = nn.Sequential(
            nn.Linear(self.channels * 3, self.channels * 2),
            nn.GELU(),
            nn.Linear(self.channels * 2, self.channels),
        )

    def forward(self, slot_tokens: Tensor) -> Tensor:
        if slot_tokens.shape[1] != 5:
            global_ctx = slot_tokens.mean(dim=1, keepdim=True).expand_as(slot_tokens)
            zeros = torch.zeros_like(global_ctx)
            ctx = torch.cat([slot_tokens, global_ctx, zeros], dim=-1)
            return slot_tokens + self.update(self.norm(ctx))

        hand_ctx = slot_tokens[:, :2].mean(dim=1, keepdim=True)
        obj_ctx = slot_tokens[:, 2:5].mean(dim=1, keepdim=True)
        global_ctx = slot_tokens.mean(dim=1, keepdim=True)
        group_ctx = torch.cat([
            hand_ctx.expand(-1, 2, -1),
            obj_ctx.expand(-1, 3, -1),
        ], dim=1)
        relation_ctx = torch.cat([
            slot_tokens[:, 2:3],
            slot_tokens[:, 3:4],
            slot_tokens[:, 0:1],
            slot_tokens[:, 1:2],
            obj_ctx,
        ], dim=1)
        ctx = torch.cat([
            group_ctx,
            relation_ctx,
            global_ctx.expand_as(slot_tokens),
        ], dim=-1)
        return slot_tokens + self.update(self.norm(ctx))


class MaskConditionedSlotRefineLayer(nn.Module):
    """One lightweight fixed-query refinement step."""

    def __init__(self,
                 channels: int,
                 num_heads: int = 4,
                 use_topology: bool = False,
                 use_mcsr: bool = False) -> None:
        super().__init__()
        self.use_topology = bool(use_topology)
        self.use_mcsr = bool(use_mcsr)
        if self.use_topology:
            self.token_mixer = HardTopologyTokenMixer(
                channels, num_heads=num_heads)
        else:
            self.token_mixer = nn.TransformerEncoderLayer(
                d_model=int(channels),
                nhead=int(num_heads),
                dim_feedforward=int(channels) * 4,
                dropout=0.0,
                batch_first=True,
                activation="gelu")
        self.evidence_norm = nn.LayerNorm(int(channels))
        self.evidence_update = nn.Sequential(
            nn.Linear(int(channels), int(channels) * 2),
            nn.GELU(),
            nn.Linear(int(channels) * 2, int(channels)),
        )

    @staticmethod
    def masked_pool(pixel_feat: Tensor,
                    mask_prob: Tensor,
                    eps: float = 1e-6) -> Tensor:
        weights = mask_prob.to(dtype=pixel_feat.dtype)
        denom = weights.flatten(2).sum(-1).clamp_min(float(eps))
        return torch.einsum("bnhw,bchw->bnc", weights, pixel_feat) / denom.unsqueeze(-1)

    def forward(self,
                slot_tokens: Tensor,
                relation_feat: Tensor,
                mask_logits: Optional[Tensor] = None) -> Tensor:
        slot_tokens = self.token_mixer(slot_tokens)
        if self.use_mcsr and mask_logits is not None:
            evidence = self.masked_pool(relation_feat, mask_logits.sigmoid())
            slot_tokens = slot_tokens + self.evidence_update(
                self.evidence_norm(evidence))
        return slot_tokens


class VOSStyleMemoryAdapter(nn.Module):
    """Lightweight VOS-inspired memory interaction for single-frame finetuning."""

    def __init__(self,
                 channels: int,
                 adapter_type: str = "xmem",
                 num_memory_tokens: int = 8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.adapter_type = str(adapter_type).lower()
        self.num_memory_tokens = int(num_memory_tokens)
        if self.adapter_type not in ("xmem", "cutie", "livos"):
            raise ValueError(
                "adapter_type must be one of 'xmem', 'cutie', or 'livos', "
                f"got {self.adapter_type!r}")
        self.memory_tokens = nn.Parameter(
            torch.randn(self.num_memory_tokens, self.channels) * 0.02)
        self.query_proj = nn.Linear(self.channels, self.channels)
        self.key_proj = nn.Linear(self.channels, self.channels)
        self.value_proj = nn.Linear(self.channels, self.channels)
        self.out_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
        )
        self.coarse_pool_proj = nn.Linear(self.channels, self.channels)
        self.livos_gate = nn.Sequential(
            nn.Conv2d(self.channels + 1, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.Sigmoid(),
        )

    @staticmethod
    def _phi(x: Tensor) -> Tensor:
        return F.elu(x) + 1.0

    def _base_tokens(self,
                     relation_feat: Tensor,
                     coarse_prob: Tensor,
                     slot_queries: Tensor) -> Tensor:
        batch_size = relation_feat.shape[0]
        learned = self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        coarse_weight = coarse_prob.clamp(0.0, 1.0)
        pooled = torch.einsum(
            "bchw,bnhw->bnc",
            relation_feat,
            coarse_weight).squeeze(1)
        denom = coarse_weight.flatten(2).sum(-1).clamp_min(1e-6)
        pooled = pooled / denom
        pooled = self.coarse_pool_proj(pooled).unsqueeze(1)
        if self.adapter_type == "cutie":
            return torch.cat([slot_queries, pooled, learned], dim=1)
        return torch.cat([pooled, learned], dim=1)

    def forward(self,
                relation_feat: Tensor,
                coarse_prob: Tensor,
                slot_queries: Tensor) -> Tensor:
        batch_size, channels, height, width = relation_feat.shape
        pixel_tokens = relation_feat.flatten(2).transpose(1, 2)
        memory = self._base_tokens(relation_feat, coarse_prob, slot_queries)
        if self.adapter_type == "livos":
            q = self._phi(self.query_proj(pixel_tokens))
            k = self._phi(self.key_proj(memory))
            v = self.value_proj(memory)
            kv = torch.einsum("bmc,bmd->bcd", k, v)
            k_sum = k.sum(dim=1)
            denom = torch.einsum("bnc,bc->bn", q, k_sum).clamp_min(1e-6)
            context = torch.einsum("bnc,bcd->bnd", q, kv) / denom.unsqueeze(-1)
            context = context.transpose(1, 2).reshape(
                batch_size, channels, height, width)
            # LiVOS-style gating: memory can correct pixels, but only through
            # a local confidence gate from current feature and coarse mass.
            gate = self.livos_gate(torch.cat([relation_feat, coarse_prob], dim=1))
            return relation_feat + gate * self.out_proj(context)

        q = self.query_proj(pixel_tokens)
        k = self.key_proj(memory)
        v = self.value_proj(memory)
        attn = torch.softmax(
            torch.einsum("bnc,bmc->bnm", q, k) * (channels ** -0.5),
            dim=-1)
        context = torch.einsum("bnm,bmc->bnc", attn, v)
        context = context.transpose(1, 2).reshape(batch_size, channels, height, width)
        return relation_feat + self.out_proj(context)


class SlotEMAVideoMemoryAdapter(nn.Module):
    """Minimal streaming semantic-slot memory for fixed5 HOS videos.

    The state stores one token and one low-resolution probability mask per
    semantic slot.  Unlike XMem/Cutie-style memories, this adapter does not
    perform pixel-level matching; it only refines the current fixed semantic
    queries and optionally smooths the final slot probabilities.
    """

    def __init__(self,
                 channels: int,
                 num_slots: int = 5,
                 use_read_gate: bool = False,
                 use_update_gate: bool = False,
                 use_prob_ema: bool = False,
                 token_momentum: float = 0.85,
                 prob_momentum: float = 0.75) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_slots = int(num_slots)
        self.use_read_gate = bool(use_read_gate)
        self.use_update_gate = bool(use_update_gate)
        self.use_prob_ema = bool(use_prob_ema)
        self.token_momentum = float(token_momentum)
        self.prob_momentum = float(prob_momentum)
        self.read_proj = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.channels),
            nn.GELU(),
            nn.Linear(self.channels, self.channels),
        )
        self.read_gate = nn.Sequential(
            nn.LayerNorm(self.channels * 2),
            nn.Linear(self.channels * 2, self.channels),
            nn.GELU(),
            nn.Linear(self.channels, 1),
        )
        self.update_gate = nn.Sequential(
            nn.LayerNorm(self.channels * 2),
            nn.Linear(self.channels * 2, self.channels),
            nn.GELU(),
            nn.Linear(self.channels, 1),
        )

    @staticmethod
    def _resize_masks(masks: Tensor, size: Sequence[int]) -> Tensor:
        if tuple(masks.shape[-2:]) == tuple(size):
            return masks.float().clamp(0.0, 1.0)
        return F.interpolate(
            masks.float(),
            size=tuple(size),
            mode="bilinear",
            align_corners=False).clamp(0.0, 1.0)

    @staticmethod
    def _masked_pool(pixel_feat: Tensor,
                     masks: Tensor,
                     eps: float = 1e-6) -> Tensor:
        weights = masks.to(dtype=pixel_feat.dtype)
        denom = weights.flatten(2).sum(-1).clamp_min(float(eps))
        return torch.einsum("bnhw,bchw->bnc", weights, pixel_feat) / denom.unsqueeze(-1)

    def read(self,
             slot_tokens: Tensor,
             state: Optional[Dict[str, Tensor]]) -> tuple[Tensor, Dict[str, Tensor]]:
        if not state or "tokens" not in state:
            return slot_tokens, {}
        memory_tokens = state["tokens"].to(dtype=slot_tokens.dtype)
        read_delta = self.read_proj(memory_tokens)
        if self.use_read_gate:
            gate = torch.sigmoid(
                self.read_gate(torch.cat([slot_tokens, memory_tokens], dim=-1)))
        else:
            gate = slot_tokens.new_ones(slot_tokens.shape[:2] + (1, ))
        return slot_tokens + gate * read_delta, {
            "read_gate": gate.squeeze(-1),
            "memory_read_energy": read_delta.square().mean(),
        }

    def blend_logits(self,
                     logits: Tensor,
                     state: Optional[Dict[str, Tensor]]) -> Tensor:
        if not self.use_prob_ema or not state or "masks" not in state:
            return logits
        memory_prob = self._resize_masks(state["masks"], logits.shape[-2:])
        current_prob = logits.sigmoid()
        blended = (
            float(self.prob_momentum) * memory_prob +
            (1.0 - float(self.prob_momentum)) * current_prob)
        return torch.logit(blended.clamp(1e-4, 1.0 - 1e-4))

    def update(self,
               state: Optional[Dict[str, Tensor]],
               relation_feat: Tensor,
               masks: Tensor,
               coarse_prob: Tensor) -> Dict[str, Tensor]:
        del coarse_prob
        masks_low = self._resize_masks(masks, relation_feat.shape[-2:])
        new_tokens = self._masked_pool(relation_feat, masks_low)
        if state is None or "tokens" not in state:
            update_gate = new_tokens.new_ones(new_tokens.shape[:2])
            tokens = new_tokens
            masks_state = masks_low
        else:
            old_tokens = state["tokens"].to(dtype=new_tokens.dtype)
            if self.use_update_gate:
                update_gate = torch.sigmoid(
                    self.update_gate(
                        torch.cat([old_tokens, new_tokens], dim=-1))).squeeze(-1)
            else:
                update_gate = new_tokens.new_full(
                    new_tokens.shape[:2], 1.0 - float(self.token_momentum))
            gate = update_gate.unsqueeze(-1)
            tokens = old_tokens * (1.0 - gate) + new_tokens * gate
            old_masks = self._resize_masks(state["masks"], relation_feat.shape[-2:])
            mask_gate = (
                update_gate.unsqueeze(-1).unsqueeze(-1)
                if self.use_update_gate else
                masks_low.new_full(masks_low.shape[:2] + (1, 1),
                                   1.0 - float(self.prob_momentum)))
            masks_state = old_masks * (1.0 - mask_gate) + masks_low * mask_gate
        return dict(
            memory_type="slotema",
            tokens=tokens,
            masks=masks_state.clamp(0.0, 1.0),
            last_update_gate=update_gate)


class TrueVideoMemoryAdapter(nn.Module):
    """Streaming read/write memory used by HOI4D clip training."""

    def __init__(self,
                 channels: int,
                 memory_type: str,
                 num_slots: int = 5) -> None:
        super().__init__()
        self.channels = int(channels)
        self.memory_type = str(memory_type).lower()
        self.num_slots = int(num_slots)
        if self.memory_type not in ("xmem", "cutie", "livos"):
            raise ValueError(
                "memory_type must be one of 'xmem', 'cutie', or 'livos', "
                f"got {self.memory_type!r}")
        self.key_proj = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.value_proj = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.read_out = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
        )
        self.object_proj = nn.Linear(self.channels, self.channels)
        self.sensory_update = nn.Sequential(
            ConvBNAct(self.channels * 2 + 1, self.channels, 3),
            nn.Conv2d(self.channels, self.channels, 1),
        )
        self.livos_gate = nn.Sequential(
            nn.Conv2d(self.channels + 1, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.Sigmoid(),
        )

    @staticmethod
    def _phi(x: Tensor) -> Tensor:
        return F.elu(x) + 1.0

    @staticmethod
    def _resize_masks(masks: Tensor, size: Sequence[int]) -> Tensor:
        return F.interpolate(
            masks.float(),
            size=tuple(size),
            mode="bilinear",
            align_corners=False).clamp(0.0, 1.0)

    @staticmethod
    def _resize_prob(prob: Tensor, size: Sequence[int]) -> Tensor:
        if tuple(prob.shape[-2:]) == tuple(size):
            return prob.float().clamp(0.0, 1.0)
        return F.interpolate(
            prob.float(),
            size=tuple(size),
            mode="bilinear",
            align_corners=False).clamp(0.0, 1.0)

    @staticmethod
    def _masked_pool(pixel_feat: Tensor,
                     masks: Tensor,
                     eps: float = 1e-6) -> Tensor:
        denom = masks.flatten(2).sum(-1).clamp_min(float(eps))
        return torch.einsum("bnhw,bchw->bnc", masks, pixel_feat) / denom.unsqueeze(-1)

    def _make_slot_values(self, relation_feat: Tensor,
                          masks: Tensor) -> tuple[Tensor, Tensor]:
        masks_low = self._resize_masks(masks, relation_feat.shape[-2:])
        projected = self.value_proj(relation_feat)
        values = self._masked_pool(projected, masks_low)
        keys = self._masked_pool(self.key_proj(relation_feat), masks_low)
        return keys, values

    def init_state(self, relation_feat: Tensor, masks: Tensor,
                   coarse_prob: Tensor) -> Dict[str, Tensor]:
        keys, values = self._make_slot_values(relation_feat, masks)
        coarse_low = self._resize_prob(coarse_prob, relation_feat.shape[-2:])
        state: Dict[str, Tensor] = dict(memory_type=self.memory_type)
        if self.memory_type == "xmem":
            state.update(keys=keys, values=values)
        elif self.memory_type == "cutie":
            masks_low = self._resize_masks(masks, relation_feat.shape[-2:])
            sensory = self.sensory_update(
                torch.cat([relation_feat, relation_feat * masks_low.mean(1, keepdim=True),
                           coarse_low], dim=1))
            state.update(
                keys=keys,
                values=values,
                object_memory=self.object_proj(values),
                sensory=sensory)
        else:
            key_map = self._phi(self.key_proj(relation_feat))
            value_map = self.value_proj(relation_feat)
            masks_low = self._resize_masks(masks, relation_feat.shape[-2:])
            state_matrix = torch.einsum(
                "bkhw,bvhw,bnhw->bnkv", key_map, value_map, masks_low)
            key_sum = key_map
            state.update(
                state_matrix=state_matrix,
                key_sum=key_sum,
                object_memory=self.object_proj(values))
        return state

    def read(self, relation_feat: Tensor, coarse_prob: Tensor,
             state: Optional[Dict[str, Tensor]]) -> tuple[Tensor, Dict[str, Tensor]]:
        if not state:
            return relation_feat, {}
        b, c, h, w = relation_feat.shape
        if self.memory_type == "livos":
            coarse_low = self._resize_prob(coarse_prob, relation_feat.shape[-2:])
            q = self._phi(self.key_proj(relation_feat))
            readout = torch.einsum(
                "bkhw,bnkv->bnvhw", q, state["state_matrix"])
            normalizer = torch.einsum(
                "bkhw,bkhw->b", q, state["key_sum"]).clamp_min(1e-6)
            readout = readout / normalizer.view(b, 1, 1, 1, 1)
            context = readout.mean(dim=1)
            gate = self.livos_gate(torch.cat([relation_feat, coarse_low], dim=1))
            return relation_feat + gate * self.read_out(context), {
                "memory_read_energy": context.square().mean()
            }

        pixel_tokens = relation_feat.flatten(2).transpose(1, 2)
        keys = state["keys"]
        values = state["values"]
        if self.memory_type == "cutie":
            values = values + state["object_memory"]
            sensory_token = F.adaptive_avg_pool2d(state["sensory"], 1).flatten(1)
            values = torch.cat([values, sensory_token.unsqueeze(1)], dim=1)
            keys = torch.cat([keys, sensory_token.unsqueeze(1)], dim=1)
        q = pixel_tokens
        attn = torch.softmax(
            torch.einsum("bnc,bmc->bnm", q, keys) * (c ** -0.5),
            dim=-1)
        context = torch.einsum("bnm,bmc->bnc", attn, values)
        context = context.transpose(1, 2).reshape(b, c, h, w)
        return relation_feat + self.read_out(context), {
            "memory_read_energy": context.square().mean()
        }

    def update(self,
               state: Optional[Dict[str, Tensor]],
               relation_feat: Tensor,
               masks: Tensor,
               coarse_prob: Tensor) -> Dict[str, Tensor]:
        if state is None:
            return self.init_state(relation_feat, masks, coarse_prob)
        keys, values = self._make_slot_values(relation_feat, masks)
        if self.memory_type == "xmem":
            return dict(
                memory_type=self.memory_type,
                keys=torch.cat([state["keys"], keys], dim=1),
                values=torch.cat([state["values"], values], dim=1))
        if self.memory_type == "cutie":
            masks_low = self._resize_masks(masks, relation_feat.shape[-2:])
            sensory_delta = self.sensory_update(
                torch.cat([relation_feat, state["sensory"],
                           masks_low.mean(1, keepdim=True)], dim=1))
            return dict(
                memory_type=self.memory_type,
                keys=torch.cat([state["keys"], keys], dim=1),
                values=torch.cat([state["values"], values], dim=1),
                object_memory=torch.cat(
                    [state["object_memory"], self.object_proj(values)], dim=1),
                sensory=state["sensory"] + torch.tanh(sensory_delta))

        key_map = self._phi(self.key_proj(relation_feat))
        value_map = self.value_proj(relation_feat)
        masks_low = self._resize_masks(masks, relation_feat.shape[-2:])
        coarse_low = self._resize_prob(coarse_prob, relation_feat.shape[-2:])
        new_state = torch.einsum(
            "bkhw,bvhw,bnhw->bnkv", key_map, value_map, masks_low)
        gate = F.adaptive_avg_pool2d(
            self.livos_gate(torch.cat([relation_feat, coarse_low], dim=1)),
            1).flatten(1)
        state_matrix = state["state_matrix"] * gate[:, None, None, :] + new_state
        return dict(
            memory_type=self.memory_type,
            state_matrix=state_matrix,
            key_sum=state["key_sum"] + key_map,
            object_memory=state["object_memory"] + self.object_proj(values))


@MODELS.register_module()
class LiteHOSV2CoarseMCSRSlotHead(nn.Module):
    """Single-model fixed5 head with CPI, mask-conditioned refinement, topology."""

    def __init__(self,
                 in_channels: Sequence[int] = (96, 192, 384, 768),
                 feat_channels: int = 256,
                 num_refine_layers: int = 2,
                 num_heads: int = 4,
                 out_channels: int = 5,
                 class_out_channels: Optional[int] = None,
                 return_intermediate: bool = True,
                 use_cpi: bool = False,
                 use_mcsr: bool = False,
                 use_topology: bool = False,
                 memory_adapter_type: str = "none",
                 memory_num_tokens: int = 8,
                 cpi_kernel_size: int = 7,
                 align_corners: bool = False) -> None:
        super().__init__()
        self.in_channels = [int(c) for c in in_channels]
        self.feat_channels = int(feat_channels)
        self.out_channels = int(out_channels)
        self.num_classes = self.out_channels
        self.class_out_channels = (
            self.out_channels + 1 if class_out_channels is None
            else int(class_out_channels))
        self.num_refine_layers = int(num_refine_layers)
        self.return_intermediate = bool(return_intermediate)
        self.use_cpi = bool(use_cpi)
        self.use_mcsr = bool(use_mcsr)
        self.use_topology = bool(use_topology)
        self.memory_adapter_type = str(memory_adapter_type).lower()
        self.video_memory_type = self.memory_adapter_type
        self.align_corners = bool(align_corners)
        slotema_types = (
            "slotema",
            "slotema-gated",
            "slotema-gated-probema",
        )
        if self.out_channels != 5:
            raise ValueError(
                "LiteHOSV2CoarseMCSRSlotHead is the fixed5 single-model head; "
                f"got out_channels={self.out_channels}")
        if self.class_out_channels != self.out_channels + 1:
            raise ValueError(
                "fixed5 class head must output 5 foreground classes plus no-object, "
                f"got class_out_channels={self.class_out_channels}")
        if self.num_refine_layers < 1:
            raise ValueError("num_refine_layers must be >= 1")

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.feat_channels, 1, bias=False),
                nn.BatchNorm2d(self.feat_channels),
                nn.GELU(),
            ) for ch in self.in_channels
        ])
        self.coarse_head = nn.Sequential(
            ConvBNAct(self.feat_channels, self.feat_channels, 3),
            nn.Conv2d(self.feat_channels, 1, 1),
        )
        self.relation_refine = nn.Sequential(
            ConvBNAct(self.feat_channels + 1, self.feat_channels, 3),
            ConvBNAct(self.feat_channels, self.feat_channels, 3),
        )
        self.cpi = (
            CoarsePixelInteractionMixer(
                self.feat_channels, kernel_size=int(cpi_kernel_size))
            if self.use_cpi else None)
        self.streaming_memory_adapter = (
            SlotEMAVideoMemoryAdapter(
                self.feat_channels,
                num_slots=self.out_channels,
                use_read_gate=self.memory_adapter_type in (
                    "slotema-gated", "slotema-gated-probema"),
                use_update_gate=self.memory_adapter_type in (
                    "slotema-gated", "slotema-gated-probema"),
                use_prob_ema=self.memory_adapter_type == "slotema-gated-probema")
            if self.memory_adapter_type in slotema_types else None)
        self.memory_adapter = (
            None if (
                self.memory_adapter_type in ("none", "") or
                self.memory_adapter_type in slotema_types)
            else VOSStyleMemoryAdapter(
                self.feat_channels,
                adapter_type=self.memory_adapter_type,
                num_memory_tokens=int(memory_num_tokens)))
        self.slot_queries = nn.Parameter(
            torch.randn(self.out_channels, self.feat_channels) * 0.02)
        self.refine_layers = nn.ModuleList([
            MaskConditionedSlotRefineLayer(
                self.feat_channels,
                num_heads=int(num_heads),
                use_topology=self.use_topology,
                use_mcsr=self.use_mcsr)
            for _ in range(self.num_refine_layers)
        ])
        self.slot_norm = nn.LayerNorm(self.feat_channels)
        self.mask_embed = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels),
            nn.GELU(),
            nn.Linear(self.feat_channels, self.feat_channels),
        )
        self.class_embed = nn.Linear(
            self.feat_channels, self.class_out_channels)

    def _fuse_features(self, feats: Sequence[Tensor]) -> Tensor:
        if len(feats) != len(self.proj):
            raise ValueError(f"Expected {len(self.proj)} feature maps, got {len(feats)}")
        target_size = feats[0].shape[-2:]
        fused = 0.0
        for feat, proj in zip(feats, self.proj):
            value = proj(feat)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=self.align_corners)
            fused = fused + value
        return fused / float(len(feats))

    def _pack_outputs(self,
                      all_logits: List[Tensor],
                      all_class_logits: List[Tensor]) -> Dict[str, Tensor]:
        prob5_layers = [logits.sigmoid() for logits in all_logits]
        all_logits4 = [
            torch.logit(
                torch.stack(
                    [
                        prob5[:, 0],
                        prob5[:, 1],
                        torch.maximum(prob5[:, 2], prob5[:, 4]),
                        torch.maximum(prob5[:, 3], prob5[:, 4]),
                    ],
                    dim=1).clamp(1e-4, 1.0 - 1e-4))
            for prob5 in prob5_layers
        ]
        return dict(
            logits5=all_logits[-1],
            all_logits5=all_logits,
            logits4=all_logits4[-1],
            all_logits4=all_logits4,
            class_logits=all_class_logits[-1],
            all_class_logits=all_class_logits)

    def forward(self,
                feats: Sequence[Tensor],
                depth_prior: Optional[Tensor] = None,
                mask_pilot_targets: Optional[Tensor] = None,
                video_memory_state: Optional[Dict[str, Tensor]] = None
                ) -> Dict[str, Tensor]:
        del depth_prior, mask_pilot_targets
        pixel_feat = self._fuse_features(feats)
        coarse_logits = self.coarse_head(pixel_feat)
        coarse_prob = coarse_logits.sigmoid()
        relation_feat = self.relation_refine(
            torch.cat([pixel_feat, coarse_prob], dim=1))
        if self.cpi is not None:
            relation_feat = self.cpi(relation_feat, coarse_prob)

        batch_size = relation_feat.shape[0]
        slot_tokens = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        memory_aux = {}
        if self.streaming_memory_adapter is not None:
            slot_tokens, memory_aux = self.streaming_memory_adapter.read(
                slot_tokens, video_memory_state)
        if self.memory_adapter is not None:
            relation_feat = self.memory_adapter(
                relation_feat, coarse_prob, slot_tokens)
        all_logits: List[Tensor] = []
        all_class_logits: List[Tensor] = []
        prev_logits: Optional[Tensor] = None
        for layer in self.refine_layers:
            slot_tokens = layer(slot_tokens, relation_feat, prev_logits)
            normed_tokens = self.slot_norm(slot_tokens)
            mask_kernels = self.mask_embed(normed_tokens)
            logits = torch.einsum("bnc,bchw->bnhw", mask_kernels, relation_feat)
            all_logits.append(logits)
            all_class_logits.append(self.class_embed(normed_tokens))
            prev_logits = logits
        if self.streaming_memory_adapter is not None and all_logits:
            all_logits[-1] = self.streaming_memory_adapter.blend_logits(
                all_logits[-1], video_memory_state)

        if not self.return_intermediate:
            all_logits = all_logits[-1:]
            all_class_logits = all_class_logits[-1:]
        result = self._pack_outputs(all_logits, all_class_logits)
        result.update(
            coarse_logits=coarse_logits,
            slot_tokens=self.slot_norm(slot_tokens),
            relation_feat=relation_feat,
            pixel_feat=pixel_feat,
            dynamic_prototypes=None,
            prototype_attn=None,
            relation_dynamic=None,
            object_support_logits2=None,
            contact_guidance_logits=None,
            hand_condition_low=None,
            object_relation_feat=None,
            memory_aux=memory_aux,
            decoder_design=dict(
                type="coarse_mcsr_slot",
                use_cpi=self.use_cpi,
                use_mcsr=self.use_mcsr,
                use_topology=self.use_topology,
                memory_adapter_type=self.memory_adapter_type,
                num_refine_layers=self.num_refine_layers))
        return result

    def update_video_memory(self,
                            state: Optional[Dict[str, Tensor]],
                            relation_feat: Tensor,
                            masks: Tensor,
                            coarse_prob: Tensor) -> Optional[Dict[str, Tensor]]:
        if self.streaming_memory_adapter is None:
            return state
        return self.streaming_memory_adapter.update(
            state, relation_feat, masks, coarse_prob)


class LinearCrossAttentionDecoderLayer(nn.Module):
    """Small decoder layer with ELU+1 linear cross-attention over pixels."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 act_cfg: Optional[dict] = None) -> None:
        super().__init__()
        del act_cfg
        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.self_attn = nn.MultiheadAttention(
            self.embed_dims, self.num_heads, dropout=0.0, batch_first=True)
        self.q_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.k_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.v_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.cross_out = nn.Linear(self.embed_dims, self.embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dims, int(feedforward_channels)),
            nn.ReLU(inplace=True),
            nn.Linear(int(feedforward_channels), self.embed_dims),
        )
        self.norm1 = nn.LayerNorm(self.embed_dims)
        self.norm2 = nn.LayerNorm(self.embed_dims)
        self.norm3 = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def _with_pos(x: Tensor, pos: Optional[Tensor]) -> Tensor:
        return x if pos is None else x + pos

    @staticmethod
    def _phi(x: Tensor) -> Tensor:
        return F.elu(x) + 1.0

    def _linear_cross_attn(self,
                           query: Tensor,
                           key: Tensor,
                           value: Tensor,
                           query_pos: Optional[Tensor] = None,
                           key_pos: Optional[Tensor] = None,
                           cross_attn_mask: Optional[Tensor] = None) -> Tensor:
        q = self._phi(self.q_proj(self._with_pos(query, query_pos)))
        k = self._phi(self.k_proj(self._with_pos(key, key_pos)))
        v = self.v_proj(value)
        if cross_attn_mask is not None:
            batch_size, num_queries = query.shape[:2]
            if cross_attn_mask.dim() == 3 and cross_attn_mask.shape[0] % batch_size == 0:
                num_heads = cross_attn_mask.shape[0] // batch_size
                mask = cross_attn_mask.reshape(
                    batch_size, num_heads, num_queries,
                    cross_attn_mask.shape[-1])[:, 0]
            elif cross_attn_mask.dim() == 3:
                mask = cross_attn_mask
            else:
                raise ValueError(
                    "linear cross-attention mask must be [B*H,Q,N] or [B,Q,N], "
                    f"got {tuple(cross_attn_mask.shape)}")
            allowed = (~mask).to(dtype=k.dtype).unsqueeze(-1)
            k_masked = k.unsqueeze(1) * allowed
            kv = torch.einsum("bqnc,bnd->bqcd", k_masked, v)
            k_sum = k_masked.sum(dim=2)
            denom = torch.einsum("bqc,bqc->bq", q, k_sum).clamp_min(1e-6)
            out = torch.einsum("bqc,bqcd->bqd", q, kv) / denom.unsqueeze(-1)
            return self.cross_out(out)
        kv = torch.einsum("bnc,bnd->bcd", k, v)
        k_sum = k.sum(dim=1)
        denom = torch.einsum("bqc,bc->bq", q, k_sum).clamp_min(1e-6)
        out = torch.einsum("bqc,bcd->bqd", q, kv) / denom.unsqueeze(-1)
        return self.cross_out(out)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        q_norm = self.norm1(query)
        self_query = self._with_pos(q_norm, query_pos)
        self_out = self.self_attn(
            self_query, self_query, q_norm, need_weights=False)[0]
        query = query + self_out
        query = query + self._linear_cross_attn(
            self.norm2(query),
            key,
            value,
            query_pos=query_pos,
            key_pos=key_pos,
            cross_attn_mask=kwargs.get("cross_attn_mask"))
        query = query + self.ffn(self.norm3(query))
        return query


class RelationPrototypeExtractor(nn.Module):
    """INP-style current-frame prototype extractor for four relation slots."""

    def __init__(self,
                 channels: int,
                 num_slots: int = 4,
                 num_heads: int = 4,
                 mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_slots = int(num_slots)
        self.num_heads = int(num_heads)
        if self.channels % self.num_heads != 0:
            raise ValueError(
                f"channels={self.channels} must be divisible by num_heads={self.num_heads}")
        self.prototype_tokens = nn.Parameter(
            torch.randn(self.num_slots, self.channels) * 0.02)
        self.norm_q = nn.LayerNorm(self.channels)
        self.norm_kv = nn.LayerNorm(self.channels)
        self.q = nn.Linear(self.channels, self.channels)
        self.kv = nn.Linear(self.channels, self.channels * 2)
        self.proj = nn.Linear(self.channels, self.channels)
        hidden = int(self.channels * mlp_ratio)
        self.norm_out = nn.LayerNorm(self.channels)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
        )

    def forward(self, pixel_feat: Tensor) -> Tuple[Tensor, Tensor]:
        b, c, h, w = pixel_feat.shape
        tokens = pixel_feat.flatten(2).transpose(1, 2)
        query = self.prototype_tokens.unsqueeze(0).expand(b, -1, -1)
        query = self.norm_q(query)
        tokens = self.norm_kv(tokens)
        head_dim = c // self.num_heads
        q = self.q(query).reshape(b, self.num_slots, self.num_heads, head_dim)
        q = q.permute(0, 2, 1, 3)
        kv = self.kv(tokens).reshape(b, h * w, 2, self.num_heads, head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        proto = (attn @ v).transpose(1, 2).reshape(b, self.num_slots, c)
        proto = self.proj(proto)
        proto = proto + self.mlp(self.norm_out(proto))
        attn_map = attn.mean(dim=1).reshape(b, self.num_slots, h, w)
        return proto, attn_map


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(value)).item()


@MODELS.register_module()
class LiteHOSV2FastEncoder(nn.Module):
    """Small current-frame encoder used before gate and fast path."""

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 out_channels: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, channels, 3),
            ConvBNAct(channels, channels, 3),
            nn.Conv2d(channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.out_channels = int(out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.stem(inputs)


@MODELS.register_module()
class LiteHOSV2RelationSlotHead(nn.Module):
    """Compact role-indexed decoder; paper configurations use five foreground roles."""

    def __init__(self,
                 in_channels: Sequence[int] = (64, 128, 256, 512),
                 feat_channels: int = 192,
                 num_decoder_layers: int = 4,
                 num_heads: int = 4,
                 out_channels: int = 4,
                 class_out_channels: Optional[int] = None,
                 return_intermediate: bool = False,
                 use_depth_prior: bool = False,
                 use_relation_prototypes: bool = False,
                 use_object_support_aux: bool = False,
                 use_hand_condition_lowres: bool = False,
                 detach_hand_condition: bool = True,
                 use_coarse_contact_guidance: bool = False,
                 use_scse_refine: bool = False,
                 concat_attention: Optional[str] = None,
                 decoder_attention: Optional[str] = None,
                 use_mask_logit_refine: bool = False,
                 use_query_evidence_update: bool = False,
                 use_semantic_group_mixer: bool = False,
                 use_query_specific_mask_embed: bool = False,
                 use_relation_topology_mixer: bool = False,
                 relation_topology_mixer_type: str = "hard",
                 relation_topology_pattern: str = "matched",
                 use_hand_conditioned_object_refine: bool = False,
                 hand_object_mapping: str = "matched",
                 use_foreground_context_object_refine: bool = False,
                 use_hybrid_hand_object_queries: bool = False,
                 pid_aggregation: Optional[str] = None,
                 pixel_self_attn_num_layers: int = 0,
                 pixel_self_attn_window_size: int = 7,
                 pixel_self_attn_num_heads: Optional[int] = None,
                 dynamic_query_init_scale: float = 0.1,
                 use_global_query_bias: Optional[bool] = None,
                 coarse_prior_activation: str = "sigmoid",
                 use_coarse_uncertainty_prior: bool = False,
                 coarse_condition_mask: Optional[Sequence[bool]] = None,
                 use_coarse_reliability_head: bool = False,
                 use_reliability_query_pooling: bool = False,
                 use_coarse_reliability_guided_aggregator: bool = False,
                 coarse_reliability_aggregator_type: str = "dw",
                 coarse_prior_perturb_prob: float = 0.0,
                 coarse_prior_noise_std: float = 0.0,
                 coarse_prior_dropout_prob: float = 0.0,
                 gt_coarse_prior_anneal_iters: int = 0,
                 gt_coarse_prior_start_weight: float = 1.0,
                 video_memory_type: str = "none",
                 align_corners: bool = False) -> None:
        super().__init__()
        self.in_channels = [int(c) for c in in_channels]
        self.feat_channels = int(feat_channels)
        self.num_classes = int(out_channels)
        self.out_channels = int(out_channels)
        self.class_out_channels = (
            self.out_channels if class_out_channels is None
            else int(class_out_channels))
        self.return_intermediate = bool(return_intermediate)
        if self.out_channels not in (4, 5, 6):
            raise ValueError(
                "LiteHOSV2RelationSlotHead supports out_channels=4, 5, or legacy softmax6, "
                f"got {self.out_channels}")
        self.use_depth_prior = bool(use_depth_prior)
        self.use_relation_prototypes = bool(use_relation_prototypes)
        self.use_object_support_aux = bool(use_object_support_aux)
        self.use_hand_condition_lowres = bool(use_hand_condition_lowres)
        self.detach_hand_condition = bool(detach_hand_condition)
        self.use_coarse_contact_guidance = bool(use_coarse_contact_guidance)
        self.use_scse_refine = bool(use_scse_refine)
        self.concat_attention_name = concat_attention
        self.decoder_attention_name = decoder_attention
        self.use_mask_logit_refine = bool(use_mask_logit_refine)
        self.use_query_evidence_update = bool(use_query_evidence_update)
        self.use_semantic_group_mixer = bool(use_semantic_group_mixer)
        self.use_query_specific_mask_embed = bool(use_query_specific_mask_embed)
        self.use_relation_topology_mixer = bool(use_relation_topology_mixer)
        self.relation_topology_mixer_type = (
            str(relation_topology_mixer_type).strip().lower().replace("-", "_"))
        self.relation_topology_pattern = str(
            relation_topology_pattern).strip().lower()
        self.use_hand_conditioned_object_refine = bool(
            use_hand_conditioned_object_refine)
        self.hand_object_mapping = str(hand_object_mapping).strip().lower()
        self.use_foreground_context_object_refine = bool(
            use_foreground_context_object_refine)
        self.use_hybrid_hand_object_queries = bool(use_hybrid_hand_object_queries)
        self.use_coarse_uncertainty_prior = bool(use_coarse_uncertainty_prior)
        if coarse_condition_mask is None:
            coarse_condition_mask = (True, True, True)
        if len(coarse_condition_mask) != 3:
            raise ValueError(
                "coarse_condition_mask must contain exactly (use_p, use_u, use_r)")
        self.coarse_condition_mask = tuple(bool(v) for v in coarse_condition_mask)
        self.use_coarse_reliability_head = bool(use_coarse_reliability_head)
        self.use_reliability_query_pooling = bool(use_reliability_query_pooling)
        self.use_coarse_reliability_guided_aggregator = bool(
            use_coarse_reliability_guided_aggregator)
        self.coarse_reliability_aggregator_type = (
            str(coarse_reliability_aggregator_type).strip().lower().replace("-", "_"))
        self.coarse_prior_perturb_prob = float(coarse_prior_perturb_prob)
        self.coarse_prior_noise_std = float(coarse_prior_noise_std)
        self.coarse_prior_dropout_prob = float(coarse_prior_dropout_prob)
        self.gt_coarse_prior_anneal_iters = int(gt_coarse_prior_anneal_iters)
        self.gt_coarse_prior_start_weight = float(gt_coarse_prior_start_weight)
        if not 0.0 <= self.coarse_prior_perturb_prob <= 1.0:
            raise ValueError("coarse_prior_perturb_prob must be in [0, 1]")
        if self.coarse_prior_noise_std < 0.0:
            raise ValueError("coarse_prior_noise_std must be non-negative")
        if not 0.0 <= self.coarse_prior_dropout_prob <= 1.0:
            raise ValueError("coarse_prior_dropout_prob must be in [0, 1]")
        if self.gt_coarse_prior_anneal_iters < 0:
            raise ValueError("gt_coarse_prior_anneal_iters must be non-negative")
        if not 0.0 <= self.gt_coarse_prior_start_weight <= 1.0:
            raise ValueError("gt_coarse_prior_start_weight must be in [0, 1]")
        self.coarse_prior_activation = str(coarse_prior_activation).lower()
        if self.coarse_prior_activation in ("raw", "logits", "none"):
            self.coarse_prior_activation = "identity"
        if self.coarse_prior_activation not in ("sigmoid", "identity"):
            raise ValueError(
                "coarse_prior_activation must be 'sigmoid' or 'identity', "
                f"got {coarse_prior_activation!r}")
        if (self.gt_coarse_prior_anneal_iters > 0 and
                self.coarse_prior_activation != "sigmoid"):
            raise ValueError(
                "GT coarse-prior annealing requires sigmoid coarse prior activation")
        self.pid_aggregation_name = (
            None if pid_aggregation is None else
            str(pid_aggregation).strip().lower().replace("-", "_"))
        self.pixel_self_attn_num_layers = int(pixel_self_attn_num_layers)
        self.video_memory_type = str(video_memory_type).lower()
        if self.pid_aggregation_name in ("", "none", "identity"):
            self.pid_aggregation_name = None
        if self.pid_aggregation_name not in (None, "pid3_lightbag"):
            raise ValueError(
                "pid_aggregation must be None or 'pid3_lightbag', "
                f"got {pid_aggregation!r}")
        if self.use_hybrid_hand_object_queries and (
                self.out_channels != 5 or self.class_out_channels < 6):
            raise ValueError(
                "Hybrid hand-object queries require out_channels=5 and "
                "class_out_channels>=6")
        if self.use_relation_topology_mixer and self.out_channels != 5:
            raise ValueError("Relation topology mixer currently requires fixed5 slots")
        if self.use_hand_conditioned_object_refine and self.out_channels != 5:
            raise ValueError(
                "Hand-conditioned object refinement currently requires fixed5 slots")
        if (self.use_foreground_context_object_refine and
                not self.use_hand_conditioned_object_refine):
            raise ValueError(
                "use_foreground_context_object_refine requires "
                "use_hand_conditioned_object_refine=True")
        if self.relation_topology_mixer_type not in ("hard", "edge_gated"):
            raise ValueError(
                "relation_topology_mixer_type must be 'hard' or 'edge_gated', "
                f"got {relation_topology_mixer_type!r}")
        if self.relation_topology_pattern not in ("matched", "crossed"):
            raise ValueError(
                "relation_topology_pattern must be 'matched' or 'crossed', "
                f"got {relation_topology_pattern!r}")
        if self.hand_object_mapping not in ("matched", "swapped"):
            raise ValueError(
                "hand_object_mapping must be 'matched' or 'swapped', "
                f"got {hand_object_mapping!r}")
        if (not self.use_coarse_uncertainty_prior and
                self.coarse_condition_mask != (True, True, True)):
            raise ValueError(
                "coarse_condition_mask requires use_coarse_uncertainty_prior=True "
                "so the relation refiner keeps the frozen [F,P,U,R] graph")
        if self.coarse_reliability_aggregator_type in ("efficient", "lite"):
            self.coarse_reliability_aggregator_type = "dw"
        if self.coarse_reliability_aggregator_type in ("dense", "legacy_dense"):
            self.coarse_reliability_aggregator_type = "legacy"
        if self.coarse_reliability_aggregator_type not in ("dw", "legacy"):
            raise ValueError(
                "coarse_reliability_aggregator_type must be 'dw' or 'legacy', "
                f"got {coarse_reliability_aggregator_type!r}")
        if self.use_relation_prototypes and self.out_channels != 4:
            raise ValueError("Relation prototypes currently require out_channels=4")
        if self.use_object_support_aux and not self.use_relation_prototypes:
            raise ValueError("use_object_support_aux requires use_relation_prototypes=True")
        if self.use_hand_condition_lowres and self.out_channels != 4:
            raise ValueError("Low-res hand conditioning currently requires out_channels=4")
        self.use_global_query_bias = (
            not self.use_relation_prototypes
            if use_global_query_bias is None else bool(use_global_query_bias))
        self.align_corners = bool(align_corners)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.feat_channels, 1, bias=False),
                nn.BatchNorm2d(self.feat_channels),
                nn.GELU(),
            ) for ch in self.in_channels
        ])
        self.coarse_head = nn.Sequential(
            ConvBNAct(self.feat_channels, self.feat_channels, 3),
            nn.Conv2d(self.feat_channels, 1, 1),
        )
        self.register_buffer(
            "_gt_coarse_prior_fallback_iter",
            torch.zeros((), dtype=torch.long),
            persistent=True)
        self.coarse_contact_guidance = (
            CoarseContactGuidedAggregator(self.feat_channels)
            if self.use_coarse_contact_guidance else None)
        self.coarse_reliability_head = (
            CoarseReliabilityHead(self.feat_channels)
            if self.use_coarse_reliability_head else None)
        self.coarse_reliability_aggregator = (
            (LegacyDenseCoarseReliabilityGuidedAggregator(self.feat_channels)
             if self.coarse_reliability_aggregator_type == "legacy"
             else CoarseReliabilityGuidedAggregator(self.feat_channels))
            if self.use_coarse_reliability_guided_aggregator else None)
        self.slot_queries = nn.Parameter(
            torch.randn(self.out_channels, self.feat_channels) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_channels,
            nhead=int(num_heads),
            dim_feedforward=self.feat_channels * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.slot_mixer = nn.TransformerEncoder(encoder_layer, int(num_decoder_layers))
        self.slot_norm = nn.LayerNorm(self.feat_channels)
        self.mask_embed = nn.Linear(self.feat_channels, self.feat_channels)
        self.mask_embed_per_slot = (
            nn.ModuleList([
                nn.Linear(self.feat_channels, self.feat_channels)
                for _ in range(self.out_channels)
            ]) if self.use_query_specific_mask_embed else None)
        self.class_embed = nn.Linear(self.feat_channels, self.class_out_channels)
        prior_channels = 3 if self.use_coarse_uncertainty_prior else 1
        refine_in_channels = (
            self.feat_channels + prior_channels + int(self.use_depth_prior))
        self.refine = nn.Sequential(
            ConvBNAct(refine_in_channels, self.feat_channels, 3),
            ConvBNAct(self.feat_channels, self.feat_channels, 3),
        )
        self.reliability_query_proj = (
            nn.Sequential(
                nn.LayerNorm(self.feat_channels * 2),
                nn.Linear(self.feat_channels * 2, self.feat_channels),
            ) if self.use_reliability_query_pooling else None)
        self.refine_scse = (
            SCSEBlock(self.feat_channels) if self.use_scse_refine else None)
        self.concat_attention = build_lite_attention(
            self.concat_attention_name, self.feat_channels)
        self.decoder_attention = build_lite_attention(
            self.decoder_attention_name, self.feat_channels)
        self.pid_aggregation = (
            PID3LightBagAggregator(self.feat_channels)
            if self.pid_aggregation_name == "pid3_lightbag" else None)
        self.mask_logit_refine = (
            nn.Sequential(
                nn.Conv2d(self.out_channels, self.feat_channels, 1, bias=False),
                nn.BatchNorm2d(self.feat_channels),
                nn.GELU(),
            ) if self.use_mask_logit_refine else None)
        self.query_evidence_norm = (
            nn.LayerNorm(self.feat_channels)
            if self.use_query_evidence_update else None)
        self.query_evidence_update = (
            nn.Sequential(
                nn.Linear(self.feat_channels, self.feat_channels * 2),
                nn.GELU(),
                nn.Linear(self.feat_channels * 2, self.feat_channels),
            ) if self.use_query_evidence_update else None)
        self.semantic_group_mixer = (
            SemanticGroupTokenMixer(self.feat_channels, self.out_channels)
            if self.use_semantic_group_mixer else None)
        if self.use_relation_topology_mixer:
            mixer_cls = (
                EdgeGatedTopologyTokenMixer
                if self.relation_topology_mixer_type == "edge_gated"
                else HardTopologyTokenMixer)
            self.relation_topology_mixer = mixer_cls(
                self.feat_channels,
                int(num_heads),
                topology_pattern=self.relation_topology_pattern)
        else:
            self.relation_topology_mixer = None
        object_refine_context_dim = (
            self.feat_channels * 3
            if self.use_foreground_context_object_refine
            else self.feat_channels * 2)
        self.hand_object_refine_norm = (
            nn.LayerNorm(object_refine_context_dim)
            if self.use_hand_conditioned_object_refine else None)
        self.hand_object_refine_update = (
            nn.Sequential(
                nn.Linear(object_refine_context_dim, self.feat_channels * 2),
                nn.GELU(),
                nn.Linear(self.feat_channels * 2, self.feat_channels),
            ) if self.use_hand_conditioned_object_refine else None)
        pixel_attn_heads = (
            int(num_heads) if pixel_self_attn_num_heads is None
            else int(pixel_self_attn_num_heads))
        self.pixel_self_attn = nn.Sequential(*[
            WindowPixelSelfAttention(
                self.feat_channels,
                num_heads=pixel_attn_heads,
                window_size=int(pixel_self_attn_window_size))
            for _ in range(self.pixel_self_attn_num_layers)
        ])
        self.prototype_extractor = (
            RelationPrototypeExtractor(
                self.feat_channels,
                num_slots=4,
                num_heads=int(num_heads))
            if self.use_relation_prototypes else None)
        self.dynamic_query_norm = (
            nn.LayerNorm(self.feat_channels) if self.use_relation_prototypes else None)
        self.dynamic_query_logit_scale = (
            nn.Parameter(torch.full((4, ), _logit(dynamic_query_init_scale)))
            if self.use_relation_prototypes else None)
        self.object_support_embed = (
            nn.Linear(self.feat_channels, self.feat_channels)
            if self.use_object_support_aux else None)
        self.hand_condition_object_refine = (
            nn.Sequential(
                ConvBNAct(self.feat_channels + 3, self.feat_channels, 3),
                ConvBNAct(self.feat_channels, self.feat_channels, 3),
            ) if self.use_hand_condition_lowres else None)
        slotema_types = (
            "slotema",
            "slotema-gated",
            "slotema-gated-probema",
        )
        self.streaming_memory_adapter = (
            SlotEMAVideoMemoryAdapter(
                self.feat_channels,
                num_slots=self.out_channels,
                use_read_gate=self.video_memory_type in (
                    "slotema-gated", "slotema-gated-probema"),
                use_update_gate=self.video_memory_type in (
                    "slotema-gated", "slotema-gated-probema"),
                use_prob_ema=self.video_memory_type == "slotema-gated-probema")
            if self.video_memory_type in slotema_types else None)
        self.video_memory_adapter = (
            None if self.video_memory_type in ("none", "") + slotema_types
            else TrueVideoMemoryAdapter(
                self.feat_channels,
                memory_type=self.video_memory_type,
                num_slots=self.out_channels))

    @staticmethod
    def _masked_pool(pixel_feat: Tensor,
                     mask_prob: Tensor,
                     eps: float = 1e-6) -> Tensor:
        weights = mask_prob.to(dtype=pixel_feat.dtype)
        denom = weights.flatten(2).sum(-1).clamp_min(float(eps))
        return torch.einsum("bnhw,bchw->bnc", weights, pixel_feat) / denom.unsqueeze(-1)

    def _mask_kernels(self, normed_tokens: Tensor) -> Tensor:
        if self.mask_embed_per_slot is None:
            return self.mask_embed(normed_tokens)
        kernels = [
            head(normed_tokens[:, idx])
            for idx, head in enumerate(self.mask_embed_per_slot)
        ]
        return torch.stack(kernels, dim=1)

    def _class_logits(self, normed_tokens: Tensor) -> Tensor:
        logits = self.class_embed(normed_tokens)
        if not self.use_hybrid_hand_object_queries:
            return logits
        bias = logits.new_full(logits.shape, -20.0)
        no_object_idx = self.class_out_channels - 1
        # class order:
        #   6-way: LH, RH, LO, RO, TO, no-object
        #   7-way: LH, RH, LO, RO, TO, distractor, no-object
        bias[:, 0, [0, no_object_idx]] = 0.0
        bias[:, 1, [1, no_object_idx]] = 0.0
        if logits.shape[1] > 2:
            bias[:, 2:, 2:self.class_out_channels] = 0.0
        return logits + bias

    def _object_context_from_hands(self, hand_context: Tensor) -> Tensor:
        """Map LH/RH evidence to LO/RO/TO without changing tensor shape."""
        left_idx, right_idx = (
            (0, 1) if self.hand_object_mapping == "matched" else (1, 0))
        return torch.stack(
            [
                hand_context[:, left_idx],
                hand_context[:, right_idx],
                0.5 * (hand_context[:, 0] + hand_context[:, 1]),
            ],
            dim=1)

    def _pack_outputs(self,
                      all_logits: List[Tensor],
                      all_class_logits: List[Tensor]) -> Dict[str, Tensor]:
        if self.out_channels == 5:
            prob5_layers = [logits.sigmoid() for logits in all_logits]
            all_logits4 = [
                torch.logit(
                    torch.stack(
                        [
                            prob5[:, 0],
                            prob5[:, 1],
                            torch.maximum(prob5[:, 2], prob5[:, 4]),
                            torch.maximum(prob5[:, 3], prob5[:, 4]),
                        ],
                        dim=1).clamp(1e-4, 1.0 - 1e-4))
                for prob5 in prob5_layers
            ]
            result = dict(
                logits5=all_logits[-1],
                all_logits5=all_logits,
                logits4=all_logits4[-1],
                all_logits4=all_logits4)
        else:
            result = dict(
                logits4=all_logits[-1],
                all_logits4=all_logits)
        result.update(
            class_logits=all_class_logits[-1],
            all_class_logits=all_class_logits)
        return result

    def _forward_slots(self,
                       queries: Tensor,
                       relation_feat: Tensor,
                       coarse_prob: Tensor,
                       coarse_reliability: Optional[Tensor] = None
                       ) -> tuple[List[Tensor], List[Tensor], Tensor, Optional[Tensor], Optional[Tensor]]:
        all_logits = []
        all_class_logits = []
        hand_condition_low = None
        object_relation_feat = None
        slot_tokens = queries
        current_relation_feat = relation_feat
        prev_logits = None
        for layer in self.slot_mixer.layers:
            slot_tokens = layer(slot_tokens)
            if self.query_evidence_update is not None and prev_logits is not None:
                evidence = self._masked_pool(
                    current_relation_feat, prev_logits.sigmoid())
                slot_tokens = slot_tokens + self.query_evidence_update(
                    self.query_evidence_norm(evidence))
            if self.semantic_group_mixer is not None:
                slot_tokens = self.semantic_group_mixer(slot_tokens)
            if self.relation_topology_mixer is not None:
                slot_tokens = self.relation_topology_mixer(slot_tokens)
            normed_tokens = self.slot_norm(slot_tokens)
            mask_kernels = self._mask_kernels(normed_tokens)
            logits = torch.einsum(
                "bnc,bchw->bnhw", mask_kernels, current_relation_feat)
            if self.hand_object_refine_update is not None:
                hand_context = self._masked_pool(
                    current_relation_feat, logits[:, :2].sigmoid())
                object_context = self._object_context_from_hands(hand_context)
                if self.use_foreground_context_object_refine:
                    fg_mask = coarse_prob
                    if coarse_reliability is not None:
                        fg_mask = (fg_mask * coarse_reliability).clamp(0.0, 1.0)
                    fg_context = self._masked_pool(
                        current_relation_feat, fg_mask).expand(-1, 3, -1)
                    object_context = torch.cat([object_context, fg_context], dim=-1)
                object_tokens = slot_tokens[:, 2:5]
                object_update = self.hand_object_refine_update(
                    self.hand_object_refine_norm(
                        torch.cat([object_tokens, object_context], dim=-1)))
                slot_tokens = torch.cat(
                    [slot_tokens[:, :2], object_tokens + object_update], dim=1)
                normed_tokens = self.slot_norm(slot_tokens)
                mask_kernels = self._mask_kernels(normed_tokens)
                logits = torch.einsum(
                    "bnc,bchw->bnhw", mask_kernels, current_relation_feat)
            if self.hand_condition_object_refine is not None:
                hand_condition_low = logits[:, :2].sigmoid()
                if self.detach_hand_condition:
                    hand_condition_low = hand_condition_low.detach()
                object_relation_feat = self.hand_condition_object_refine(
                    torch.cat([current_relation_feat, coarse_prob, hand_condition_low], dim=1))
                object_logits = torch.einsum(
                    "bnc,bchw->bnhw", mask_kernels[:, 2:4], object_relation_feat)
                logits = torch.cat([logits[:, :2], object_logits], dim=1)
            all_logits.append(logits)
            all_class_logits.append(self._class_logits(normed_tokens))
            prev_logits = logits
            if self.mask_logit_refine is not None:
                current_relation_feat = relation_feat + self.mask_logit_refine(
                    logits.sigmoid())
        if self.slot_mixer.norm is not None:
            slot_tokens = self.slot_mixer.norm(slot_tokens)
        return all_logits, all_class_logits, self.slot_norm(slot_tokens), hand_condition_low, object_relation_feat

    def _fuse_features(self, feats: Sequence[Tensor]) -> Tensor:
        if len(feats) != len(self.proj):
            raise ValueError(f"Expected {len(self.proj)} feature maps, got {len(feats)}")
        target_size = feats[0].shape[-2:]
        fused = 0.0
        for feat, proj in zip(feats, self.proj):
            value = proj(feat)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=self.align_corners)
            fused = fused + value
        return fused / float(len(feats))

    def _coarse_prior_from_logits(self, coarse_logits: Tensor) -> Tensor:
        if self.coarse_prior_activation == "identity":
            return coarse_logits
        return coarse_logits.sigmoid()

    @staticmethod
    def _coarse_uncertainty(coarse_prob: Tensor) -> Tensor:
        prob = coarse_prob.clamp(0.0, 1.0)
        return (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)

    def _maybe_perturb_coarse_prob(self, coarse_prob: Tensor) -> Tensor:
        if not self.training:
            return coarse_prob
        prob = coarse_prob.clamp(0.0, 1.0)
        if (self.coarse_prior_perturb_prob > 0.0 and
                self.coarse_prior_noise_std > 0.0):
            active = (
                torch.rand(
                    prob.shape[0], 1, 1, 1,
                    device=prob.device,
                    dtype=prob.dtype) < self.coarse_prior_perturb_prob)
            noise = torch.randn_like(prob) * self.coarse_prior_noise_std
            prob = torch.where(active, prob + noise, prob).clamp(0.0, 1.0)
        if self.coarse_prior_dropout_prob > 0.0:
            drop = torch.rand_like(prob) < self.coarse_prior_dropout_prob
            unknown = prob.new_full((), 0.5)
            prob = torch.where(drop, unknown, prob)
        return prob

    def _current_train_iter(self) -> tuple[int, bool]:
        try:
            cur_iter = MessageHub.get_current_instance().get_info("iter")
        except Exception:
            cur_iter = None
        if cur_iter is not None:
            return int(cur_iter), True
        return int(self._gt_coarse_prior_fallback_iter.item()), False

    def _gt_coarse_prior_weight(self) -> tuple[float, bool]:
        if (not self.training or self.gt_coarse_prior_anneal_iters <= 0 or
                self.gt_coarse_prior_start_weight <= 0.0):
            return 0.0, True
        cur_iter, from_message_hub = self._current_train_iter()
        progress = min(
            max(float(cur_iter) / float(self.gt_coarse_prior_anneal_iters), 0.0),
            1.0)
        weight = self.gt_coarse_prior_start_weight * (1.0 - progress)
        return float(max(weight, 0.0)), from_message_hub

    def _coarse_target_from_pilot(
            self,
            mask_pilot_targets: Optional[Tensor],
            target_size: tuple[int, int],
            dtype: torch.dtype,
            device: torch.device) -> Optional[Tensor]:
        if mask_pilot_targets is None:
            return None
        target = mask_pilot_targets.to(device=device, dtype=dtype)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        elif target.dim() == 4:
            target = target.max(dim=1, keepdim=True).values
        else:
            return None
        if target.shape[-2:] != target_size:
            target = F.interpolate(
                target,
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners)
        return target.clamp(0.0, 1.0)

    def _maybe_mix_gt_coarse_prior(
            self,
            coarse_prob: Tensor,
            mask_pilot_targets: Optional[Tensor]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        weight, from_message_hub = self._gt_coarse_prior_weight()
        if weight <= 0.0:
            return coarse_prob, None, None
        gt_coarse = self._coarse_target_from_pilot(
            mask_pilot_targets,
            coarse_prob.shape[-2:],
            coarse_prob.dtype,
            coarse_prob.device)
        if gt_coarse is None:
            return coarse_prob, None, None
        mixed_prob = coarse_prob * (1.0 - weight) + gt_coarse * weight
        if self.training and not from_message_hub:
            self._gt_coarse_prior_fallback_iter.add_(1)
        weight_tensor = coarse_prob.new_full((coarse_prob.shape[0], 1, 1, 1), weight)
        return mixed_prob.clamp(0.0, 1.0), gt_coarse, weight_tensor

    def _coarse_prior_components(
            self,
            pixel_feat: Tensor,
            coarse_logits: Tensor,
            mask_pilot_targets: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        coarse_prob = self._coarse_prior_from_logits(coarse_logits)
        coarse_prob, gt_coarse_prob, gt_coarse_weight = (
            self._maybe_mix_gt_coarse_prior(coarse_prob, mask_pilot_targets))
        coarse_prob = self._maybe_perturb_coarse_prob(coarse_prob)
        uncertainty = self._coarse_uncertainty(coarse_prob)
        reliability_logits = None
        if self.coarse_reliability_head is not None:
            reliability_logits = self.coarse_reliability_head(
                pixel_feat, coarse_prob, uncertainty)
            reliability = reliability_logits.sigmoid()
        else:
            reliability = (1.0 - uncertainty).clamp(0.0, 1.0)
        if self.use_coarse_uncertainty_prior:
            components = (coarse_prob, uncertainty, reliability)
            prior = torch.cat([
                value if enabled else torch.zeros_like(value)
                for value, enabled in zip(components, self.coarse_condition_mask)
            ], dim=1)
        else:
            prior = coarse_prob
        return (
            prior, coarse_prob, uncertainty, reliability, reliability_logits,
            gt_coarse_prob, gt_coarse_weight)

    def forward(self,
                feats: Sequence[Tensor],
                depth_prior: Optional[Tensor] = None,
                mask_pilot_targets: Optional[Tensor] = None,
                video_memory_state: Optional[Dict[str, Tensor]] = None
                ) -> Dict[str, Tensor]:
        pixel_feat = self._fuse_features(feats)
        coarse_logits = self.coarse_head(pixel_feat)
        (coarse_prior, coarse_prob, coarse_uncertainty, coarse_reliability,
         reliability_logits, gt_coarse_prob, gt_coarse_weight) = (
            self._coarse_prior_components(
                pixel_feat, coarse_logits, mask_pilot_targets))
        prior_parts = [pixel_feat, coarse_prior]
        if self.use_depth_prior:
            if depth_prior is None:
                depth_prior = pixel_feat.new_zeros(
                    pixel_feat.shape[0], 1, *pixel_feat.shape[-2:])
            if depth_prior.shape[-2:] != pixel_feat.shape[-2:]:
                depth_prior = F.interpolate(
                    depth_prior,
                    size=pixel_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners)
            if depth_prior.shape[1] != 1:
                depth_prior = depth_prior[:, :1]
            prior_parts.append(depth_prior.clamp(0.0, 1.0))
        contact_guidance_logits = None
        pid_boundary_logits = None
        if self.coarse_reliability_aggregator is not None:
            relation_feat = self.coarse_reliability_aggregator(
                pixel_feat,
                coarse_prob,
                coarse_uncertainty,
                coarse_reliability)
        elif self.coarse_contact_guidance is not None:
            relation_feat, contact_guidance_logits = self.coarse_contact_guidance(
                pixel_feat, coarse_prob)
        else:
            relation_feat = self.refine(torch.cat(prior_parts, dim=1))
            if self.refine_scse is not None:
                relation_feat = self.refine_scse(relation_feat)
        if self.pid_aggregation is not None:
            relation_feat, pid_boundary_logits = self.pid_aggregation(
                pixel_feat, relation_feat, coarse_prob)
        if self.concat_attention is not None:
            relation_feat = self.concat_attention(relation_feat)
        if self.pixel_self_attn_num_layers > 0:
            relation_feat = self.pixel_self_attn(relation_feat)
        queries = self.slot_queries.unsqueeze(0).expand(relation_feat.shape[0], -1, -1)
        memory_aux = {}
        if self.video_memory_adapter is not None:
            relation_feat, memory_aux = self.video_memory_adapter.read(
                relation_feat, coarse_prob, video_memory_state)
        if self.decoder_attention is not None:
            relation_feat = self.decoder_attention(relation_feat)
        dynamic_prototypes = None
        prototype_attn = None
        relation_dynamic = None
        object_support_logits2 = None
        hand_condition_low = None
        object_relation_feat = None
        if self.prototype_extractor is not None:
            dynamic_prototypes, prototype_attn = self.prototype_extractor(pixel_feat)
            relation_dynamic = torch.stack([
                dynamic_prototypes[:, 0],
                dynamic_prototypes[:, 1],
                dynamic_prototypes[:, 2] + dynamic_prototypes[:, 0],
                dynamic_prototypes[:, 3] + dynamic_prototypes[:, 1],
            ], dim=1)
            relation_dynamic = self.dynamic_query_norm(relation_dynamic)
            scale = torch.sigmoid(self.dynamic_query_logit_scale).view(1, 4, 1)
            queries = queries + relation_dynamic * scale
            if self.object_support_embed is not None:
                object_kernels = self.object_support_embed(dynamic_prototypes[:, 2:4])
                object_support_logits2 = torch.einsum(
                    "bnc,bchw->bnhw", object_kernels, relation_feat)
        if self.use_global_query_bias:
            pooled = F.adaptive_avg_pool2d(relation_feat * (0.5 + coarse_prob), 1).flatten(1)
            queries = queries + pooled.unsqueeze(1)
        if self.reliability_query_proj is not None:
            trusted = (coarse_prob * coarse_reliability).clamp(0.0, 1.0)
            uncertain = (coarse_prob * coarse_uncertainty).clamp(0.0, 1.0)
            trusted_context = self._masked_pool(relation_feat, trusted).squeeze(1)
            uncertain_context = self._masked_pool(relation_feat, uncertain).squeeze(1)
            query_delta = self.reliability_query_proj(
                torch.cat([trusted_context, uncertain_context], dim=-1))
            queries = queries + query_delta.unsqueeze(1)
        if self.streaming_memory_adapter is not None:
            queries, slot_memory_aux = self.streaming_memory_adapter.read(
                queries, video_memory_state)
            memory_aux.update(slot_memory_aux)
        all_logits, all_class_logits, slot_tokens, hand_condition_low, object_relation_feat = (
            self._forward_slots(
                queries, relation_feat, coarse_prob, coarse_reliability))
        if self.streaming_memory_adapter is not None and all_logits:
            all_logits[-1] = self.streaming_memory_adapter.blend_logits(
                all_logits[-1], video_memory_state)
        if not self.return_intermediate:
            all_logits = all_logits[-1:]
            all_class_logits = all_class_logits[-1:]
        result = self._pack_outputs(all_logits, all_class_logits)
        result.update(
            coarse_logits=coarse_logits,
            slot_tokens=slot_tokens,
            relation_feat=relation_feat,
            pixel_feat=pixel_feat,
            dynamic_prototypes=dynamic_prototypes,
            prototype_attn=prototype_attn,
            relation_dynamic=relation_dynamic,
            object_support_logits2=object_support_logits2,
            contact_guidance_logits=contact_guidance_logits,
            pid_boundary_logits=pid_boundary_logits,
            coarse_prior_prob=coarse_prob,
            coarse_uncertainty=coarse_uncertainty,
            coarse_reliability=coarse_reliability,
            coarse_reliability_logits=reliability_logits,
            gt_coarse_prior_prob=gt_coarse_prob,
            gt_coarse_prior_weight=gt_coarse_weight,
            hand_condition_low=hand_condition_low,
            object_relation_feat=object_relation_feat,
            memory_aux=memory_aux)
        return result

    def update_video_memory(self,
                            state: Optional[Dict[str, Tensor]],
                            relation_feat: Tensor,
                            masks: Tensor,
        coarse_prob: Tensor) -> Optional[Dict[str, Tensor]]:
        if self.video_memory_adapter is None:
            if self.streaming_memory_adapter is None:
                return state
            return self.streaming_memory_adapter.update(
                state, relation_feat, masks, coarse_prob)
        return self.video_memory_adapter.update(
            state, relation_feat, masks, coarse_prob)




@MODELS.register_module()
class LiteHOSV2PropagationRefinerHead(nn.Module):
    """Fixed-slot VOS-style propagation and conservative refinement head."""

    def __init__(self,
                 in_channels: int = 64,
                 state_channels: int = 192,
                 hidden_channels: int = 64,
                 align_corners: bool = False) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.state_channels = int(state_channels)
        self.hidden_channels = int(hidden_channels)
        self.align_corners = bool(align_corners)
        self.state_proj = nn.Linear(self.state_channels, self.hidden_channels)
        self.affine_head = nn.Sequential(
            nn.Linear(self.hidden_channels + self.in_channels, self.hidden_channels),
            nn.GELU(),
            nn.Linear(self.hidden_channels, 6),
        )
        self.offset_head = nn.Sequential(
            ConvBNAct(self.in_channels + 5, self.hidden_channels, 3),
            nn.Conv2d(self.hidden_channels, 2, 3, padding=1),
            nn.Tanh(),
        )
        self.refine = nn.Sequential(
            ConvBNAct(self.in_channels + 5, self.hidden_channels, 3),
            ConvBNAct(self.hidden_channels, self.hidden_channels, 3),
            nn.Conv2d(self.hidden_channels, 4, 1),
        )
        self.gate = nn.Sequential(
            ConvBNAct(self.in_channels + 5, self.hidden_channels, 3),
            nn.Conv2d(self.hidden_channels, 4, 1),
        )
        self._init_identity_affine()

    def _init_identity_affine(self) -> None:
        last = self.affine_head[-1]
        nn.init.zeros_(last.weight)
        bias = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        with torch.no_grad():
            last.bias.copy_(bias)

    def _warp(self, x: Tensor, theta: Tensor, offset: Tensor) -> Tensor:
        b, n, h, w = x.shape
        warped = []
        for slot in range(n):
            grid = F.affine_grid(
                theta[:, slot],
                size=(b, 1, h, w),
                align_corners=self.align_corners)
            if offset.shape[-2:] != (h, w):
                off = F.interpolate(
                    offset,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=self.align_corners)
            else:
                off = offset
            grid = grid + off.permute(0, 2, 3, 1) * 0.05
            warped_slot = F.grid_sample(
                x[:, slot:slot + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=self.align_corners)
            warped.append(warped_slot)
        return torch.cat(warped, dim=1)

    def forward(self,
                current_feat: Tensor,
                prev_state: LiteHOSRelationState) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mask_prob4 = prev_state.mask_prob4.to(device=current_feat.device, dtype=current_feat.dtype)
        coarse_prob = prev_state.coarse_prob.to(device=current_feat.device, dtype=current_feat.dtype)
        if mask_prob4.shape[-2:] != current_feat.shape[-2:]:
            mask_low = F.interpolate(mask_prob4, current_feat.shape[-2:], mode="bilinear", align_corners=False)
        else:
            mask_low = mask_prob4
        if coarse_prob.shape[-2:] != current_feat.shape[-2:]:
            coarse_low = F.interpolate(coarse_prob, current_feat.shape[-2:], mode="bilinear", align_corners=False)
        else:
            coarse_low = coarse_prob

        feat_pool = F.adaptive_avg_pool2d(current_feat, 1).flatten(1)
        token = prev_state.slot_tokens.to(device=current_feat.device, dtype=current_feat.dtype)
        token = self.state_proj(token)
        affine_in = torch.cat([
            token,
            feat_pool.unsqueeze(1).expand(-1, 4, -1),
        ], dim=-1)
        theta = self.affine_head(affine_in).view(current_feat.shape[0], 4, 2, 3)
        offset = self.offset_head(torch.cat([current_feat, mask_low, coarse_low], dim=1))
        warped_low = self._warp(mask_low, theta, offset)
        warped = F.interpolate(
            warped_low,
            size=mask_prob4.shape[-2:],
            mode="bilinear",
            align_corners=False)
        coarse_warped = F.interpolate(
            self._warp(coarse_low, theta[:, :1], offset),
            size=mask_prob4.shape[-2:],
            mode="bilinear",
            align_corners=False)

        refine_in = torch.cat([current_feat, warped_low, coarse_low], dim=1)
        repair_logits_low = self.refine(refine_in)
        gate_low = torch.sigmoid(self.gate(refine_in))
        repair_logits = F.interpolate(
            repair_logits_low,
            size=mask_prob4.shape[-2:],
            mode="bilinear",
            align_corners=False)
        gate = F.interpolate(
            gate_low,
            size=mask_prob4.shape[-2:],
            mode="bilinear",
            align_corners=False)
        warped_logits = torch.logit(warped.clamp(1e-4, 1.0 - 1e-4))
        logits4 = (1.0 - gate) * warped_logits + gate * repair_logits
        return logits4, warped, coarse_warped, theta


@MODELS.register_module()
class LiteHOSV2RouteGateHead(nn.Module):
    """Supervised three-action gate used after the fast encoder."""

    def __init__(self,
                 in_channels: int = 64,
                 hidden_channels: int = 64,
                 max_age: int = 5) -> None:
        super().__init__()
        self.max_age = int(max_age)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 8, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 3),
        )

    def forward(self,
                current_feat: Tensor,
                prev_state: Optional[LiteHOSRelationState] = None) -> Tensor:
        b = current_feat.shape[0]
        pooled = F.adaptive_avg_pool2d(current_feat, 1).flatten(1)
        if prev_state is None:
            stats = current_feat.new_zeros((b, 8))
            logits = self.mlp(torch.cat([pooled, stats], dim=1))
            logits[:, 2] = logits[:, 2] + 4.0
            return logits

        prob = prev_state.mask_prob4.to(device=current_feat.device, dtype=current_feat.dtype)
        age = prev_state.age.to(device=current_feat.device, dtype=current_feat.dtype).view(b, 1)
        entropy = prev_state.uncertainty4.to(device=current_feat.device, dtype=current_feat.dtype).mean(1, keepdim=True)
        area = prob.flatten(2).mean(-1)
        violation = relation_violation_score(prob).view(b, 1)
        age_norm = age / max(float(self.max_age), 1.0)
        stats = torch.cat([age_norm, entropy, area, violation, current_feat.new_zeros((b, 1))], dim=1)
        logits = self.mlp(torch.cat([pooled, stats], dim=1))
        stale = (age.view(-1) >= float(self.max_age)).to(dtype=logits.dtype)
        logits[:, 2] = logits[:, 2] + stale * 4.0
        no_hand_score = 1.0 - area[:, :2].amax(dim=1)
        logits[:, 0] = logits[:, 0] + no_hand_score
        return logits
