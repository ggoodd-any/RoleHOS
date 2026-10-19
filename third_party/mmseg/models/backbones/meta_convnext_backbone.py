# Copyright (c) OpenMMLab. All rights reserved.
"""Meta ConvNeXt and EUPE ViT wrappers used by LiteHOSV2 ablations."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmseg.registry import MODELS




def _expand_rgb_stem(source: torch.Tensor,
                     target_shape: Sequence[int],
                     extra_channel_init: str = "zeros") -> torch.Tensor:
    """Copy a pretrained RGB stem and deterministically append zero channels."""
    target_shape = tuple(int(v) for v in target_shape)
    compatible = (
        source.ndim == 4 and len(target_shape) == 4 and
        source.shape[0] == target_shape[0] and
        source.shape[2:] == target_shape[2:] and
        source.shape[1] == 3 and target_shape[1] > 3)
    if not compatible:
        raise ValueError(
            f"Cannot adapt RGB stem from {tuple(source.shape)} to {target_shape}")
    if extra_channel_init != "zeros":
        raise ValueError(
            "extra_channel_init currently supports only 'zeros', "
            f"got {extra_channel_init!r}")
    expanded = source.new_zeros(target_shape)
    expanded[:, :3].copy_(source)
    return expanded


def _load_checkpoint(model: torch.nn.Module,
                     checkpoint: str,
                     arch: str,
                     extra_channel_init: str = "zeros") -> None:
    if not checkpoint:
        return
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing {arch} checkpoint: {path}")
    state_dict = torch.load(str(path), map_location="cpu")
    if isinstance(state_dict, dict):
        state_dict = state_dict.get("state_dict") or state_dict.get("model") or state_dict
    stem_key = "downsample_layers.0.0.weight"
    if stem_key in state_dict:
        source = state_dict[stem_key]
        target = model.state_dict()[stem_key]
        if source.shape != target.shape:
            state_dict = dict(state_dict)
            state_dict[stem_key] = _expand_rgb_stem(
                source, target.shape, extra_channel_init=extra_channel_init)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {arch} from {path}")
    print(f"missing keys when loading {arch}: {missing}")
    print(f"unexpected keys when loading {arch}: {unexpected}")


@MODELS.register_module()
class ConvNeXtDINOBackbone(BaseModule):
    """DINOv3 ConvNeXt feature wrapper with four NCHW feature outputs."""

    def __init__(self,
                 arch: str = "tiny",
                 checkpoint: str = "",
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 in_channels: int = 3,
                 extra_channel_init: str = "zeros",
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        from eupe.models.convnext import ConvNeXt, convnext_sizes

        if arch not in convnext_sizes:
            raise ValueError(f"Unsupported ConvNeXt arch {arch!r}")
        if int(in_channels) not in (3, 4, 5):
            raise ValueError("in_channels must be one of 3, 4, or 5")
        if str(extra_channel_init) != "zeros":
            raise ValueError("extra_channel_init must be 'zeros'")
        size = convnext_sizes[arch]
        self.in_channels = int(in_channels)
        self.extra_channel_init = str(extra_channel_init)
        self.model = ConvNeXt(
            depths=size["depths"], dims=size["dims"], in_chans=self.in_channels)
        self.out_indices = tuple(int(i) for i in out_indices)
        self.out_channels = [size["dims"][i] for i in self.out_indices]
        _load_checkpoint(
            self.model,
            checkpoint,
            f"dinov3_convnext_{arch}",
            extra_channel_init=self.extra_channel_init)

    def forward(self, x):
        return self.model.get_intermediate_layers(
            x, n=self.out_indices, reshape=True, norm=True)

    def init_weights(self) -> None:
        return None


