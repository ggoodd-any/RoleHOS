# Copyright (c) OpenMMLab. All rights reserved.
"""Standalone LiteHOSV2 V0 relation-state segmentor."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils.portable_points import point_sample
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import OptConfigType, OptMultiConfig, OptSampleList, SampleList
from ..utils import get_uncertain_point_coords_with_randomness, resize
from ..utils.relation_state import (
    LiteHOSRelationState,
    ROUTE_NAMES,
    foreground5_prob_to_multilabel4_prob,
    foreground5_to_compat6,
    labels6_to_foreground5,
    labels6_to_multilabel4,
    labels6_to_softmax6,
    make_state_from_logits4,
    multilabel4_to_compat6,
    physical_relation_loss,
    softmax6_prob_to_multilabel4_prob,
    softmax6_to_compat6,
)
from .base import BaseSegmentor


def _cuda_sync_if_needed(tensor: Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _elapsed_ms(start: float) -> float:
    return float((time.perf_counter() - start) * 1000.0)


@MODELS.register_module()
class LiteHOSV2RelationStateSegmentor(BaseSegmentor):
    """V0 four-slot streaming HOS model.

    The public research interface is :meth:`forward_relation`, which can run a
    single slow refresh, a cached fast propagation/refinement, or an automatic
    gate-selected route.
    """

    def __init__(self,
                 backbone: Optional[dict],
                 fast_encoder: dict,
                 slow_head: dict,
                 fast_head: dict,
                 gate_head: dict,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 rgbd_early_fusion: bool = False,
                 depth_norm_mean: float = 0.5,
                 depth_norm_std: float = 0.25,
                 semantic_output_mode: str = "multilabel4",
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone) if backbone is not None else None
        self.fast_encoder = MODELS.build(dict(type="LiteHOSV2FastEncoder", **fast_encoder))
        slow_head = dict(slow_head)
        slow_head.setdefault("type", "LiteHOSV2RelationSlotHead")
        self.slow_head = MODELS.build(slow_head)
        self.fast_head = MODELS.build(dict(type="LiteHOSV2PropagationRefinerHead", **fast_head))
        self.gate_head = MODELS.build(dict(type="LiteHOSV2RouteGateHead", **gate_head))
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}
        self.rgbd_early_fusion = bool(rgbd_early_fusion)
        self.depth_norm_mean = float(depth_norm_mean)
        self.depth_norm_std = max(float(depth_norm_std), 1e-6)
        self.semantic_output_mode = str(semantic_output_mode)
        if self.semantic_output_mode not in (
                "multilabel4", "fixed4", "fixed5", "hungarian5",
                "hungarian5_sam_distractor", "softmax6"):
            raise ValueError(
                "semantic_output_mode must be 'multilabel4', 'fixed4', 'fixed5', "
                "'hungarian5', 'hungarian5_sam_distractor', or 'softmax6', "
                f"got {self.semantic_output_mode!r}")
        self.align_corners = bool(self.test_cfg.get("align_corners", False))
        if self.semantic_output_mode == "softmax6":
            self.num_classes = 6
        elif self.semantic_output_mode in (
                "fixed5", "hungarian5", "hungarian5_sam_distractor"):
            self.num_classes = 5
        else:
            self.num_classes = 4
        self.out_channels = self.num_classes
        self.loss_bce_weight = float(self.train_cfg.get("loss_bce_weight", 1.0))
        self.loss_dice_weight = float(self.train_cfg.get("loss_dice_weight", 1.0))
        self.loss_tversky_weight = float(
            self.train_cfg.get("loss_tversky_weight", 0.0))
        self.tversky_alpha = float(self.train_cfg.get("tversky_alpha", 0.3))
        self.tversky_beta = float(self.train_cfg.get("tversky_beta", 0.7))
        self.tversky_eps = float(self.train_cfg.get("tversky_eps", 1.0))
        self.loss_point_bce_weight = float(
            self.train_cfg.get("loss_point_bce_weight", 0.0))
        self.loss_point_dice_weight = float(
            self.train_cfg.get("loss_point_dice_weight", 0.0))
        self.point_loss_num_points = int(
            self.train_cfg.get("point_loss_num_points", 4096))
        self.point_loss_oversample_ratio = float(
            self.train_cfg.get("point_loss_oversample_ratio", 3.0))
        self.point_loss_importance_sample_ratio = float(
            self.train_cfg.get("point_loss_importance_sample_ratio", 0.75))
        if self.point_loss_num_points < 0:
            raise ValueError("point_loss_num_points must be non-negative")
        if self.point_loss_oversample_ratio < 1.0:
            raise ValueError("point_loss_oversample_ratio must be >= 1")
        if not 0.0 <= self.point_loss_importance_sample_ratio <= 1.0:
            raise ValueError(
                "point_loss_importance_sample_ratio must be in [0, 1]")
        slot_loss_weights = self.train_cfg.get("slot_loss_weights", None)
        if slot_loss_weights is None:
            self.register_buffer("slot_loss_weights", None, persistent=False)
        else:
            slot_loss_weights = [float(value) for value in slot_loss_weights]
            if len(slot_loss_weights) != self.num_classes:
                raise ValueError(
                    "slot_loss_weights must contain "
                    f"{self.num_classes} weights for mode "
                    f"{self.semantic_output_mode!r}, got "
                    f"{len(slot_loss_weights)}")
            self.register_buffer(
                "slot_loss_weights",
                torch.tensor(slot_loss_weights, dtype=torch.float32),
                persistent=False)
        self.use_fixed5_logit_calibration = bool(
            self.train_cfg.get("fixed5_logit_calibration", False))
        self.fixed5_logit_bias = None
        self.fixed5_logit_log_scale = None
        if self.use_fixed5_logit_calibration:
            if self.semantic_output_mode != "fixed5":
                raise ValueError(
                    "fixed5_logit_calibration requires semantic_output_mode='fixed5'")
            bias_init = self.train_cfg.get(
                "fixed5_logit_bias_init", [0.0] * self.num_classes)
            scale_init = self.train_cfg.get(
                "fixed5_logit_scale_init", [1.0] * self.num_classes)
            if len(bias_init) != self.num_classes or len(scale_init) != self.num_classes:
                raise ValueError(
                    "fixed5_logit_bias_init and fixed5_logit_scale_init must "
                    f"contain {self.num_classes} values")
            bias = torch.tensor(bias_init, dtype=torch.float32).view(
                1, self.num_classes, 1, 1)
            scale = torch.tensor(scale_init, dtype=torch.float32).clamp_min(
                1e-4).log().view(1, self.num_classes, 1, 1)
            self.fixed5_logit_bias = nn.Parameter(bias)
            self.fixed5_logit_log_scale = nn.Parameter(scale)
        self.loss_cls_weight = float(
            self.train_cfg.get(
                "loss_cls_weight",
                self.train_cfg.get("loss_query_cls_weight", 1.0)))
        self.loss_cls_no_object_weight = float(
            self.train_cfg.get("loss_cls_no_object_weight", 0.1))
        self.loss_mask_pilot_weight = float(
            self.train_cfg.get("loss_mask_pilot_weight", 1.0))
        softmax_class_weight = self.train_cfg.get("softmax_class_weight", None)
        if softmax_class_weight is None:
            self.softmax_class_weight = None
        else:
            self.softmax_class_weight = [
                float(value) for value in softmax_class_weight
            ]
            if len(self.softmax_class_weight) != self.num_classes:
                raise ValueError(
                    "softmax_class_weight must contain "
                    f"{self.num_classes} weights for mode "
                    f"{self.semantic_output_mode!r}, got "
                    f"{len(self.softmax_class_weight)}")
        self.loss_coarse_bce_weight = float(
            self.train_cfg.get("loss_coarse_bce_weight", 0.0))
        self.loss_coarse_dice_weight = float(
            self.train_cfg.get("loss_coarse_dice_weight", 0.0))
        self.loss_coarse_reliability_weight = float(
            self.train_cfg.get("loss_coarse_reliability_weight", 0.0))
        self.loss_compat6_ce_weight = float(
            self.train_cfg.get("loss_compat6_ce_weight", 0.0))
        self.loss_compat6_dice_weight = float(
            self.train_cfg.get("loss_compat6_dice_weight", 0.0))
        self.loss_to_consistency_weight = float(
            self.train_cfg.get("loss_to_consistency_weight", 0.0))
        self.loss_lr_exclusivity_weight = float(
            self.train_cfg.get("loss_lr_exclusivity_weight", 0.0))
        self.loss_contact_guidance_bce_weight = float(
            self.train_cfg.get("loss_contact_guidance_bce_weight", 0.0))
        self.loss_contact_guidance_dice_weight = float(
            self.train_cfg.get("loss_contact_guidance_dice_weight", 0.0))
        self.loss_cooccurrence_weight = float(
            self.train_cfg.get("loss_cooccurrence_weight", 0.0))
        self.loss_relation_consistency_weight = float(
            self.train_cfg.get("loss_relation_consistency_weight", 0.0))
        self.cooccurrence_area_tau = float(
            self.train_cfg.get("cooccurrence_area_tau", 150.0 / (448.0 * 448.0)))
        self.cooccurrence_temperature = float(
            self.train_cfg.get(
                "cooccurrence_temperature",
                max(self.cooccurrence_area_tau * 0.25, 1e-6)))
        self.loss_physical_weight = float(
            self.train_cfg.get("loss_physical_weight", 0.0))
        self.loss_video_memory_read_weight = float(
            self.train_cfg.get("loss_video_memory_read_weight", 0.0))
        self.loss_video_memory_read_gate_weight = float(
            self.train_cfg.get("loss_video_memory_read_gate_weight", 0.0))
        self.loss_video_memory_update_gate_weight = float(
            self.train_cfg.get("loss_video_memory_update_gate_weight", 0.0))
        self.video_memory_detach_state = bool(
            self.train_cfg.get("video_memory_detach_state", True))
        self.slot_loss_reduction = str(
            self.train_cfg.get("slot_loss_reduction", "mean"))
        if self.slot_loss_reduction not in ("mean", "sum"):
            raise ValueError(
                "slot_loss_reduction must be 'mean' or 'sum', "
                f"got {self.slot_loss_reduction!r}")
        self.loss_proto_coherence_weight = float(
            self.train_cfg.get("loss_proto_coherence_weight", 0.0))
        self.loss_object_support_bce_weight = float(
            self.train_cfg.get("loss_object_support_bce_weight", 0.0))
        self.loss_object_support_dice_weight = float(
            self.train_cfg.get("loss_object_support_dice_weight", 0.0))
        self.support_boundary_kernel = int(
            self.train_cfg.get("support_boundary_kernel", 15))
        self.support_contact_kernel = int(
            self.train_cfg.get("support_contact_kernel", 31))
        self.use_depth_aware_support = bool(
            self.train_cfg.get("use_depth_aware_support", False))
        self.depth_contact_tau = float(
            self.train_cfg.get("depth_contact_tau", 0.08))
        self.depth_contact_weight = float(
            self.train_cfg.get("depth_contact_weight", 0.7))
        self.contact2d_weight = float(
            self.train_cfg.get("contact2d_weight", 0.3))
        self.loss_object_support_absent_topk_weight = float(
            self.train_cfg.get("loss_object_support_absent_topk_weight", 0.0))
        self.loss_object_final_absent_topk_weight = float(
            self.train_cfg.get("loss_object_final_absent_topk_weight", 0.0))
        self.loss_fixed5_object_absent_topk_weight = float(
            self.train_cfg.get("loss_fixed5_object_absent_topk_weight", 0.0))
        self.loss_opposite_hand_absent_topk_weight = float(
            self.train_cfg.get("loss_opposite_hand_absent_topk_weight", 0.0))
        self.absent_topk_ratio = float(
            self.train_cfg.get("absent_topk_ratio", 0.01))
        self.absent_slot_area_eps = float(
            self.train_cfg.get("absent_slot_area_eps", 1e-6))
        self.physical_area_tau = float(
            self.train_cfg.get("physical_area_tau", 0.01))
        self.physical_support_kernel = int(
            self.train_cfg.get("physical_support_kernel", 31))
        self.physical_min_support_ratio = float(
            self.train_cfg.get("physical_min_support_ratio", 0.03))

    @property
    def with_decode_head(self) -> bool:
        return True

    def _rgbd_backbone_input(self,
                             inputs: Tensor,
                             depth_prior: Optional[Tensor]) -> Tensor:
        if not self.rgbd_early_fusion:
            return inputs
        if depth_prior is None:
            depth = inputs.new_zeros(inputs.shape[0], 1, *inputs.shape[-2:])
        else:
            depth = depth_prior.to(device=inputs.device, dtype=inputs.dtype)
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            if depth.shape[1] != 1:
                depth = depth[:, :1]
            if tuple(depth.shape[-2:]) != tuple(inputs.shape[-2:]):
                depth = resize(
                    depth,
                    size=inputs.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False)
            depth = depth.clamp(0.0, 1.0)
        depth = (depth - self.depth_norm_mean) / self.depth_norm_std
        return torch.cat([inputs, depth], dim=1)

    def extract_feat(self,
                     inputs: Tensor,
                     depth_prior: Optional[Tensor] = None) -> Sequence[Tensor]:
        backbone_inputs = self._rgbd_backbone_input(inputs, depth_prior)
        if self.backbone is None:
            feat = self.fast_encoder(inputs)
            return (feat, feat, feat, feat)
        feats = self.backbone(backbone_inputs)
        if not isinstance(feats, (list, tuple)):
            feats = (feats, )
        return feats

    def _extract_slow_feat(self,
                           inputs: Tensor,
                           depth_prior: Optional[Tensor] = None) -> Sequence[Tensor]:
        feats = self.extract_feat(inputs, depth_prior=depth_prior)
        if len(feats) == len(self.slow_head.in_channels):
            return feats
        if len(feats) == 1:
            return tuple(feats[0] for _ in self.slow_head.in_channels)
        raise ValueError(
            f"Slow head expects {len(self.slow_head.in_channels)} features, got {len(feats)}"
        )

    def _apply_fixed5_logit_calibration(
            self, logits: Optional[Tensor]) -> Optional[Tensor]:
        if logits is None or not self.use_fixed5_logit_calibration:
            return logits
        assert self.fixed5_logit_bias is not None
        assert self.fixed5_logit_log_scale is not None
        bias = self.fixed5_logit_bias.to(device=logits.device, dtype=logits.dtype)
        scale = self.fixed5_logit_log_scale.exp().to(
            device=logits.device, dtype=logits.dtype)
        return logits * scale + bias

    def _slow(self,
              inputs: Tensor,
              depth_prior: Optional[Tensor] = None,
              mask_pilot_targets: Optional[Tensor] = None,
              video_memory_state: Optional[Dict[str, object]] = None
              ) -> Dict[str, object]:
        feats = self._extract_slow_feat(inputs, depth_prior=depth_prior)
        decoder_depth = None if self.rgbd_early_fusion else depth_prior
        slow_out = self.slow_head(
            feats,
            depth_prior=decoder_depth,
            mask_pilot_targets=mask_pilot_targets,
            video_memory_state=video_memory_state)
        raw_logits = slow_out["logits4"]
        raw_logits5 = slow_out.get("logits5")
        query_logits5 = raw_logits5
        coarse_logits = slow_out["coarse_logits"]
        contact_guidance_logits = slow_out.get("contact_guidance_logits")
        coarse_prior_prob = slow_out.get("coarse_prior_prob")
        coarse_uncertainty = slow_out.get("coarse_uncertainty")
        coarse_reliability = slow_out.get("coarse_reliability")
        coarse_reliability_logits = slow_out.get("coarse_reliability_logits")
        slot_tokens = slow_out["slot_tokens"]
        relation_feat = slow_out["relation_feat"]
        all_logits4 = slow_out.get("all_logits4")
        all_logits5 = slow_out.get("all_logits5")
        all_query_logits5 = all_logits5
        all_class_logits = slow_out.get("all_class_logits")
        mp_raw_logits4 = slow_out.get("mp_logits4")
        mp_raw_logits5 = slow_out.get("mp_logits5")
        mp_all_logits4 = slow_out.get("mp_all_logits4")
        mp_all_logits5 = slow_out.get("mp_all_logits5")
        mp_all_class_logits = slow_out.get("mp_all_class_logits")
        raw_logits = resize(
            raw_logits,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
            warning=False)
        coarse_logits = resize(
            coarse_logits,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
            warning=False)
        if contact_guidance_logits is not None:
            contact_guidance_logits = resize(
                contact_guidance_logits,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False)
        if coarse_prior_prob is not None:
            coarse_prior_prob = resize(
                coarse_prior_prob,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False).clamp(0.0, 1.0)
        if coarse_uncertainty is not None:
            coarse_uncertainty = resize(
                coarse_uncertainty,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False).clamp(0.0, 1.0)
        if coarse_reliability is not None:
            coarse_reliability = resize(
                coarse_reliability,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False).clamp(0.0, 1.0)
        if coarse_reliability_logits is not None:
            coarse_reliability_logits = resize(
                coarse_reliability_logits,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False)
        if all_logits4 is not None:
            all_logits4 = [
                resize(
                    logits,
                    size=inputs.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False)
                for logits in all_logits4
            ]
        if raw_logits5 is not None:
            raw_logits5 = resize(
                raw_logits5,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False)
            raw_logits5 = self._apply_fixed5_logit_calibration(raw_logits5)
        if all_logits5 is not None:
            all_logits5 = [
                self._apply_fixed5_logit_calibration(
                    resize(
                        logits,
                        size=inputs.shape[-2:],
                        mode="bilinear",
                        align_corners=self.align_corners,
                        warning=False))
                for logits in all_logits5
            ]
        if mp_raw_logits4 is not None:
            mp_raw_logits4 = resize(
                mp_raw_logits4,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False)
        if mp_raw_logits5 is not None:
            mp_raw_logits5 = resize(
                mp_raw_logits5,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False)
        if mp_all_logits4 is not None:
            mp_all_logits4 = [
                resize(
                    logits,
                    size=inputs.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False)
                for logits in mp_all_logits4
            ]
        if mp_all_logits5 is not None:
            mp_all_logits5 = [
                resize(
                    logits,
                    size=inputs.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False)
                for logits in mp_all_logits5
            ]
        semantic_logits5 = raw_logits5
        semantic_all_logits5 = all_logits5
        logits6 = None
        logits_softmax = None
        all_logits_softmax = None
        if self.semantic_output_mode == "softmax6":
            logits_softmax = raw_logits
            all_logits_softmax = all_logits4
        if self.semantic_output_mode == "softmax6":
            logits6 = logits_softmax
            prob6 = logits_softmax.softmax(dim=1)
            logits4 = torch.logit(
                softmax6_prob_to_multilabel4_prob(prob6).clamp(1e-4, 1.0 - 1e-4))
        elif self.semantic_output_mode == "fixed5":
            if raw_logits5 is None:
                raise RuntimeError("fixed5 mode requires slow_head to return logits5")
            prob4 = foreground5_prob_to_multilabel4_prob(raw_logits5.sigmoid())
            logits4 = torch.logit(prob4.clamp(1e-4, 1.0 - 1e-4))
        elif self.semantic_output_mode in ("hungarian5", "hungarian5_sam_distractor"):
            if raw_logits5 is None or slow_out.get("class_logits") is None:
                raise RuntimeError(
                    f"{self.semantic_output_mode} mode requires slow_head to return "
                    "logits5 and class_logits")
            semantic_logits5 = self._query_logits_to_semantic5(
                raw_logits5, slow_out["class_logits"])
            if all_query_logits5 is not None and all_class_logits is not None:
                semantic_all_logits5 = [
                    self._query_logits_to_semantic5(layer_logits, layer_class)
                    for layer_logits, layer_class in zip(all_query_logits5, all_class_logits)
                ]
            prob4 = foreground5_prob_to_multilabel4_prob(semantic_logits5.sigmoid())
            logits4 = torch.logit(prob4.clamp(1e-4, 1.0 - 1e-4))
        else:
            logits4 = raw_logits
        state = make_state_from_logits4(
            logits4,
            relation_feat,
            coarse_logits=coarse_logits,
            route="slow_refresh",
            age=inputs.new_zeros(inputs.shape[0]))
        state.slot_tokens = slot_tokens
        if slow_out.get("dynamic_prototypes") is not None:
            state.slot_prototypes = slow_out["dynamic_prototypes"]
        else:
            state.slot_prototypes = slot_tokens
        result = dict(
            logits4=logits4,
            coarse_logits=coarse_logits,
            contact_guidance_logits=contact_guidance_logits,
            state=state,
            slot_tokens=slot_tokens,
            relation_feat=relation_feat,
            pixel_feat=slow_out.get("pixel_feat"),
            dynamic_prototypes=slow_out.get("dynamic_prototypes"),
            prototype_attn=slow_out.get("prototype_attn"),
            relation_dynamic=slow_out.get("relation_dynamic"),
            object_support_logits2=slow_out.get("object_support_logits2"),
            coarse_prior_prob=coarse_prior_prob,
            coarse_uncertainty=coarse_uncertainty,
            coarse_reliability=coarse_reliability,
            coarse_reliability_logits=coarse_reliability_logits,
            hand_condition_low=slow_out.get("hand_condition_low"),
            object_relation_feat=slow_out.get("object_relation_feat"))
        if all_logits4 is not None:
            result["all_logits4"] = all_logits4
        if semantic_logits5 is not None:
            result["logits5"] = semantic_logits5
        if semantic_all_logits5 is not None:
            result["all_logits5"] = semantic_all_logits5
        if query_logits5 is not None:
            result["query_logits5"] = query_logits5
        if all_query_logits5 is not None:
            result["all_query_logits5"] = all_query_logits5
        if logits_softmax is not None:
            result["logits_softmax"] = logits_softmax
        if all_logits_softmax is not None:
            result["all_logits_softmax"] = all_logits_softmax
        if logits6 is not None:
            result["logits6"] = logits6
        if slow_out.get("class_logits") is not None:
            result["class_logits"] = slow_out["class_logits"]
        if all_class_logits is not None:
            result["all_class_logits"] = all_class_logits
        if mp_raw_logits4 is not None:
            result["mp_logits4"] = mp_raw_logits4
        if mp_raw_logits5 is not None:
            result["mp_logits5"] = mp_raw_logits5
        if mp_all_logits4 is not None:
            result["mp_all_logits4"] = mp_all_logits4
        if mp_all_logits5 is not None:
            result["mp_all_logits5"] = mp_all_logits5
        if slow_out.get("mp_class_logits") is not None:
            result["mp_class_logits"] = slow_out["mp_class_logits"]
        if mp_all_class_logits is not None:
            result["mp_all_class_logits"] = mp_all_class_logits
        if slow_out.get("memory_aux") is not None:
            result["memory_aux"] = slow_out["memory_aux"]
        return result

    @torch.no_grad()
    def forward_video_memory_step(
            self,
            inputs: Tensor,
            video_memory_state: Optional[Dict[str, object]] = None,
            init_masks5: Optional[Tensor] = None,
            depth_prior: Optional[Tensor] = None,
            profile: bool = True) -> Dict[str, object]:
        """Run one true video-memory step and update memory from GT/pred masks.

        ``init_masks5`` is used for first-frame GT initialization.  Later calls
        omit it and write predicted fixed5 probabilities, matching the training
        read/write closure used by ``loss_video_memory``.
        """

        if self.semantic_output_mode != "fixed5":
            raise ValueError(
                "forward_video_memory_step currently requires fixed5 output")
        total_start = time.perf_counter()
        out = self._slow(
            inputs,
            depth_prior=depth_prior,
            video_memory_state=video_memory_state)
        write_masks = init_masks5
        if write_masks is None:
            write_masks = out["logits5"].sigmoid()
        if hasattr(self.slow_head, "update_video_memory"):
            video_memory_state = self.slow_head.update_video_memory(
                video_memory_state,
                out["relation_feat"],
                write_masks,
                out["coarse_logits"].sigmoid())
            if self.video_memory_detach_state:
                video_memory_state = self._detach_video_memory_state(
                    video_memory_state)
        out["video_memory_state"] = video_memory_state
        if profile:
            _cuda_sync_if_needed(out["logits4"])
            profile_data = dict(
                route="video_memory",
                video_memory_type=getattr(
                    self.slow_head, "video_memory_type", "none"),
                total_ms=_elapsed_ms(total_start),
                slow_path_ms=_elapsed_ms(total_start))
            out["profile"] = profile_data
        else:
            out["profile"] = dict(route="video_memory")
        out["route"] = "video_memory"
        return out

    def _fast(self, current_feat: Tensor,
              prev_state: LiteHOSRelationState) -> Dict[str, object]:
        logits4, warped_mask4, warped_coarse, theta = self.fast_head(
            current_feat, prev_state)
        state = make_state_from_logits4(
            logits4,
            current_feat,
            coarse_logits=torch.logit(warped_coarse.clamp(1e-4, 1.0 - 1e-4)),
            route="fast_propagate_refine",
            age=prev_state.age.to(device=logits4.device, dtype=logits4.dtype) + 1.0)
        if prev_state.slot_tokens.shape[-1] == self.fast_head.state_channels:
            state.slot_tokens = prev_state.slot_tokens.to(
                device=logits4.device, dtype=logits4.dtype)
            state.slot_prototypes = prev_state.slot_prototypes.to(
                device=logits4.device, dtype=logits4.dtype)
        return dict(
            logits4=logits4,
            state=state,
            warped_mask4=warped_mask4,
            warped_coarse=warped_coarse,
            theta=theta)

    def _select_route(self,
                      route: str,
                      gate_logits: Tensor,
                      prev_state: Optional[LiteHOSRelationState]) -> str:
        route = route.lower()
        aliases = {
            "slow": "slow_refresh",
            "slow_refresh": "slow_refresh",
            "fast": "fast_propagate_refine",
            "fast_refine": "fast_propagate_refine",
            "fast_propagate_refine": "fast_propagate_refine",
            "no_hand": "no_hand",
        }
        if route in aliases:
            selected = aliases[route]
        elif route == "auto":
            selected = ROUTE_NAMES[int(gate_logits.argmax(dim=1)[0])]
        else:
            raise ValueError(f"Unsupported relation route: {route}")
        if selected == "fast_propagate_refine" and prev_state is None:
            selected = "slow_refresh"
        return selected

    def forward_relation(self,
                         inputs: Tensor,
                         prev_state: Optional[LiteHOSRelationState] = None,
                         route: str = "auto",
                         depth_prior: Optional[Tensor] = None,
                         mask_pilot_targets: Optional[Tensor] = None,
                         profile: bool = True) -> Dict[str, object]:
        """Run one streaming step and return logits, state, route, and profile."""

        profile_data: Dict[str, object] = {}
        total_start = time.perf_counter()
        route_lower = route.lower()

        if route_lower in ("pure_slow", "single_model", "single_slow"):
            start = time.perf_counter()
            result = self._slow(
                inputs,
                depth_prior=depth_prior,
                mask_pilot_targets=mask_pilot_targets)
            _cuda_sync_if_needed(result["logits4"])
            if profile:
                profile_data["slow_path_ms"] = _elapsed_ms(start)
                profile_data["total_ms"] = _elapsed_ms(total_start)
            profile_data["route"] = "pure_slow"
            result.update(route="pure_slow", profile=profile_data)
            return result

        start = time.perf_counter()
        fast_feat = self.fast_encoder(inputs)
        _cuda_sync_if_needed(fast_feat)
        if profile:
            profile_data["fast_encoder_ms"] = _elapsed_ms(start)

        start = time.perf_counter()
        gate_logits = self.gate_head(fast_feat, prev_state)
        _cuda_sync_if_needed(gate_logits)
        if profile:
            profile_data["gate_ms"] = _elapsed_ms(start)
        selected = self._select_route(route, gate_logits, prev_state)

        if selected == "no_hand":
            logits4 = inputs.new_full((inputs.shape[0], 4, *inputs.shape[-2:]), -12.0)
            state = make_state_from_logits4(
                logits4,
                fast_feat,
                route="no_hand",
                age=(inputs.new_zeros(inputs.shape[0]) if prev_state is None else prev_state.age + 1.0))
            result = dict(logits4=logits4, state=state)
            if self.semantic_output_mode == "fixed5":
                result["logits5"] = inputs.new_full(
                    (inputs.shape[0], 5, *inputs.shape[-2:]), -12.0)
        elif selected == "slow_refresh":
            start = time.perf_counter()
            result = self._slow(
                inputs,
                depth_prior=depth_prior,
                mask_pilot_targets=mask_pilot_targets)
            _cuda_sync_if_needed(result["logits4"])
            if profile:
                profile_data["slow_path_ms"] = _elapsed_ms(start)
        else:
            start = time.perf_counter()
            result = self._fast(fast_feat, prev_state)
            _cuda_sync_if_needed(result["logits4"])
            if profile:
                profile_data["fast_path_ms"] = _elapsed_ms(start)

        profile_data["route"] = selected
        if profile:
            profile_data["total_ms"] = _elapsed_ms(total_start)
        result.update(route=selected, gate_logits=gate_logits, profile=profile_data)
        return result

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: Optional[List[dict]] = None) -> Tensor:
        out = self.forward_relation(inputs, route="slow")
        return out.get("logits5", out.get("logits6", out.get("logits_softmax", out["logits4"])))

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        return self.encode_decode(inputs)

    def _target4_from_samples(self, data_samples: SampleList, device) -> Tensor:
        targets = []
        for sample in data_samples:
            if hasattr(sample, "gt_sem_seg_4"):
                target = sample.gt_sem_seg_4.data
                if target.dim() == 3 and target.shape[0] == 4:
                    targets.append(target.float())
                    continue
            label = sample.gt_sem_seg.data
            targets.append(labels6_to_multilabel4(label.unsqueeze(0))[0])
        return torch.stack(targets, dim=0).to(device=device)

    def _target6_from_samples(self, data_samples: SampleList, device) -> Tensor:
        targets = []
        for sample in data_samples:
            label = sample.gt_sem_seg.data
            targets.append(labels6_to_softmax6(label.unsqueeze(0))[0])
        return torch.stack(targets, dim=0).to(device=device)

    def _target5_from_samples(self, data_samples: SampleList, device) -> Tensor:
        targets = []
        for sample in data_samples:
            label = sample.gt_sem_seg.data
            targets.append(labels6_to_foreground5(label.unsqueeze(0))[0])
        return torch.stack(targets, dim=0).to(device=device)

    def _hungarian5d_targets_from_samples(
            self,
            data_samples: SampleList,
            device) -> List[tuple[Tensor, Tensor]]:
        """Build per-image targets for 5 foreground classes + distractor.

        Class order is LH, RH, LO, RO, TO, distractor, no-object. The first
        two hand targets are fixed semantic masks. Object targets have a
        strict budget of three: original LO/RO/TO masks are kept first, then
        SAM-generated near-hand distractors fill remaining slots.
        """
        target5 = self._target5_from_samples(data_samples, device)
        result = []
        eps = float(self.absent_slot_area_eps)
        for sample_idx, sample in enumerate(data_samples):
            masks = []
            labels = []
            sample_target = target5[sample_idx]
            present = sample_target.flatten(1).mean(-1) > eps
            for class_idx in (0, 1):
                if bool(present[class_idx]):
                    masks.append(sample_target[class_idx])
                    labels.append(class_idx)
            object_count = 0
            for class_idx in (2, 3, 4):
                if bool(present[class_idx]) and object_count < 3:
                    masks.append(sample_target[class_idx])
                    labels.append(class_idx)
                    object_count += 1
            if object_count < 3 and hasattr(sample, "gt_sam_distractor_masks"):
                distractor = sample.gt_sam_distractor_masks.data.to(
                    device=device, dtype=sample_target.dtype)
                if distractor.dim() == 2:
                    distractor = distractor.unsqueeze(0)
                if distractor.dim() != 3:
                    raise ValueError(
                        "Expected gt_sam_distractor_masks [N,H,W], got "
                        f"{tuple(distractor.shape)}")
                if distractor.shape[-2:] != sample_target.shape[-2:]:
                    distractor = resize(
                        distractor.unsqueeze(0),
                        size=sample_target.shape[-2:],
                        mode="nearest",
                        align_corners=None,
                        warning=False)[0]
                for mask in distractor:
                    if object_count >= 3:
                        break
                    if float(mask.mean()) <= eps:
                        continue
                    masks.append(mask.clamp(0, 1))
                    labels.append(5)
                    object_count += 1
            if masks:
                result.append((
                    torch.stack(masks, dim=0).to(device=device),
                    torch.tensor(labels, dtype=torch.long, device=device)))
            else:
                empty = sample_target.new_zeros((0, *sample_target.shape[-2:]))
                result.append((
                    empty,
                    torch.empty((0, ), dtype=torch.long, device=device)))
        return result

    def _target5_video_from_samples(self, data_samples: SampleList,
                                    device) -> Tensor:
        targets = []
        for sample in data_samples:
            label = sample.gt_sem_seg.data
            if label.dim() == 4 and label.shape[0] == 1:
                label = label[0]
            if label.dim() != 3:
                raise ValueError(
                    "Video memory training expects each gt_sem_seg to be "
                    f"[T,H,W], got {tuple(label.shape)}")
            flat = label.unsqueeze(1)
            target = labels6_to_foreground5(flat.reshape(-1, *flat.shape[-2:]))
            targets.append(target.reshape(label.shape[0], 5, *label.shape[-2:]))
        return torch.stack(targets, dim=0).to(device=device)

    @staticmethod
    def _query_logits_to_semantic5(query_logits: Tensor,
                                   class_logits: Tensor) -> Tensor:
        class_prob = class_logits.softmax(dim=-1)[..., :5]
        mask_prob = query_logits.sigmoid()
        prob5 = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
        return torch.logit(prob5.clamp(1e-4, 1.0 - 1e-4))

    def _target4_softmax_from_samples(self, data_samples: SampleList, device) -> Tensor:
        targets = []
        for sample in data_samples:
            label = sample.gt_sem_seg.data
            if label.dim() == 3 and label.shape[0] == 1:
                label = label[0]
            target = label.new_full(label.shape[-2:], 255, dtype=torch.long)
            target[label == 1] = 0
            target[label == 2] = 1
            target[label == 3] = 2
            target[label == 4] = 3
            targets.append(target)
        return torch.stack(targets, dim=0).to(device=device)

    def _target_softmax_from_samples(self, data_samples: SampleList, device) -> Tensor:
        if self.semantic_output_mode == "softmax6":
            return self._target6_from_samples(data_samples, device)
        raise ValueError(
            "_target_softmax_from_samples requires a softmax semantic_output_mode")

    def _depth_prior_from_samples(self,
                                  data_samples: Optional[SampleList],
                                  size: Sequence[int],
                                  device,
                                  dtype) -> Optional[Tensor]:
        if not data_samples:
            return None
        priors = []
        for sample in data_samples:
            if not hasattr(sample, "gt_depth_map"):
                return None
            depth = sample.gt_depth_map.data
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            if depth.dim() != 3:
                return None
            depth = depth[:1].to(device=device, dtype=dtype).unsqueeze(0)
            if tuple(depth.shape[-2:]) != tuple(size):
                depth = resize(
                    depth,
                    size=size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False)
            priors.append(depth.squeeze(0).clamp(0.0, 1.0))
        return torch.stack(priors, dim=0)

    def _contact_target_from_samples(self,
                                     data_samples: Optional[SampleList],
                                     size: Sequence[int],
                                     device,
                                     dtype) -> Optional[Tensor]:
        if not data_samples:
            return None
        targets = []
        for sample in data_samples:
            if not hasattr(sample, "gt_contact_map"):
                return None
            contact = sample.gt_contact_map.data
            if contact.dim() == 2:
                contact = contact.unsqueeze(0)
            if contact.dim() != 3:
                return None
            contact = contact[:1].to(device=device, dtype=dtype).unsqueeze(0)
            if tuple(contact.shape[-2:]) != tuple(size):
                contact = resize(
                    contact,
                    size=size,
                    mode="nearest",
                    align_corners=None,
                    warning=False)
            targets.append(contact.squeeze(0).clamp(0.0, 1.0))
        return torch.stack(targets, dim=0)

    @staticmethod
    def _normalized_slot_weights(slot_weights: Optional[Tensor],
                                 num_slots: int,
                                 device: torch.device,
                                 dtype: torch.dtype) -> Optional[Tensor]:
        if slot_weights is None:
            return None
        weights = torch.as_tensor(slot_weights, device=device, dtype=dtype)
        if weights.dim() != 1 or weights.numel() != int(num_slots):
            raise ValueError(
                "slot_weights must be a 1D tensor with one value per slot, "
                f"got shape {tuple(weights.shape)} for {num_slots} slots")
        return weights / weights.mean().clamp_min(1e-6)

    @staticmethod
    def _reduce_slot_losses(loss: Tensor,
                            reduction: str,
                            slot_weights: Optional[Tensor] = None) -> Tensor:
        weights = LiteHOSV2RelationStateSegmentor._normalized_slot_weights(
            slot_weights, loss.numel(), loss.device, loss.dtype)
        if weights is not None:
            loss = loss * weights
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unsupported slot loss reduction: {reduction!r}")

    @staticmethod
    def _dice_loss(logits: Tensor,
                   target: Tensor,
                   eps: float = 1.0,
                   reduction: str = "mean",
                   slot_weights: Optional[Tensor] = None) -> Tensor:
        prob = logits.sigmoid()
        dims = (0, 2, 3)
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        loss = 1.0 - (2.0 * inter + eps) / (denom + eps)
        return LiteHOSV2RelationStateSegmentor._reduce_slot_losses(
            loss, reduction, slot_weights=slot_weights)

    @staticmethod
    def _slot_bce_loss(logits: Tensor,
                       target: Tensor,
                       reduction: str = "mean",
                       slot_weights: Optional[Tensor] = None) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none").mean(dim=(0, 2, 3))
        return LiteHOSV2RelationStateSegmentor._reduce_slot_losses(
            loss, reduction, slot_weights=slot_weights)

    @staticmethod
    def _point_sampled_slot_losses(
            logits: Tensor,
            target: Tensor,
            num_points: int,
            oversample_ratio: float,
            importance_sample_ratio: float,
            reduction: str = "mean",
            slot_weights: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if num_points <= 0:
            zero = logits.sum() * 0.0
            return zero, zero
        if target.shape[-2:] != logits.shape[-2:]:
            target = resize(
                target.float(),
                size=logits.shape[-2:],
                mode="nearest",
                align_corners=None,
                warning=False)
        batch_size, num_slots = logits.shape[:2]
        flat_logits = logits.reshape(
            batch_size * num_slots, 1, *logits.shape[-2:])
        flat_target = target.reshape(
            batch_size * num_slots, 1, *target.shape[-2:])
        labels = torch.zeros(
            batch_size * num_slots,
            dtype=torch.long,
            device=logits.device)
        coords = get_uncertain_point_coords_with_randomness(
            flat_logits,
            labels,
            num_points=int(num_points),
            oversample_ratio=float(oversample_ratio),
            importance_sample_ratio=float(importance_sample_ratio))
        point_logits = point_sample(flat_logits, coords).squeeze(1)
        point_target = point_sample(flat_target, coords).squeeze(1)
        bce = F.binary_cross_entropy_with_logits(
            point_logits, point_target, reduction="none")
        bce = bce.view(batch_size, num_slots, -1).mean(dim=(0, 2))
        point_logits = point_logits.view(batch_size, num_slots, -1)
        point_target = point_target.view(batch_size, num_slots, -1)
        point_prob = point_logits.sigmoid()
        inter = (point_prob * point_target).sum(dim=(0, 2))
        denom = point_prob.sum(dim=(0, 2)) + point_target.sum(dim=(0, 2))
        dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        return (
            LiteHOSV2RelationStateSegmentor._reduce_slot_losses(
                bce, reduction, slot_weights=slot_weights),
            LiteHOSV2RelationStateSegmentor._reduce_slot_losses(
                dice, reduction, slot_weights=slot_weights))

    @staticmethod
    def _tversky_loss(logits: Tensor,
                      target: Tensor,
                      alpha: float = 0.3,
                      beta: float = 0.7,
                      eps: float = 1.0,
                      reduction: str = "mean",
                      slot_weights: Optional[Tensor] = None) -> Tensor:
        prob = logits.sigmoid()
        dims = (0, 2, 3)
        tp = (prob * target).sum(dims)
        fp = (prob * (1.0 - target)).sum(dims)
        fn = ((1.0 - prob) * target).sum(dims)
        denom = tp + float(alpha) * fp + float(beta) * fn
        loss = 1.0 - (tp + float(eps)) / (denom + float(eps))
        return LiteHOSV2RelationStateSegmentor._reduce_slot_losses(
            loss, reduction, slot_weights=slot_weights)

    @staticmethod
    def _fixed_query_class_loss(class_logits: Tensor,
                                target_masks: Tensor,
                                absent_area_eps: float = 1e-6,
                                no_object_weight: float = 0.1) -> Tensor:
        if class_logits.dim() != 3:
            raise ValueError(
                f"Expected class logits [B,Q,C], got {tuple(class_logits.shape)}")
        batch_size, num_queries, num_classes = class_logits.shape
        if target_masks.dim() != 4:
            raise ValueError(
                f"Expected target masks [B,Q,H,W], got {tuple(target_masks.shape)}")
        if target_masks.shape[:2] != (batch_size, num_queries):
            raise ValueError(
                "Class logits and target masks must share B,Q dimensions, got "
                f"{tuple(class_logits.shape)} and {tuple(target_masks.shape)}")
        if num_classes != num_queries + 1:
            raise ValueError(
                "Fixed-query class loss requires N foreground classes plus "
                f"one no-object class, got Q={num_queries}, C={num_classes}")
        foreground_target = torch.arange(
            num_queries, device=class_logits.device, dtype=torch.long)
        target = foreground_target.unsqueeze(0).expand(batch_size, -1).clone()
        area = target_masks.flatten(2).mean(-1)
        target = target.masked_fill(area <= float(absent_area_eps), num_queries)
        class_weight = class_logits.new_ones(num_classes)
        class_weight[-1] = float(no_object_weight)
        return F.cross_entropy(
            class_logits.reshape(-1, num_classes),
            target.reshape(-1),
            weight=class_weight)

    def _hungarian_query_losses(self,
                                mask_logits: Tensor,
                                class_logits: Tensor,
                                target5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        from scipy.optimize import linear_sum_assignment

        batch_size, num_queries = mask_logits.shape[:2]
        num_classes = class_logits.shape[-1]
        if num_classes != 6:
            raise ValueError(
                f"hungarian5 expects class_out_channels=6, got {num_classes}")
        if target5.shape[-2:] != mask_logits.shape[-2:]:
            target5 = resize(
                target5.float(),
                size=mask_logits.shape[-2:],
                mode="nearest",
                align_corners=None,
                warning=False)
        cls_targets = torch.full(
            (batch_size, num_queries),
            5,
            dtype=torch.long,
            device=class_logits.device)
        bce_terms = []
        dice_terms = []
        class_prob = class_logits.softmax(dim=-1)
        for batch_idx in range(batch_size):
            target = target5[batch_idx]
            present = target.flatten(1).mean(-1) > self.absent_slot_area_eps
            gt_classes = torch.nonzero(present, as_tuple=False).flatten()
            if gt_classes.numel() == 0:
                continue
            gt_masks = target[gt_classes]
            q_logits = mask_logits[batch_idx]
            q_prob = q_logits.sigmoid()
            cost_class = -class_prob[batch_idx][:, gt_classes]
            q_flat = q_logits.flatten(1)
            gt_flat = gt_masks.flatten(1)
            cost_bce = F.binary_cross_entropy_with_logits(
                q_flat[:, None, :].expand(-1, gt_flat.shape[0], -1),
                gt_flat[None, :, :].expand(num_queries, -1, -1),
                reduction="none").mean(-1)
            prob_flat = q_prob.flatten(1)
            inter = torch.einsum("qn,gn->qg", prob_flat, gt_flat)
            denom = prob_flat.sum(-1)[:, None] + gt_flat.sum(-1)[None, :]
            cost_dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
            cost = (
                2.0 * cost_class +
                5.0 * cost_bce +
                5.0 * cost_dice).detach().cpu()
            row_ind, col_ind = linear_sum_assignment(cost.numpy())
            rows = torch.as_tensor(row_ind, device=mask_logits.device, dtype=torch.long)
            cols = torch.as_tensor(col_ind, device=mask_logits.device, dtype=torch.long)
            matched_classes = gt_classes[cols]
            cls_targets[batch_idx, rows] = matched_classes
            matched_logits = q_logits[rows]
            matched_targets = gt_masks[cols].to(dtype=matched_logits.dtype)
            bce_terms.append(
                F.binary_cross_entropy_with_logits(
                    matched_logits, matched_targets, reduction="mean"))
            dice_terms.append(
                self._dice_loss(
                    matched_logits.unsqueeze(0),
                    matched_targets.unsqueeze(0),
                    reduction="mean"))
        class_weight = class_logits.new_ones(num_classes)
        class_weight[-1] = self.loss_cls_no_object_weight
        loss_cls = F.cross_entropy(
            class_logits.reshape(-1, num_classes),
            cls_targets.reshape(-1),
            weight=class_weight)
        zero = mask_logits.sum() * 0.0
        loss_bce = torch.stack(bce_terms).mean() if bce_terms else zero
        loss_dice = torch.stack(dice_terms).mean() if dice_terms else zero
        return loss_bce, loss_dice, loss_cls

    def _hungarian_query_losses_from_targets(
            self,
            mask_logits: Tensor,
            class_logits: Tensor,
            batch_targets: List[tuple[Tensor, Tensor]],
            no_object_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        from scipy.optimize import linear_sum_assignment

        batch_size, num_queries = mask_logits.shape[:2]
        num_classes = class_logits.shape[-1]
        if num_classes <= int(no_object_idx):
            raise ValueError(
                f"Expected class_out_channels>{no_object_idx}, got {num_classes}")
        cls_targets = torch.full(
            (batch_size, num_queries),
            int(no_object_idx),
            dtype=torch.long,
            device=class_logits.device)
        bce_terms = []
        dice_terms = []
        class_prob = class_logits.softmax(dim=-1)
        for batch_idx, (gt_masks, gt_classes) in enumerate(batch_targets):
            if gt_masks.numel() == 0 or gt_classes.numel() == 0:
                continue
            gt_masks = gt_masks.to(device=mask_logits.device, dtype=mask_logits.dtype)
            gt_classes = gt_classes.to(device=mask_logits.device, dtype=torch.long)
            if gt_masks.shape[-2:] != mask_logits.shape[-2:]:
                gt_masks = resize(
                    gt_masks.unsqueeze(0),
                    size=mask_logits.shape[-2:],
                    mode="nearest",
                    align_corners=None,
                    warning=False)[0]
            q_logits = mask_logits[batch_idx]
            q_prob = q_logits.sigmoid()
            cost_class = -class_prob[batch_idx][:, gt_classes]
            q_flat = q_logits.flatten(1)
            gt_flat = gt_masks.flatten(1)
            cost_bce = F.binary_cross_entropy_with_logits(
                q_flat[:, None, :].expand(-1, gt_flat.shape[0], -1),
                gt_flat[None, :, :].expand(num_queries, -1, -1),
                reduction="none").mean(-1)
            prob_flat = q_prob.flatten(1)
            inter = torch.einsum("qn,gn->qg", prob_flat, gt_flat)
            denom = prob_flat.sum(-1)[:, None] + gt_flat.sum(-1)[None, :]
            cost_dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
            cost = (
                2.0 * cost_class +
                5.0 * cost_bce +
                5.0 * cost_dice).detach().cpu()
            row_ind, col_ind = linear_sum_assignment(cost.numpy())
            rows = torch.as_tensor(row_ind, device=mask_logits.device, dtype=torch.long)
            cols = torch.as_tensor(col_ind, device=mask_logits.device, dtype=torch.long)
            cls_targets[batch_idx, rows] = gt_classes[cols]
            matched_logits = q_logits[rows]
            matched_targets = gt_masks[cols].to(dtype=matched_logits.dtype)
            bce_terms.append(
                F.binary_cross_entropy_with_logits(
                    matched_logits, matched_targets, reduction="mean"))
            dice_terms.append(
                self._dice_loss(
                    matched_logits.unsqueeze(0),
                    matched_targets.unsqueeze(0),
                    reduction="mean"))
        class_weight = class_logits.new_ones(num_classes)
        class_weight[int(no_object_idx)] = self.loss_cls_no_object_weight
        loss_cls = F.cross_entropy(
            class_logits.reshape(-1, num_classes),
            cls_targets.reshape(-1),
            weight=class_weight)
        zero = mask_logits.sum() * 0.0
        loss_bce = torch.stack(bce_terms).mean() if bce_terms else zero
        loss_dice = torch.stack(dice_terms).mean() if dice_terms else zero
        return loss_bce, loss_dice, loss_cls

    def _decoder_cooccurrence_loss(self, logits4: Tensor) -> Tensor:
        if logits4.shape[1] < 4:
            return logits4.sum() * 0.0
        prob = logits4.sigmoid()
        area = prob.flatten(2).mean(-1)
        tau = float(self.cooccurrence_area_tau)
        temperature = max(float(self.cooccurrence_temperature), 1e-6)
        left_absent = torch.sigmoid((tau - area[:, 0]) / temperature)
        right_absent = torch.sigmoid((tau - area[:, 1]) / temperature)
        left_object_area = area[:, 2]
        right_object_area = area[:, 3]
        loss = left_absent * left_object_area + right_absent * right_object_area
        if logits4.shape[1] >= 5:
            two_hand_absent = torch.maximum(left_absent, right_absent)
            loss = loss + two_hand_absent * area[:, 4]
        return loss.mean()

    def _relation_consistency_loss(self, logits5: Tensor) -> Tensor:
        if logits5.shape[1] < 5:
            return logits5.sum() * 0.0
        prob = logits5.sigmoid()
        area = prob.flatten(2).mean(-1)
        tau = float(self.cooccurrence_area_tau)
        temperature = max(float(self.cooccurrence_temperature), 1e-6)
        left_absent = torch.sigmoid((tau - area[:, 0]) / temperature)
        right_absent = torch.sigmoid((tau - area[:, 1]) / temperature)
        left_object_area = area[:, 2]
        right_object_area = area[:, 3]
        two_hand_object_area = area[:, 4]
        hand_overlap = (prob[:, 0] * prob[:, 1]).flatten(1).mean(-1)
        object_overlap = (
            prob[:, 2] * prob[:, 3] +
            prob[:, 2] * prob[:, 4] +
            prob[:, 3] * prob[:, 4]).flatten(1).mean(-1) / 3.0
        loss = (
            left_absent * left_object_area +
            right_absent * right_object_area +
            torch.maximum(left_absent, right_absent) * two_hand_object_area +
            hand_overlap +
            object_overlap)
        return loss.mean()

    @staticmethod
    def _fixed5_to_soft_compat6_prob(logits5: Tensor) -> Tensor:
        if logits5.dim() != 4 or logits5.shape[1] != 5:
            raise ValueError(f"Expected fixed5 logits [B,5,H,W], got {tuple(logits5.shape)}")
        prob5 = logits5.sigmoid()
        bg = (1.0 - prob5.max(dim=1, keepdim=True).values).clamp(0.0, 1.0)
        prob6 = torch.cat([bg, prob5], dim=1)
        return prob6 / prob6.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _soft_prob6_dice_loss(prob6: Tensor,
                              target6: Tensor,
                              ignore_index: int = 255,
                              eps: float = 1.0) -> Tensor:
        valid = target6 != int(ignore_index)
        if not bool(valid.any()):
            return prob6.sum() * 0.0
        safe_target = target6.clamp(0, prob6.shape[1] - 1)
        one_hot = F.one_hot(safe_target, num_classes=prob6.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2).to(
            device=prob6.device, dtype=prob6.dtype)
        valid_f = valid.unsqueeze(1).to(dtype=prob6.dtype)
        prob6 = prob6 * valid_f
        one_hot = one_hot * valid_f
        dims = (0, 2, 3)
        inter = (prob6 * one_hot).sum(dims)
        denom = prob6.sum(dims) + one_hot.sum(dims)
        return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()

    def _compat6_soft_losses(self,
                             logits5: Tensor,
                             target6: Tensor) -> tuple[Tensor, Tensor]:
        prob6 = self._fixed5_to_soft_compat6_prob(logits5)
        if prob6.shape[-2:] != target6.shape[-2:]:
            prob6 = resize(
                prob6,
                size=target6.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False).clamp_min(1e-6)
            prob6 = prob6 / prob6.sum(dim=1, keepdim=True).clamp_min(1e-6)
        valid = target6 != 255
        if bool(valid.any()):
            loss_ce = F.nll_loss(
                prob6.clamp_min(1e-6).log(),
                target6,
                ignore_index=255)
        else:
            loss_ce = logits5.sum() * 0.0
        loss_dice = self._soft_prob6_dice_loss(prob6, target6)
        return loss_ce, loss_dice

    @staticmethod
    def _two_hand_object_consistency_loss(logits5: Tensor,
                                          target5: Tensor,
                                          eps: float = 1e-6) -> Tensor:
        if logits5.shape[1] < 5 or target5.shape[1] < 5:
            return logits5.sum() * 0.0
        prob = logits5.sigmoid()
        lo_ro_target = (target5[:, 2:4].sum(dim=1, keepdim=True) > 0.5).to(prob.dtype)
        to_target = target5[:, 4:5].to(dtype=prob.dtype)
        conflict = prob[:, 4:5] * lo_ro_target + prob[:, 2:4].mean(dim=1, keepdim=True) * to_target
        support = (lo_ro_target + to_target).clamp(0.0, 1.0)
        denom = support.sum().clamp_min(float(eps))
        return (conflict * support).sum() / denom

    @staticmethod
    def _left_right_exclusivity_loss(logits5: Tensor) -> Tensor:
        if logits5.shape[1] < 5:
            return logits5.sum() * 0.0
        prob = logits5.sigmoid()
        hand_overlap = (prob[:, 0] * prob[:, 1]).flatten(1).mean(-1)
        object_overlap = (
            prob[:, 2] * prob[:, 3] +
            prob[:, 2] * prob[:, 4] +
            prob[:, 3] * prob[:, 4]).flatten(1).mean(-1) / 3.0
        return (hand_overlap + object_overlap).mean()

    @staticmethod
    def _softmax_dice_loss(logits: Tensor,
                           target: Tensor,
                           class_weight: Optional[Tensor] = None,
                           ignore_index: int = 255,
                           eps: float = 1.0) -> Tensor:
        prob = logits.softmax(dim=1)
        valid = target != int(ignore_index)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        safe_target = target.clamp(0, prob.shape[1] - 1)
        one_hot = F.one_hot(safe_target, num_classes=prob.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=prob.dtype, device=prob.device)
        valid = valid.unsqueeze(1).to(dtype=prob.dtype)
        prob = prob * valid
        one_hot = one_hot * valid
        dims = (0, 2, 3)
        inter = (prob * one_hot).sum(dims)
        denom = prob.sum(dims) + one_hot.sum(dims)
        loss = 1.0 - (2.0 * inter + eps) / (denom + eps)
        if class_weight is None:
            return loss.mean()
        class_weight = class_weight.to(device=loss.device, dtype=loss.dtype)
        if class_weight.numel() != loss.numel():
            raise ValueError(
                "class_weight must match softmax dice channels, got "
                f"{class_weight.numel()} weights for {loss.numel()} classes")
        return (loss * class_weight).sum() / class_weight.sum().clamp_min(1e-6)

    @staticmethod
    def _coarse_target_from_target4(target4: Tensor) -> Tensor:
        return target4.max(dim=1, keepdim=True).values

    @staticmethod
    def _odd_kernel(kernel: int) -> int:
        kernel = max(int(kernel), 1)
        return kernel if kernel % 2 == 1 else kernel + 1

    @staticmethod
    def _max_pool_mask(mask: Tensor, kernel: int) -> Tensor:
        kernel = LiteHOSV2RelationStateSegmentor._odd_kernel(kernel)
        return F.max_pool2d(mask, kernel, stride=1, padding=kernel // 2)

    @staticmethod
    def _min_pool_mask(mask: Tensor, kernel: int) -> Tensor:
        kernel = LiteHOSV2RelationStateSegmentor._odd_kernel(kernel)
        return -F.max_pool2d(-mask, kernel, stride=1, padding=kernel // 2)

    @staticmethod
    def _inside_distance_approx(mask: Tensor, max_steps: int = 8) -> Tensor:
        mask = mask.float().clamp(0.0, 1.0)
        current = mask
        distance = mask.new_zeros(mask.shape)
        steps = max(int(max_steps), 1)
        for step in range(steps):
            distance = distance + current * float(step + 1)
            current = LiteHOSV2RelationStateSegmentor._min_pool_mask(current, 3)
            current = (current > 0.5).to(dtype=mask.dtype)
        return (distance / float(steps)).clamp(0.0, 1.0) * mask

    @staticmethod
    def object_support_target_from_target4(
            target4: Tensor,
            boundary_kernel: int = 15,
            contact_kernel: int = 31,
            depth_prior: Optional[Tensor] = None,
            depth_contact_tau: float = 0.08,
            depth_contact_weight: float = 0.7,
            contact2d_weight: float = 0.5,
            return_components: bool = False):
        """Build object-only LO/RO soft support targets from four-slot GT."""

        target4 = target4.float().clamp(0.0, 1.0)
        hands = target4[:, :2]
        objects = target4[:, 2:4]
        boundary_dilate = LiteHOSV2RelationStateSegmentor._max_pool_mask(
            objects, boundary_kernel)
        boundary_erode = LiteHOSV2RelationStateSegmentor._min_pool_mask(
            objects, boundary_kernel)
        boundary_band = (boundary_dilate - boundary_erode).clamp(0.0, 1.0)
        distance_inside = LiteHOSV2RelationStateSegmentor._inside_distance_approx(
            objects, max_steps=max(boundary_kernel // 2, 1))
        hand_neighborhood = LiteHOSV2RelationStateSegmentor._max_pool_mask(
            hands, contact_kernel)
        object_neighborhood = LiteHOSV2RelationStateSegmentor._max_pool_mask(
            objects, contact_kernel)
        contact_neighborhood = (hand_neighborhood * object_neighborhood).clamp(0.0, 1.0)
        depth_contact = None
        if depth_prior is not None:
            depth = depth_prior.to(device=target4.device, dtype=target4.dtype)
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            if depth.shape[1] != 1:
                depth = depth[:, :1]
            if depth.shape[-2:] != target4.shape[-2:]:
                depth = F.interpolate(
                    depth,
                    size=target4.shape[-2:],
                    mode="bilinear",
                    align_corners=False)
            depth = depth.clamp(0.0, 1.0)
            hand_area = hands.flatten(2).sum(-1).clamp_min(1e-6)
            hand_depth = (hands * depth).flatten(2).sum(-1) / hand_area
            hand_depth = hand_depth.view(target4.shape[0], 2, 1, 1)
            depth_consistency = torch.exp(
                -(depth - hand_depth).abs() / max(float(depth_contact_tau), 1e-6))
            hand_exists = (hands.flatten(2).sum(-1) > 1e-6).to(target4.dtype)
            hand_exists = hand_exists.view(target4.shape[0], 2, 1, 1)
            depth_contact = (
                contact_neighborhood * depth_consistency * hand_exists).clamp(0.0, 1.0)
            contact_term = torch.maximum(
                depth_contact * float(depth_contact_weight),
                contact_neighborhood * float(contact2d_weight))
        else:
            contact_term = contact_neighborhood * float(contact2d_weight)
        support = torch.maximum(
            objects,
            torch.maximum(
                boundary_band * 0.7,
                torch.maximum(distance_inside, contact_term)))
        support = support.clamp(0.0, 1.0)
        if not return_components:
            return support
        components = dict(
            object_core=objects,
            boundary_band=boundary_band,
            distance_inside=distance_inside,
            contact_neighborhood=contact_neighborhood)
        if depth_contact is not None:
            components["depth_contact_neighborhood"] = depth_contact
        return support, components

    @staticmethod
    def _masked_average_feature(feat: Tensor,
                                mask4: Tensor,
                                eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        if mask4.shape[-2:] != feat.shape[-2:]:
            mask4 = F.interpolate(mask4.float(), size=feat.shape[-2:], mode="area")
        mask4 = mask4.to(device=feat.device, dtype=feat.dtype)
        area = mask4.flatten(2).sum(-1)
        proto = torch.einsum("bnhw,bchw->bnc", mask4, feat) / area.clamp_min(eps).unsqueeze(-1)
        valid = area > eps
        return proto, valid

    def _prototype_coherence_loss(self,
                                  dynamic_prototypes: Optional[Tensor],
                                  pixel_feat: Optional[Tensor],
                                  target4: Tensor) -> Tensor:
        if dynamic_prototypes is None or pixel_feat is None:
            return target4.sum() * 0.0
        target_proto, valid = self._masked_average_feature(
            pixel_feat.detach(), target4)
        if not bool(valid.any()):
            return dynamic_prototypes.sum() * 0.0
        pred = F.normalize(dynamic_prototypes[valid], dim=-1)
        target = F.normalize(target_proto[valid], dim=-1)
        return (1.0 - (pred * target).sum(dim=-1)).mean()

    def _object_support_loss(self,
                             object_support_logits2: Optional[Tensor],
                             target4: Tensor,
                             depth_prior: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if object_support_logits2 is None:
            zero = target4.sum() * 0.0
            return zero, zero
        target = self.object_support_target_from_target4(
            target4,
            boundary_kernel=self.support_boundary_kernel,
            contact_kernel=self.support_contact_kernel,
            depth_prior=depth_prior if self.use_depth_aware_support else None,
            depth_contact_tau=self.depth_contact_tau,
            depth_contact_weight=self.depth_contact_weight,
            contact2d_weight=(self.contact2d_weight
                              if self.use_depth_aware_support else 0.5))
        if target.shape[-2:] != object_support_logits2.shape[-2:]:
            target = resize(
                target,
                size=object_support_logits2.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False).clamp(0.0, 1.0)
        loss_bce = F.binary_cross_entropy_with_logits(
            object_support_logits2, target)
        loss_dice = self._dice_loss(object_support_logits2, target)
        return loss_bce, loss_dice

    @staticmethod
    def _absent_topk_loss(logits: Tensor,
                          absent_mask: Tensor,
                          topk_ratio: float = 0.01) -> Tensor:
        if logits.dim() != 4:
            raise ValueError(f"Expected logits [B,N,H,W], got {tuple(logits.shape)}")
        absent_mask = absent_mask.to(device=logits.device, dtype=torch.bool)
        if absent_mask.shape != logits.shape[:2]:
            raise ValueError(
                f"Expected absent mask {tuple(logits.shape[:2])}, got {tuple(absent_mask.shape)}")
        if not bool(absent_mask.any()):
            return logits.sum() * 0.0
        values = logits[absent_mask].reshape(int(absent_mask.sum()), -1)
        k = max(int(values.shape[-1] * float(topk_ratio)), 1)
        topk = values.topk(k, dim=-1).values
        return F.softplus(topk).mean()

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        if inputs.dim() == 5:
            return self.loss_video_memory(inputs, data_samples)
        depth_prior = self._depth_prior_from_samples(
            data_samples, inputs.shape[-2:], inputs.device, inputs.dtype)
        target4 = self._target4_from_samples(data_samples, inputs.device)
        mask_pilot_targets = None
        if self.semantic_output_mode == "fixed4":
            mask_pilot_targets = target4
        elif self.semantic_output_mode == "fixed5":
            mask_pilot_targets = self._target5_from_samples(
                data_samples, inputs.device)
        out = self.forward_relation(
            inputs,
            route="slow",
            depth_prior=depth_prior,
            mask_pilot_targets=mask_pilot_targets,
            profile=False)
        logits4 = out["logits4"]
        target4 = target4.to(device=logits4.device)
        target5_aux = None
        if (mask_pilot_targets is not None and mask_pilot_targets.dim() == 4 and
                mask_pilot_targets.shape[1] == 5):
            target5_aux = mask_pilot_targets.to(device=logits4.device)
        if self.semantic_output_mode == "softmax6":
            logits_softmax = out["logits_softmax"]
            target_softmax = self._target_softmax_from_samples(
                data_samples, logits_softmax.device)
            class_weight = (
                logits_softmax.new_tensor(self.softmax_class_weight)
                if self.softmax_class_weight is not None else None)
            logits_for_loss = out.get("all_logits_softmax") or [logits_softmax]
            loss_ce_layers = []
            loss_dice_layers = []
            for layer_logits in logits_for_loss:
                if bool((target_softmax != 255).any()):
                    loss_ce_layers.append(
                        F.cross_entropy(
                            layer_logits,
                            target_softmax,
                            ignore_index=255,
                            weight=class_weight))
                else:
                    loss_ce_layers.append(layer_logits.sum() * 0.0)
                loss_dice_layers.append(
                    self._softmax_dice_loss(
                        layer_logits, target_softmax, class_weight=class_weight))
            losses = dict(
                loss_ce=loss_ce_layers[-1] * self.loss_bce_weight,
                loss_softmax_dice=loss_dice_layers[-1] * self.loss_dice_weight)
            if len(logits_for_loss) > 1:
                for layer_idx, (layer_ce, layer_dice) in enumerate(
                        zip(loss_ce_layers[:-1], loss_dice_layers[:-1])):
                    losses[f"d{layer_idx}.loss_ce"] = (
                        layer_ce * self.loss_bce_weight)
                    losses[f"d{layer_idx}.loss_softmax_dice"] = (
                        layer_dice * self.loss_dice_weight)
                for layer_idx, (layer_ce, layer_dice) in enumerate(
                        zip(loss_ce_layers, loss_dice_layers)):
                    losses[f"vis_decoder{layer_idx}_ce"] = layer_ce.detach()
                    losses[f"vis_decoder{layer_idx}_softmax_dice"] = layer_dice.detach()
            class_logits_for_loss = out.get("all_class_logits")
            if class_logits_for_loss is None and out.get("class_logits") is not None:
                class_logits_for_loss = [out["class_logits"]]
            if class_logits_for_loss is not None and self.loss_cls_weight > 0.0:
                loss_cls_layers = [
                    self._fixed_query_class_loss(
                        layer_logits,
                        target4,
                        absent_area_eps=self.absent_slot_area_eps,
                        no_object_weight=self.loss_cls_no_object_weight)
                    for layer_logits in class_logits_for_loss
                ]
                losses["loss_cls"] = loss_cls_layers[-1] * self.loss_cls_weight
                if len(loss_cls_layers) > 1:
                    for layer_idx, layer_cls in enumerate(loss_cls_layers[:-1]):
                        losses[f"d{layer_idx}.loss_cls"] = (
                            layer_cls * self.loss_cls_weight)
                    for layer_idx, layer_cls in enumerate(loss_cls_layers):
                        losses[f"vis_decoder{layer_idx}_cls"] = layer_cls.detach()
        else:
            if self.semantic_output_mode in ("hungarian5", "hungarian5_sam_distractor"):
                target_for_loss = (
                    self._hungarian5d_targets_from_samples(
                        data_samples, logits4.device)
                    if self.semantic_output_mode == "hungarian5_sam_distractor"
                    else self._target5_from_samples(data_samples, logits4.device))
                logits_for_loss = out.get("all_query_logits5") or [out["query_logits5"]]
                class_logits_for_loss = out.get("all_class_logits")
                if class_logits_for_loss is None and out.get("class_logits") is not None:
                    class_logits_for_loss = [out["class_logits"]]
                if class_logits_for_loss is None:
                    raise RuntimeError(
                        f"{self.semantic_output_mode} mode requires class logits")
                loss_bce_layers = []
                loss_dice_layers = []
                loss_cls_layers = []
                for layer_logits, layer_class_logits in zip(
                        logits_for_loss, class_logits_for_loss):
                    if self.semantic_output_mode == "hungarian5_sam_distractor":
                        layer_bce, layer_dice, layer_cls = (
                            self._hungarian_query_losses_from_targets(
                                layer_logits,
                                layer_class_logits,
                                target_for_loss,
                                no_object_idx=6))
                    else:
                        layer_bce, layer_dice, layer_cls = self._hungarian_query_losses(
                            layer_logits, layer_class_logits, target_for_loss)
                    loss_bce_layers.append(layer_bce)
                    loss_dice_layers.append(layer_dice)
                    loss_cls_layers.append(layer_cls)
                losses = dict(
                    loss_bce=loss_bce_layers[-1] * self.loss_bce_weight,
                    loss_dice=loss_dice_layers[-1] * self.loss_dice_weight,
                    loss_cls=loss_cls_layers[-1] * self.loss_cls_weight)
                if len(loss_bce_layers) > 1:
                    for layer_idx, (layer_bce, layer_dice, layer_cls) in enumerate(
                            zip(loss_bce_layers[:-1], loss_dice_layers[:-1],
                                loss_cls_layers[:-1])):
                        losses[f"d{layer_idx}.loss_bce"] = (
                            layer_bce * self.loss_bce_weight)
                        losses[f"d{layer_idx}.loss_dice"] = (
                            layer_dice * self.loss_dice_weight)
                        losses[f"d{layer_idx}.loss_cls"] = (
                            layer_cls * self.loss_cls_weight)
                    for layer_idx, (layer_bce, layer_dice, layer_cls) in enumerate(
                            zip(loss_bce_layers, loss_dice_layers, loss_cls_layers)):
                        losses[f"vis_decoder{layer_idx}_bce"] = layer_bce.detach()
                        losses[f"vis_decoder{layer_idx}_dice"] = layer_dice.detach()
                        losses[f"vis_decoder{layer_idx}_cls"] = layer_cls.detach()
            elif self.semantic_output_mode == "fixed5":
                target_for_loss = mask_pilot_targets.to(device=logits4.device)
                logits_for_loss = out.get("all_logits5") or [out["logits5"]]
            else:
                target_for_loss = target4
                logits_for_loss = out.get("all_logits4") or [logits4]
            if self.semantic_output_mode not in ("hungarian5", "hungarian5_sam_distractor"):
                loss_bce_layers = [
                    self._slot_bce_loss(
                        layer_logits,
                        target_for_loss,
                        reduction=self.slot_loss_reduction,
                        slot_weights=self.slot_loss_weights)
                    for layer_logits in logits_for_loss
                ]
                loss_dice_layers = [
                    self._dice_loss(
                        layer_logits,
                        target_for_loss,
                        reduction=self.slot_loss_reduction,
                        slot_weights=self.slot_loss_weights)
                    for layer_logits in logits_for_loss
                ]
                loss_tversky_layers = []
                if self.loss_tversky_weight > 0.0:
                    loss_tversky_layers = [
                        self._tversky_loss(
                            layer_logits,
                            target_for_loss,
                            alpha=self.tversky_alpha,
                            beta=self.tversky_beta,
                            eps=self.tversky_eps,
                            reduction=self.slot_loss_reduction,
                            slot_weights=self.slot_loss_weights)
                        for layer_logits in logits_for_loss
                    ]
                losses = dict(
                    loss_bce=loss_bce_layers[-1] * self.loss_bce_weight,
                    loss_dice=loss_dice_layers[-1] * self.loss_dice_weight)
                if loss_tversky_layers:
                    losses["loss_tversky"] = (
                        loss_tversky_layers[-1] * self.loss_tversky_weight)
                if (self.loss_point_bce_weight > 0.0 or
                        self.loss_point_dice_weight > 0.0):
                    point_bce, point_dice = self._point_sampled_slot_losses(
                        logits_for_loss[-1],
                        target_for_loss,
                        num_points=self.point_loss_num_points,
                        oversample_ratio=self.point_loss_oversample_ratio,
                        importance_sample_ratio=(
                            self.point_loss_importance_sample_ratio),
                        reduction=self.slot_loss_reduction,
                        slot_weights=self.slot_loss_weights)
                    if self.loss_point_bce_weight > 0.0:
                        losses["loss_point_bce"] = (
                            point_bce * self.loss_point_bce_weight)
                    if self.loss_point_dice_weight > 0.0:
                        losses["loss_point_dice"] = (
                            point_dice * self.loss_point_dice_weight)
                if len(logits_for_loss) > 1:
                    for layer_idx, (layer_bce, layer_dice) in enumerate(
                            zip(loss_bce_layers[:-1], loss_dice_layers[:-1])):
                        losses[f"d{layer_idx}.loss_bce"] = (
                            layer_bce * self.loss_bce_weight)
                        losses[f"d{layer_idx}.loss_dice"] = (
                            layer_dice * self.loss_dice_weight)
                    if loss_tversky_layers:
                        for layer_idx, layer_tversky in enumerate(
                                loss_tversky_layers[:-1]):
                            losses[f"d{layer_idx}.loss_tversky"] = (
                                layer_tversky * self.loss_tversky_weight)
                    for layer_idx, (layer_bce, layer_dice) in enumerate(
                            zip(loss_bce_layers, loss_dice_layers)):
                        losses[f"vis_decoder{layer_idx}_bce"] = layer_bce.detach()
                        losses[f"vis_decoder{layer_idx}_dice"] = layer_dice.detach()
                    if loss_tversky_layers:
                        for layer_idx, layer_tversky in enumerate(loss_tversky_layers):
                            losses[f"vis_decoder{layer_idx}_tversky"] = (
                                layer_tversky.detach())
            if self.semantic_output_mode in ("fixed4", "fixed5") and self.loss_cls_weight > 0.0:
                class_logits_for_loss = out.get("all_class_logits")
                if class_logits_for_loss is None and out.get("class_logits") is not None:
                    class_logits_for_loss = [out["class_logits"]]
                if class_logits_for_loss is not None:
                    loss_cls_layers = [
                        self._fixed_query_class_loss(
                            layer_logits,
                            target_for_loss,
                            absent_area_eps=self.absent_slot_area_eps,
                            no_object_weight=self.loss_cls_no_object_weight)
                        for layer_logits in class_logits_for_loss
                    ]
                    losses["loss_cls"] = loss_cls_layers[-1] * self.loss_cls_weight
                    if len(loss_cls_layers) > 1:
                        for layer_idx, layer_cls in enumerate(loss_cls_layers[:-1]):
                            losses[f"d{layer_idx}.loss_cls"] = (
                                layer_cls * self.loss_cls_weight)
                        for layer_idx, layer_cls in enumerate(loss_cls_layers):
                            losses[f"vis_decoder{layer_idx}_cls"] = layer_cls.detach()
            if (self.semantic_output_mode in ("fixed4", "fixed5") and
                    self.loss_mask_pilot_weight > 0.0):
                if self.semantic_output_mode == "fixed5":
                    mp_logits_for_loss = out.get("mp_all_logits5")
                else:
                    mp_logits_for_loss = out.get("mp_all_logits4")
                if mp_logits_for_loss:
                    mp_bce_layers = [
                        self._slot_bce_loss(
                            layer_logits,
                            target_for_loss,
                            reduction=self.slot_loss_reduction,
                            slot_weights=self.slot_loss_weights)
                        for layer_logits in mp_logits_for_loss
                    ]
                    mp_dice_layers = [
                        self._dice_loss(
                            layer_logits,
                            target_for_loss,
                            reduction=self.slot_loss_reduction,
                            slot_weights=self.slot_loss_weights)
                        for layer_logits in mp_logits_for_loss
                    ]
                    mp_tversky_layers = []
                    if self.loss_tversky_weight > 0.0:
                        mp_tversky_layers = [
                            self._tversky_loss(
                                layer_logits,
                                target_for_loss,
                                alpha=self.tversky_alpha,
                                beta=self.tversky_beta,
                                eps=self.tversky_eps,
                                reduction=self.slot_loss_reduction,
                                slot_weights=self.slot_loss_weights)
                            for layer_logits in mp_logits_for_loss
                        ]
                    mp_weight = self.loss_mask_pilot_weight
                    losses["mp.loss_bce"] = (
                        mp_bce_layers[-1] * self.loss_bce_weight * mp_weight)
                    losses["mp.loss_dice"] = (
                        mp_dice_layers[-1] * self.loss_dice_weight * mp_weight)
                    if mp_tversky_layers:
                        losses["mp.loss_tversky"] = (
                            mp_tversky_layers[-1] *
                            self.loss_tversky_weight * mp_weight)
                    if len(mp_logits_for_loss) > 1:
                        for layer_idx, (layer_bce, layer_dice) in enumerate(
                                zip(mp_bce_layers[:-1], mp_dice_layers[:-1])):
                            losses[f"mp.d{layer_idx}.loss_bce"] = (
                                layer_bce * self.loss_bce_weight * mp_weight)
                            losses[f"mp.d{layer_idx}.loss_dice"] = (
                                layer_dice * self.loss_dice_weight * mp_weight)
                        if mp_tversky_layers:
                            for layer_idx, layer_tversky in enumerate(
                                    mp_tversky_layers[:-1]):
                                losses[f"mp.d{layer_idx}.loss_tversky"] = (
                                    layer_tversky *
                                    self.loss_tversky_weight * mp_weight)
                        for layer_idx, (layer_bce, layer_dice) in enumerate(
                                zip(mp_bce_layers, mp_dice_layers)):
                            losses[f"vis_mp_decoder{layer_idx}_bce"] = layer_bce.detach()
                            losses[f"vis_mp_decoder{layer_idx}_dice"] = layer_dice.detach()
                        if mp_tversky_layers:
                            for layer_idx, layer_tversky in enumerate(mp_tversky_layers):
                                losses[f"vis_mp_decoder{layer_idx}_tversky"] = (
                                    layer_tversky.detach())
                    mp_class_logits_for_loss = out.get("mp_all_class_logits")
                    if mp_class_logits_for_loss is None and out.get("mp_class_logits") is not None:
                        mp_class_logits_for_loss = [out["mp_class_logits"]]
                    if mp_class_logits_for_loss is not None and self.loss_cls_weight > 0.0:
                        mp_cls_layers = [
                            self._fixed_query_class_loss(
                                layer_logits,
                                target_for_loss,
                                absent_area_eps=self.absent_slot_area_eps,
                                no_object_weight=self.loss_cls_no_object_weight)
                            for layer_logits in mp_class_logits_for_loss
                        ]
                        losses["mp.loss_cls"] = (
                            mp_cls_layers[-1] * self.loss_cls_weight * mp_weight)
                        if len(mp_cls_layers) > 1:
                            for layer_idx, layer_cls in enumerate(mp_cls_layers[:-1]):
                                losses[f"mp.d{layer_idx}.loss_cls"] = (
                                    layer_cls * self.loss_cls_weight * mp_weight)
                            for layer_idx, layer_cls in enumerate(mp_cls_layers):
                                losses[f"vis_mp_decoder{layer_idx}_cls"] = layer_cls.detach()
        if self.loss_proto_coherence_weight > 0.0:
            losses["loss_proto_coherence"] = (
                self._prototype_coherence_loss(
                    out.get("dynamic_prototypes"),
                    out.get("pixel_feat"),
                    target4) * self.loss_proto_coherence_weight)
        if (self.loss_object_support_bce_weight > 0.0 or
                self.loss_object_support_dice_weight > 0.0):
            loss_support_bce, loss_support_dice = self._object_support_loss(
                out.get("object_support_logits2"), target4, depth_prior=depth_prior)
            if self.loss_object_support_bce_weight > 0.0:
                losses["loss_object_support_bce"] = (
                    loss_support_bce * self.loss_object_support_bce_weight)
            if self.loss_object_support_dice_weight > 0.0:
                losses["loss_object_support_dice"] = (
                    loss_support_dice * self.loss_object_support_dice_weight)
        object_area = target4[:, 2:4].flatten(2).mean(-1)
        object_absent = object_area <= self.absent_slot_area_eps
        if self.loss_object_support_absent_topk_weight > 0.0:
            object_support_logits2 = out.get("object_support_logits2")
            if object_support_logits2 is None:
                losses["loss_object_support_absent_topk"] = target4.sum() * 0.0
            else:
                losses["loss_object_support_absent_topk"] = (
                    self._absent_topk_loss(
                        object_support_logits2,
                        object_absent,
                        topk_ratio=self.absent_topk_ratio) *
                    self.loss_object_support_absent_topk_weight)
        if self.loss_object_final_absent_topk_weight > 0.0:
            losses["loss_object_final_absent_topk"] = (
                self._absent_topk_loss(
                    logits4[:, 2:4],
                    object_absent,
                    topk_ratio=self.absent_topk_ratio) *
                self.loss_object_final_absent_topk_weight)
        if (self.loss_fixed5_object_absent_topk_weight > 0.0 and
                out.get("logits5") is not None and target5_aux is not None):
            logits5 = out["logits5"]
            target5_resized = target5_aux
            if logits5.shape[-2:] != target5_resized.shape[-2:]:
                target5_resized = resize(
                    target5_resized,
                    size=logits5.shape[-2:],
                    mode="nearest",
                    align_corners=None,
                    warning=False)
            object_absent5 = (
                target5_resized[:, 2:5].flatten(2).mean(-1) <=
                self.absent_slot_area_eps)
            losses["loss_fixed5_object_absent_topk"] = (
                self._absent_topk_loss(
                    logits5[:, 2:5],
                    object_absent5,
                    topk_ratio=self.absent_topk_ratio) *
                self.loss_fixed5_object_absent_topk_weight)
        if self.loss_opposite_hand_absent_topk_weight > 0.0:
            hand_area = target4[:, :2].flatten(2).mean(-1)
            single_hand = (hand_area > self.absent_slot_area_eps).sum(dim=1) == 1
            absent_hand = (hand_area <= self.absent_slot_area_eps) & single_hand[:, None]
            losses["loss_opposite_hand_absent_topk"] = (
                self._absent_topk_loss(
                    logits4[:, :2],
                    absent_hand,
                    topk_ratio=self.absent_topk_ratio) *
                self.loss_opposite_hand_absent_topk_weight)
        if self.loss_cooccurrence_weight > 0.0:
            cooccurrence_logits = out.get("logits5", logits4)
            losses["loss_cooccurrence"] = (
                self._decoder_cooccurrence_loss(cooccurrence_logits) *
                self.loss_cooccurrence_weight)
        if self.loss_relation_consistency_weight > 0.0:
            relation_logits = out.get("logits5", logits4)
            losses["loss_relation_consistency"] = (
                self._relation_consistency_loss(relation_logits) *
                self.loss_relation_consistency_weight)
        if (self.loss_compat6_ce_weight > 0.0 or
                self.loss_compat6_dice_weight > 0.0):
            logits5 = out.get("logits5")
            if logits5 is None:
                zero = logits4.sum() * 0.0
                loss_compat6_ce = zero
                loss_compat6_dice = zero
            else:
                target6 = self._target6_from_samples(data_samples, logits5.device)
                loss_compat6_ce, loss_compat6_dice = self._compat6_soft_losses(
                    logits5, target6)
            if self.loss_compat6_ce_weight > 0.0:
                losses["loss_compat6_ce"] = (
                    loss_compat6_ce * self.loss_compat6_ce_weight)
            if self.loss_compat6_dice_weight > 0.0:
                losses["loss_compat6_dice"] = (
                    loss_compat6_dice * self.loss_compat6_dice_weight)
        if (self.loss_to_consistency_weight > 0.0 and
                out.get("logits5") is not None and target5_aux is not None):
            logits5 = out["logits5"]
            target5_resized = target5_aux
            if logits5.shape[-2:] != target5_resized.shape[-2:]:
                target5_resized = resize(
                    target5_resized,
                    size=logits5.shape[-2:],
                    mode="nearest",
                    align_corners=None,
                    warning=False)
            losses["loss_to_consistency"] = (
                self._two_hand_object_consistency_loss(
                    logits5, target5_resized) *
                self.loss_to_consistency_weight)
        if self.loss_lr_exclusivity_weight > 0.0:
            relation_logits = out.get("logits5", logits4)
            losses["loss_lr_exclusivity"] = (
                self._left_right_exclusivity_loss(relation_logits) *
                self.loss_lr_exclusivity_weight)
        if self.loss_coarse_bce_weight > 0.0 or self.loss_coarse_dice_weight > 0.0:
            coarse_logits = out["coarse_logits"]
            coarse_target = self._coarse_target_from_target4(target4)
            if coarse_logits.shape[-2:] != coarse_target.shape[-2:]:
                coarse_logits = resize(
                    coarse_logits,
                    size=coarse_target.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False)
            if self.loss_coarse_bce_weight > 0.0:
                losses["loss_coarse_bce"] = (
                    F.binary_cross_entropy_with_logits(
                        coarse_logits, coarse_target) * self.loss_coarse_bce_weight)
            if self.loss_coarse_dice_weight > 0.0:
                losses["loss_coarse_dice"] = (
                    self._dice_loss(coarse_logits, coarse_target) *
                    self.loss_coarse_dice_weight)
        if self.loss_coarse_reliability_weight > 0.0:
            reliability_logits = out.get("coarse_reliability_logits")
            coarse_target = self._coarse_target_from_target4(target4)
            if reliability_logits is None:
                losses["loss_coarse_reliability"] = (
                    target4.sum() * 0.0)
            else:
                if reliability_logits.shape[-2:] != coarse_target.shape[-2:]:
                    reliability_logits = resize(
                        reliability_logits,
                        size=coarse_target.shape[-2:],
                        mode="bilinear",
                        align_corners=self.align_corners,
                        warning=False)
                prior_prob = out.get("coarse_prior_prob")
                if prior_prob is None:
                    prior_prob = out["coarse_logits"].sigmoid()
                if prior_prob.shape[-2:] != coarse_target.shape[-2:]:
                    prior_prob = resize(
                        prior_prob,
                        size=coarse_target.shape[-2:],
                        mode="bilinear",
                        align_corners=self.align_corners,
                        warning=False)
                reliability_target = (
                    1.0 - (prior_prob.detach().clamp(0.0, 1.0) -
                           coarse_target).abs()).clamp(0.0, 1.0)
                losses["loss_coarse_reliability"] = (
                    F.binary_cross_entropy_with_logits(
                        reliability_logits,
                        reliability_target) *
                    self.loss_coarse_reliability_weight)
        if (self.loss_contact_guidance_bce_weight > 0.0 or
                self.loss_contact_guidance_dice_weight > 0.0):
            contact_logits = out.get("contact_guidance_logits")
            contact_target = self._contact_target_from_samples(
                data_samples,
                inputs.shape[-2:],
                logits4.device,
                logits4.dtype)
            if contact_logits is None or contact_target is None:
                zero = logits4.sum() * 0.0
                loss_contact_bce = zero
                loss_contact_dice = zero
            else:
                if contact_logits.shape[-2:] != contact_target.shape[-2:]:
                    contact_logits = resize(
                        contact_logits,
                        size=contact_target.shape[-2:],
                        mode="bilinear",
                        align_corners=self.align_corners,
                        warning=False)
                loss_contact_bce = F.binary_cross_entropy_with_logits(
                    contact_logits, contact_target)
                loss_contact_dice = self._dice_loss(contact_logits, contact_target)
            if self.loss_contact_guidance_bce_weight > 0.0:
                losses["loss_contact_guidance_bce"] = (
                    loss_contact_bce * self.loss_contact_guidance_bce_weight)
            if self.loss_contact_guidance_dice_weight > 0.0:
                losses["loss_contact_guidance_dice"] = (
                    loss_contact_dice * self.loss_contact_guidance_dice_weight)
        if self.loss_physical_weight > 0.0:
            loss_physical, physical_terms = physical_relation_loss(
                logits4,
                area_tau=self.physical_area_tau,
                support_kernel=self.physical_support_kernel,
                min_support_ratio=self.physical_min_support_ratio)
            losses["loss_physical"] = loss_physical * self.loss_physical_weight
            for name, value in physical_terms.items():
                losses[f"vis_physical_{name}"] = value
        return losses

    def _resize_video_head_outputs(self,
                                   raw: Dict[str, Tensor],
                                   size: Sequence[int]) -> Dict[str, object]:
        out = dict(raw)
        for key in ("logits4", "logits5", "coarse_logits"):
            if key in out and out[key] is not None:
                out[key] = resize(
                    out[key],
                    size=size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False)
        for key in ("all_logits4", "all_logits5"):
            if out.get(key) is not None:
                out[key] = [
                    resize(
                        logits,
                        size=size,
                        mode="bilinear",
                        align_corners=self.align_corners,
                        warning=False)
                    for logits in out[key]
                ]
        return out

    def _video_fixed5_losses(self,
                             frame_outputs: List[Dict[str, object]],
                             target5: Tensor) -> dict:
        logits_layers = frame_outputs[0].get("all_logits5") or [
            frame_outputs[0]["logits5"]
        ]
        num_layers = len(logits_layers)
        loss_bce_layers: List[List[Tensor]] = [[] for _ in range(num_layers)]
        loss_dice_layers: List[List[Tensor]] = [[] for _ in range(num_layers)]
        loss_tversky_layers: List[List[Tensor]] = [[] for _ in range(num_layers)]
        loss_cls_layers: List[List[Tensor]] = [[] for _ in range(num_layers)]
        coarse_bce = []
        coarse_dice = []
        for time_idx, out in enumerate(frame_outputs):
            target = target5[:, time_idx]
            all_logits5 = out.get("all_logits5") or [out["logits5"]]
            for layer_idx, layer_logits in enumerate(all_logits5):
                loss_bce_layers[layer_idx].append(
                    self._slot_bce_loss(
                        layer_logits,
                        target,
                        reduction=self.slot_loss_reduction,
                        slot_weights=self.slot_loss_weights))
                loss_dice_layers[layer_idx].append(
                    self._dice_loss(
                        layer_logits,
                        target,
                        reduction=self.slot_loss_reduction,
                        slot_weights=self.slot_loss_weights))
                if self.loss_tversky_weight > 0.0:
                    loss_tversky_layers[layer_idx].append(
                        self._tversky_loss(
                            layer_logits,
                            target,
                            alpha=self.tversky_alpha,
                            beta=self.tversky_beta,
                            eps=self.tversky_eps,
                            reduction=self.slot_loss_reduction,
                            slot_weights=self.slot_loss_weights))
            all_class_logits = out.get("all_class_logits")
            if all_class_logits is None and out.get("class_logits") is not None:
                all_class_logits = [out["class_logits"]]
            if all_class_logits is not None:
                for layer_idx, layer_class_logits in enumerate(all_class_logits):
                    loss_cls_layers[layer_idx].append(
                        self._fixed_query_class_loss(
                            layer_class_logits,
                            target,
                            absent_area_eps=self.absent_slot_area_eps,
                            no_object_weight=self.loss_cls_no_object_weight))
            if (self.loss_coarse_bce_weight > 0.0 or
                    self.loss_coarse_dice_weight > 0.0):
                coarse_target = (target[:, 2:5].amax(dim=1, keepdim=True) > 0.5).float()
                coarse_logits = out["coarse_logits"]
                coarse_bce.append(
                    F.binary_cross_entropy_with_logits(
                        coarse_logits, coarse_target))
                coarse_dice.append(self._dice_loss(coarse_logits, coarse_target))

        bce_mean = [torch.stack(values).mean() for values in loss_bce_layers]
        dice_mean = [torch.stack(values).mean() for values in loss_dice_layers]
        tversky_mean = [
            torch.stack(values).mean() for values in loss_tversky_layers
            if values
        ]
        cls_mean = [
            torch.stack(values).mean() for values in loss_cls_layers if values
        ]
        losses = dict(
            loss_bce=bce_mean[-1] * self.loss_bce_weight,
            loss_dice=dice_mean[-1] * self.loss_dice_weight)
        if tversky_mean:
            losses["loss_tversky"] = (
                tversky_mean[-1] * self.loss_tversky_weight)
        if cls_mean:
            losses["loss_cls"] = cls_mean[-1] * self.loss_cls_weight
        for layer_idx in range(num_layers - 1):
            losses[f"d{layer_idx}.loss_bce"] = (
                bce_mean[layer_idx] * self.loss_bce_weight)
            losses[f"d{layer_idx}.loss_dice"] = (
                dice_mean[layer_idx] * self.loss_dice_weight)
            if tversky_mean:
                losses[f"d{layer_idx}.loss_tversky"] = (
                    tversky_mean[layer_idx] * self.loss_tversky_weight)
            if layer_idx < len(cls_mean):
                losses[f"d{layer_idx}.loss_cls"] = (
                    cls_mean[layer_idx] * self.loss_cls_weight)
        for layer_idx in range(num_layers):
            losses[f"vis_decoder{layer_idx}_bce"] = bce_mean[layer_idx].detach()
            losses[f"vis_decoder{layer_idx}_dice"] = dice_mean[layer_idx].detach()
            if layer_idx < len(tversky_mean):
                losses[f"vis_decoder{layer_idx}_tversky"] = (
                    tversky_mean[layer_idx].detach())
            if layer_idx < len(cls_mean):
                losses[f"vis_decoder{layer_idx}_cls"] = cls_mean[layer_idx].detach()
        if coarse_bce and self.loss_coarse_bce_weight > 0.0:
            losses["loss_coarse_bce"] = (
                torch.stack(coarse_bce).mean() * self.loss_coarse_bce_weight)
        if coarse_dice and self.loss_coarse_dice_weight > 0.0:
            losses["loss_coarse_dice"] = (
                torch.stack(coarse_dice).mean() * self.loss_coarse_dice_weight)
        return losses

    @staticmethod
    def _slot_soft_dice_score(prob: Tensor,
                              target: Tensor,
                              eps: float = 1e-6) -> Tensor:
        if prob.shape[-2:] != target.shape[-2:]:
            prob = F.interpolate(
                prob.float(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False)
        prob = prob.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)
        inter = (prob * target).flatten(2).sum(-1)
        denom = prob.flatten(2).sum(-1) + target.flatten(2).sum(-1)
        return (2.0 * inter + float(eps)) / (denom + float(eps))

    def loss_video_memory(self, inputs: Tensor, data_samples: SampleList) -> dict:
        if self.semantic_output_mode != "fixed5":
            raise ValueError(
                "Video memory training currently requires fixed5 semantic output")
        batch_size, seq_len = inputs.shape[:2]
        if seq_len < 2:
            raise ValueError("Video memory training requires at least two frames")
        target5 = self._target5_video_from_samples(data_samples, inputs.device)
        if target5.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                f"Target sequence shape {tuple(target5.shape)} does not match "
                f"input B,T={(batch_size, seq_len)}")
        state = None
        frame_outputs: List[Dict[str, object]] = []
        read_energies = []
        read_gate_losses = []
        update_gate_losses = []
        for time_idx in range(seq_len):
            frame = inputs[:, time_idx]
            feats = self._extract_slow_feat(frame, depth_prior=None)
            prev_state = state
            raw = self.slow_head(
                feats,
                depth_prior=None,
                mask_pilot_targets=target5[:, time_idx],
                video_memory_state=state)
            if raw.get("memory_aux"):
                energy = raw["memory_aux"].get("memory_read_energy")
                if energy is not None:
                    read_energies.append(energy)
                read_gate = raw["memory_aux"].get("read_gate")
                if (read_gate is not None and prev_state is not None and
                        isinstance(prev_state, dict) and "masks" in prev_state):
                    read_target = self._slot_soft_dice_score(
                        prev_state["masks"], target5[:, time_idx]).detach()
                    read_gate_losses.append(
                        F.binary_cross_entropy(
                            read_gate.float().clamp(1e-4, 1.0 - 1e-4),
                            read_target.float()))
            if hasattr(self.slow_head, "update_video_memory"):
                write_masks = (
                    target5[:, time_idx]
                    if time_idx == 0 else raw["logits5"].sigmoid())
                state = self.slow_head.update_video_memory(
                    state,
                    raw["relation_feat"],
                    write_masks,
                    raw["coarse_logits"].sigmoid())
                if (state is not None and isinstance(state, dict) and
                        "last_update_gate" in state):
                    update_target = self._slot_soft_dice_score(
                        raw["logits5"].sigmoid(), target5[:, time_idx]).detach()
                    update_gate_losses.append(
                        F.binary_cross_entropy(
                            state["last_update_gate"].float().clamp(
                                1e-4, 1.0 - 1e-4),
                            update_target.float()))
                if self.video_memory_detach_state:
                    state = self._detach_video_memory_state(state)
            if time_idx > 0:
                frame_outputs.append(
                    self._resize_video_head_outputs(raw, frame.shape[-2:]))
        losses = self._video_fixed5_losses(frame_outputs, target5[:, 1:])
        if read_energies:
            losses["memory.loss_read_energy"] = (
                torch.stack(read_energies).mean() *
                self.loss_video_memory_read_weight)
        else:
            losses["memory.loss_read_energy"] = inputs.sum() * 0.0
        if read_gate_losses and self.loss_video_memory_read_gate_weight > 0.0:
            losses["memory.loss_read_gate"] = (
                torch.stack(read_gate_losses).mean() *
                self.loss_video_memory_read_gate_weight)
        if update_gate_losses and self.loss_video_memory_update_gate_weight > 0.0:
            losses["memory.loss_update_gate"] = (
                torch.stack(update_gate_losses).mean() *
                self.loss_video_memory_update_gate_weight)
        return losses

    @staticmethod
    def _detach_video_memory_state(
            state: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if state is None:
            return None
        detached: Dict[str, object] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                detached[key] = value.detach()
            else:
                detached[key] = value
        return detached

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        depth_prior = self._depth_prior_from_samples(
            data_samples, inputs.shape[-2:], inputs.device, inputs.dtype)
        out = self.forward_relation(
            inputs,
            route=self.test_cfg.get("route", "slow"),
            depth_prior=depth_prior)
        logits4 = out["logits4"]
        logits5 = out.get("logits5")
        if self.semantic_output_mode == "fixed5":
            prob5 = logits5.sigmoid()
            prob4 = foreground5_prob_to_multilabel4_prob(prob5)
            compat6 = foreground5_to_compat6(
                prob5,
                threshold=float(self.test_cfg.get("threshold", 0.5)))
        elif self.semantic_output_mode == "softmax6":
            logits6 = out["logits6"]
            prob6 = logits6.softmax(dim=1)
            prob4 = softmax6_prob_to_multilabel4_prob(prob6)
            compat6 = softmax6_to_compat6(prob6)
        else:
            prob4 = logits4.sigmoid()
            compat6 = multilabel4_to_compat6(
                prob4,
                threshold=float(self.test_cfg.get("threshold", 0.5)))
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(inputs.shape[0])]
        for idx, sample in enumerate(data_samples):
            data = {
                "seg_logits": PixelData(
                    data=out.get("logits5", out.get("logits6", out.get("logits_softmax", logits4)))[idx]),
                "pred_sem_seg": PixelData(data=compat6[idx:idx + 1]),
                "pred_sem_seg_4": PixelData(data=(prob4[idx] > 0.5).to(torch.uint8)),
            }
            if logits5 is not None:
                data["pred_sem_seg_5"] = PixelData(
                    data=(logits5[idx].sigmoid() > 0.5).to(torch.uint8))
            sample.set_data(data)
            sample.set_metainfo({
                "litehosv2_route": out["route"],
                "litehosv2_profile": out["profile"],
            })
        return data_samples
