# Copyright (c) OpenMMLab. All rights reserved.
"""Shared relation-state utilities for LiteHOSV2 V0.

The V0 public protocol is four independent foreground masks:
left hand, right hand, left object, and right object.  Legacy six-class
semantic labels remain useful for training/evaluation compatibility; class 5
(two-hand object) is mapped into both object slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


RELATION_SLOT_NAMES = (
    "left_hand",
    "right_hand",
    "left_object",
    "right_object",
)

CAREEGO_STYLE_NAMES = (
    "left_hand",
    "right_hand",
    "left_object",
    "right_object",
    "two_hand_object",
)

FOREGROUND5_NAMES = CAREEGO_STYLE_NAMES

ROUTE_NAMES = (
    "no_hand",
    "fast_propagate_refine",
    "slow_refresh",
)

ROUTE_TO_INDEX = {name: idx for idx, name in enumerate(ROUTE_NAMES)}


def apply_fixed5_presence_gate(prob5: Tensor,
                               class_logits: Tensor) -> Tensor:
    """Suppress fixed semantic slots classified as no-object.

    Fixed5 query ``q`` is supervised as semantic role ``q`` when present and
    as the final no-object class when absent. This zero-margin rule uses the
    trained presence head without adding parameters or a tuned threshold.
    """

    if prob5.dim() != 4 or prob5.shape[1] != 5:
        raise ValueError(
            f"Expected fixed5 probabilities [B,5,H,W], got {tuple(prob5.shape)}")
    if class_logits.dim() != 3 or class_logits.shape[1] != 5:
        raise ValueError(
            "Expected fixed5 class logits [B,5,C], got "
            f"{tuple(class_logits.shape)}")
    if class_logits.shape[0] != prob5.shape[0] or class_logits.shape[2] < 6:
        raise ValueError(
            "Fixed5 presence gating requires matching batches and at least "
            "five role classes plus no-object")

    role_idx = torch.arange(5, device=class_logits.device)
    role_logits = class_logits[:, role_idx, role_idx]
    no_object_logits = class_logits[:, :, -1]
    present = role_logits > no_object_logits
    return prob5 * present.to(dtype=prob5.dtype).unsqueeze(-1).unsqueeze(-1)


@dataclass
class LiteHOSRelationState:
    """Compact cached state shared by slow path, fast path, and gate."""

    mask_prob4: Tensor
    coarse_prob: Tensor
    slot_tokens: Tensor
    slot_prototypes: Tensor
    uncertainty4: Tensor
    age: Tensor
    route: str = "slow_refresh"

    def detach(self) -> "LiteHOSRelationState":
        return LiteHOSRelationState(
            mask_prob4=self.mask_prob4.detach(),
            coarse_prob=self.coarse_prob.detach(),
            slot_tokens=self.slot_tokens.detach(),
            slot_prototypes=self.slot_prototypes.detach(),
            uncertainty4=self.uncertainty4.detach(),
            age=self.age.detach(),
            route=self.route,
        )

    def to(self, *args, **kwargs) -> "LiteHOSRelationState":
        return LiteHOSRelationState(
            mask_prob4=self.mask_prob4.to(*args, **kwargs),
            coarse_prob=self.coarse_prob.to(*args, **kwargs),
            slot_tokens=self.slot_tokens.to(*args, **kwargs),
            slot_prototypes=self.slot_prototypes.to(*args, **kwargs),
            uncertainty4=self.uncertainty4.to(*args, **kwargs),
            age=self.age.to(*args, **kwargs),
            route=self.route,
        )


def labels6_to_multilabel4(labels: Tensor,
                           ignore_index: int = 255,
                           dtype: Optional[torch.dtype] = None) -> Tensor:
    """Map legacy 6-class labels to four independent V0 foreground slots.

    Input labels use the existing LiteHOS/InterFormer convention:
    0 background, 1 left hand, 2 right hand, 3 left object, 4 right object,
    5 two-hand object. Two-hand object is positive for both object slots.
    """

    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.dim() != 3:
        raise ValueError(
            f"Expected labels with shape [B,H,W] or [B,1,H,W], got {tuple(labels.shape)}"
        )
    out_dtype = dtype or torch.float32
    valid = labels != int(ignore_index)
    target = labels.new_zeros((labels.shape[0], 4, *labels.shape[-2:]),
                              dtype=torch.bool)
    target[:, 0] = (labels == 1) & valid
    target[:, 1] = (labels == 2) & valid
    target[:, 2] = ((labels == 3) | (labels == 5)) & valid
    target[:, 3] = ((labels == 4) | (labels == 5)) & valid
    return target.to(dtype=out_dtype)


def labels6_to_foreground5(labels: Tensor,
                           ignore_index: int = 255,
                           dtype: Optional[torch.dtype] = None) -> Tensor:
    """Map legacy labels to exact LH/RH/LO/RO/TO foreground masks."""

    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.dim() != 3:
        raise ValueError(
            f"Expected labels with shape [B,H,W] or [B,1,H,W], got {tuple(labels.shape)}"
        )
    out_dtype = dtype or torch.float32
    valid = labels != int(ignore_index)
    target = labels.new_zeros((labels.shape[0], 5, *labels.shape[-2:]),
                              dtype=torch.bool)
    target[:, 0] = (labels == 1) & valid
    target[:, 1] = (labels == 2) & valid
    target[:, 2] = (labels == 3) & valid
    target[:, 3] = (labels == 4) & valid
    target[:, 4] = (labels == 5) & valid
    return target.to(dtype=out_dtype)


def labels6_to_softmax6(labels: Tensor, ignore_index: int = 255) -> Tensor:
    """Return legacy 0..5 labels for standard 6-class softmax training."""

    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.dim() != 3:
        raise ValueError(
            f"Expected labels with shape [B,H,W] or [B,1,H,W], got {tuple(labels.shape)}"
        )
    target = labels.clone().long()
    valid = ((target >= 0) & (target <= 5)) | (target == int(ignore_index))
    target = target.masked_fill(~valid, int(ignore_index))
    return target


def softmax6_prob_to_multilabel4_prob(prob6: Tensor) -> Tensor:
    """Project standard BG/LH/RH/LO/RO/TO softmax probabilities to 4 slots."""

    if prob6.dim() != 4 or prob6.shape[1] != 6:
        raise ValueError(f"Expected [B,6,H,W], got {tuple(prob6.shape)}")
    return torch.stack(
        [
            prob6[:, 1],
            prob6[:, 2],
            (prob6[:, 3] + prob6[:, 5]).clamp(0.0, 1.0),
            (prob6[:, 4] + prob6[:, 5]).clamp(0.0, 1.0),
        ],
        dim=1)


def softmax6_to_compat6(prob6: Tensor) -> Tensor:
    """Project standard BG/LH/RH/LO/RO/TO softmax probabilities to labels."""

    if prob6.dim() != 4 or prob6.shape[1] != 6:
        raise ValueError(f"Expected [B,6,H,W], got {tuple(prob6.shape)}")
    return prob6.argmax(dim=1).to(torch.long)


def foreground5_prob_to_multilabel4_prob(prob5: Tensor) -> Tensor:
    """Project exact LH/RH/LO/RO/TO probabilities to four relation slots."""

    if prob5.dim() != 4 or prob5.shape[1] != 5:
        raise ValueError(f"Expected [B,5,H,W], got {tuple(prob5.shape)}")
    return torch.stack(
        [
            prob5[:, 0],
            prob5[:, 1],
            torch.maximum(prob5[:, 2], prob5[:, 4]),
            torch.maximum(prob5[:, 3], prob5[:, 4]),
        ],
        dim=1)


def foreground5_to_compat6(prob5: Tensor,
                           threshold: float = 0.5) -> Tensor:
    """Project exact foreground masks to BG/LH/RH/LO/RO/TO labels.

    Background has no query. Pixels with no active foreground class become
    background; active foreground classes compete by probability.
    """

    if prob5.dim() != 4 or prob5.shape[1] != 5:
        raise ValueError(f"Expected [B,5,H,W], got {tuple(prob5.shape)}")
    active = prob5 > float(threshold)
    scores = prob5.new_full((prob5.shape[0], 6, *prob5.shape[-2:]), -1.0)
    scores[:, 0] = 0.0
    scores[:, 1:] = prob5.masked_fill(~active, -1.0)
    return scores.argmax(dim=1)


def multilabel4_to_compat6(prob4: Tensor,
                           threshold: float = 0.5) -> Tensor:
    """Project four independent masks into the legacy 6-class display space.

    The V0 network keeps four independent sigmoid slots.  The legacy 6-class
    map is single-label, so projection uses a deterministic physical
    mutual-exclusion rule instead of write-order overwrites: active semantic
    candidates compete by probability, while the two object slots jointly form
    class 5 when both are active.
    """

    if prob4.dim() != 4 or prob4.shape[1] != 4:
        raise ValueError(f"Expected [B,4,H,W], got {tuple(prob4.shape)}")
    active = prob4 > float(threshold)
    scores = prob4.new_full(
        (prob4.shape[0], 6, *prob4.shape[-2:]), -1.0)
    scores[:, 0] = 0.0
    scores[:, 1] = prob4[:, 0].masked_fill(~active[:, 0], -1.0)
    scores[:, 2] = prob4[:, 1].masked_fill(~active[:, 1], -1.0)

    left_obj = active[:, 2]
    right_obj = active[:, 3]
    two_obj = left_obj & right_obj
    scores[:, 3] = prob4[:, 2].masked_fill(~(left_obj & ~two_obj), -1.0)
    scores[:, 4] = prob4[:, 3].masked_fill(~(right_obj & ~two_obj), -1.0)
    # Two-hand object needs both object slots.  Use the weaker object-slot
    # probability as confidence so a single very strong side cannot dominate.
    two_score = torch.minimum(prob4[:, 2], prob4[:, 3])
    scores[:, 5] = two_score.masked_fill(~two_obj, -1.0)
    return scores.argmax(dim=1)


def labels6_to_careego_style5(labels: Tensor,
                              ignore_index: int = 255,
                              dtype: Optional[torch.dtype] = None) -> Tensor:
    """Map legacy labels to CaRe-Ego-style foreground branch masks."""

    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.dim() != 3:
        raise ValueError(
            f"Expected labels with shape [B,H,W] or [B,1,H,W], got {tuple(labels.shape)}"
        )
    out_dtype = dtype or torch.float32
    valid = labels != int(ignore_index)
    target = labels.new_zeros((labels.shape[0], 5, *labels.shape[-2:]),
                              dtype=torch.bool)
    target[:, 0] = (labels == 1) & valid
    target[:, 1] = (labels == 2) & valid
    target[:, 2] = ((labels == 3) | (labels == 5)) & valid
    target[:, 3] = ((labels == 4) | (labels == 5)) & valid
    target[:, 4] = (labels == 5) & valid
    return target.to(dtype=out_dtype)


def multilabel4_to_careego_style5(prob4: Tensor,
                                  threshold: float = 0.5) -> Tensor:
    """Project four slots to CaRe-Ego-style branch predictions.

    Hands are mutually exclusive through a BG/LH/RH competition, matching the
    CaRe-Ego hand head. Object branches stay independent binary masks, and the
    two-object branch is derived from left/right object overlap.
    """

    if prob4.dim() != 4 or prob4.shape[1] != 4:
        raise ValueError(f"Expected [B,4,H,W], got {tuple(prob4.shape)}")
    active = prob4 > float(threshold)
    hand_scores = prob4.new_full(
        (prob4.shape[0], 3, *prob4.shape[-2:]), -1.0)
    hand_scores[:, 0] = 0.0
    hand_scores[:, 1] = prob4[:, 0].masked_fill(~active[:, 0], -1.0)
    hand_scores[:, 2] = prob4[:, 1].masked_fill(~active[:, 1], -1.0)
    hand_label = hand_scores.argmax(dim=1)

    left_obj = active[:, 2]
    right_obj = active[:, 3]
    pred = torch.zeros((prob4.shape[0], 5, *prob4.shape[-2:]),
                       dtype=torch.bool,
                       device=prob4.device)
    pred[:, 0] = hand_label == 1
    pred[:, 1] = hand_label == 2
    pred[:, 2] = left_obj
    pred[:, 3] = right_obj
    pred[:, 4] = left_obj & right_obj
    return pred


def mask_uncertainty4(prob4: Tensor) -> Tensor:
    """Per-slot Bernoulli uncertainty averaged spatially."""

    prob4 = prob4.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(prob4 * prob4.log() + (1.0 - prob4) * (1.0 - prob4).log())
    return entropy.flatten(2).mean(-1)


def compute_slot_prototypes(feat: Tensor,
                            mask_prob4: Tensor,
                            eps: float = 1e-6) -> Tensor:
    """Extract fixed-slot prototypes from a feature map and four masks."""

    if mask_prob4.shape[-2:] != feat.shape[-2:]:
        mask_prob4 = F.interpolate(
            mask_prob4,
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False)
    weights = mask_prob4.clamp_min(0.0)
    denom = weights.flatten(2).sum(-1).clamp_min(eps)
    proto = torch.einsum("bchw,bnhw->bnc", feat, weights) / denom.unsqueeze(-1)
    return proto


def relation_violation_score(mask_prob4: Tensor) -> Tensor:
    """Cheap physical-relation violation cue for gate training/inference."""

    if mask_prob4.dim() != 4 or mask_prob4.shape[1] != 4:
        raise ValueError(f"Expected [B,4,H,W], got {tuple(mask_prob4.shape)}")
    lh, rh, lo, ro = mask_prob4.split(1, dim=1)
    support_kernel = 9
    lh_support = F.max_pool2d(lh, support_kernel, stride=1, padding=support_kernel // 2)
    rh_support = F.max_pool2d(rh, support_kernel, stride=1, padding=support_kernel // 2)
    lo_outside = (lo * (1.0 - lh_support)).flatten(1).mean(-1)
    ro_outside = (ro * (1.0 - rh_support)).flatten(1).mean(-1)
    return 0.5 * (lo_outside + ro_outside)


def _slot_existence(prob: Tensor, area_tau: float) -> Tensor:
    area = prob.flatten(1).mean(-1)
    return 1.0 - torch.exp(-area / max(float(area_tau), 1e-6))


def _object_support_penalty(obj: Tensor,
                            hand: Tensor,
                            area_tau: float,
                            support_kernel: int,
                            min_support_ratio: float,
                            eps: float = 1e-6) -> Tensor:
    kernel = max(int(support_kernel), 1)
    if kernel % 2 == 0:
        kernel += 1
    hand_support = F.max_pool2d(
        hand, kernel, stride=1, padding=kernel // 2)
    supported = (obj * hand_support).flatten(1).sum(-1)
    total = obj.flatten(1).sum(-1).clamp_min(eps)
    support_ratio = supported / total
    obj_exists = _slot_existence(obj, area_tau=area_tau)
    return obj_exists * F.relu(float(min_support_ratio) - support_ratio)


def physical_relation_terms(prob4: Tensor,
                            area_tau: float = 0.01,
                            support_kernel: int = 31,
                            min_support_ratio: float = 0.03) -> dict[str, Tensor]:
    """Soft physical-relation penalties for four independent HOS slots.

    The terms are intentionally soft: they discourage impossible hand-object
    states without erasing the independent 4-slot protocol needed by future
    amodal segmentation.
    """

    if prob4.dim() != 4 or prob4.shape[1] != 4:
        raise ValueError(f"Expected [B,4,H,W], got {tuple(prob4.shape)}")
    lh, rh, lo, ro = prob4.split(1, dim=1)
    lh_exists = _slot_existence(lh, area_tau)
    rh_exists = _slot_existence(rh, area_tau)
    lo_exists = _slot_existence(lo, area_tau)
    ro_exists = _slot_existence(ro, area_tau)

    lo_without_lh = lo_exists * (1.0 - lh_exists)
    ro_without_rh = ro_exists * (1.0 - rh_exists)
    lo_support = _object_support_penalty(
        lo, lh, area_tau, support_kernel, min_support_ratio)
    ro_support = _object_support_penalty(
        ro, rh, area_tau, support_kernel, min_support_ratio)
    hand_overlap = (lh * rh).flatten(1).mean(-1)
    object_overlap = (lo * ro).flatten(1).mean(-1)
    two_hand_evidence = lh_exists * rh_exists
    invalid_two_hand_object = object_overlap * (1.0 - two_hand_evidence)

    return {
        "left_object_without_hand": lo_without_lh,
        "right_object_without_hand": ro_without_rh,
        "left_object_support": lo_support,
        "right_object_support": ro_support,
        "hand_overlap": hand_overlap,
        "invalid_two_hand_object": invalid_two_hand_object,
    }


def physical_relation_loss(logits4: Tensor,
                           area_tau: float = 0.01,
                           support_kernel: int = 31,
                           min_support_ratio: float = 0.03) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a scalar physical relation loss and detached term summary."""

    prob4 = logits4.sigmoid()
    terms = physical_relation_terms(
        prob4,
        area_tau=area_tau,
        support_kernel=support_kernel,
        min_support_ratio=min_support_ratio)
    loss = torch.stack([value.mean() for value in terms.values()]).mean()
    detached_terms = {
        name: value.detach().mean()
        for name, value in terms.items()
    }
    return loss, detached_terms


def physical_relation_diagnostics(prob4: Tensor,
                                  threshold: float = 0.5,
                                  min_area_ratio: float = 1e-4) -> dict[str, Tensor]:
    """Hard relation diagnostics used by evaluators and failure analysis."""

    if prob4.dim() != 4 or prob4.shape[1] != 4:
        raise ValueError(f"Expected [B,4,H,W], got {tuple(prob4.shape)}")
    pred = prob4 > float(threshold)
    lh, rh, lo, ro = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    total_pixels = float(prob4.shape[0] * prob4.shape[2] * prob4.shape[3])
    lh_area = lh.flatten(1).float().mean(-1)
    rh_area = rh.flatten(1).float().mean(-1)
    lo_area = lo.flatten(1).float().mean(-1)
    ro_area = ro.flatten(1).float().mean(-1)
    lh_exists = lh_area > float(min_area_ratio)
    rh_exists = rh_area > float(min_area_ratio)
    lo_exists = lo_area > float(min_area_ratio)
    ro_exists = ro_area > float(min_area_ratio)

    object_without_hand_count = (
        ((lo_exists & ~lh_exists).float()
         + (ro_exists & ~rh_exists).float()).sum())
    object_present_count = (lo_exists.float() + ro_exists.float()).sum().clamp_min(1.0)
    both_hands_exist = (lh_exists & rh_exists).view(-1, 1, 1)
    invalid_two = (lo & ro) & (~both_hands_exist)

    return {
        "object_without_hand_rate": object_without_hand_count / object_present_count,
        "invalid_two_hand_object_rate": invalid_two.float().sum() / max(total_pixels, 1.0),
        "hand_overlap_rate": (lh & rh).float().sum() / max(total_pixels, 1.0),
        "hand_object_overlap_rate": (
            ((lh | rh) & (lo | ro)).float().sum() / max(total_pixels, 1.0)),
        "relation_violation_score": relation_violation_score(prob4).mean(),
    }


def oracle_route_labels(fast_iou_gap: Tensor,
                        relation_violation: Tensor,
                        state_age: Tensor,
                        hand_area: Tensor,
                        iou_gap_thr: float = 0.03,
                        violation_thr: float = 0.05,
                        max_fast_age: int = 5,
                        no_hand_area_thr: float = 1e-4) -> Tensor:
    """Build supervised V0 gate labels from oracle route diagnostics.

    Returns route indices using ``ROUTE_NAMES``:
    0 no_hand, 1 fast_propagate_refine, 2 slow_refresh.
    """

    fast_iou_gap = fast_iou_gap.reshape(-1)
    relation_violation = relation_violation.reshape(-1).to(fast_iou_gap.device)
    state_age = state_age.reshape(-1).to(fast_iou_gap.device)
    hand_area = hand_area.reshape(-1).to(fast_iou_gap.device)
    label = torch.full_like(
        fast_iou_gap,
        ROUTE_TO_INDEX["slow_refresh"],
        dtype=torch.long)
    no_hand = hand_area <= float(no_hand_area_thr)
    safe_fast = (
        (fast_iou_gap <= float(iou_gap_thr))
        & (relation_violation <= float(violation_thr))
        & (state_age < float(max_fast_age))
        & (~no_hand)
    )
    label[safe_fast] = ROUTE_TO_INDEX["fast_propagate_refine"]
    label[no_hand] = ROUTE_TO_INDEX["no_hand"]
    return label


def make_state_from_logits4(logits4: Tensor,
                            feature: Tensor,
                            coarse_logits: Optional[Tensor] = None,
                            route: str = "slow_refresh",
                            age: Optional[Tensor] = None) -> LiteHOSRelationState:
    """Build a detached relation state from current logits and feature."""

    prob4 = logits4.sigmoid()
    if coarse_logits is None:
        coarse_prob = prob4.amax(dim=1, keepdim=True)
    else:
        coarse_prob = coarse_logits.sigmoid()
        if coarse_prob.shape[-2:] != logits4.shape[-2:]:
            coarse_prob = F.interpolate(
                coarse_prob,
                size=logits4.shape[-2:],
                mode="bilinear",
                align_corners=False)
    slot_prototypes = compute_slot_prototypes(feature, prob4)
    if slot_prototypes.shape[-1] < feature.shape[1]:
        slot_tokens = slot_prototypes
    else:
        slot_tokens = slot_prototypes
    if age is None:
        age = torch.zeros(logits4.shape[0], device=logits4.device)
    return LiteHOSRelationState(
        mask_prob4=prob4,
        coarse_prob=coarse_prob,
        slot_tokens=slot_tokens,
        slot_prototypes=slot_prototypes,
        uncertainty4=mask_uncertainty4(prob4),
        age=age.to(device=logits4.device, dtype=logits4.dtype),
        route=route,
    )
