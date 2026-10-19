"""Submission ablations at the SFSC interface; frozen production head is untouched.

without_pur: auxiliary foreground prediction remains; Fr=rho(F), g=GAP(Fr).
without_sfsc: no foreground head/loss/refiner; Fr=F, g=GAP(F).
Zero coarse logits in the latter mode satisfy the legacy state container only.
They are not passed to the masks, global pooling, or an enabled training loss.
"""

import torch
from torch import nn
from mmseg.registry import MODELS
from mmseg.models.decode_heads.litehosv2_relation_state_heads import (
    ConvBNAct,
    LiteHOSV2RelationSlotHead,
)


class TokenWiseFFN(nn.Module):
    """Keep the original post-norm/FFN path, replacing attention output by zero."""

    def __init__(self, layer):
        super().__init__()
        if layer.norm_first:
            raise ValueError("This control requires the production post-norm block")
        self.norm1, self.norm2 = layer.norm1, layer.norm2
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        self.dropout, self.dropout2 = layer.dropout, layer.dropout2
        self.activation = layer.activation

    def forward(self, x):
        x = self.norm1(x)
        return self.norm2(
            x
            + self.dropout2(
                self.linear2(self.dropout(self.activation(self.linear1(x))))
            )
        )


@MODELS.register_module()
class TCSVTFineAblationHead(LiteHOSV2RelationSlotHead):
    def __init__(self, global_pool_mode="weighted", token_mixing="attention", **kwargs):
        super().__init__(**kwargs)
        if global_pool_mode not in ("weighted", "ordinary"):
            raise ValueError(global_pool_mode)
        if token_mixing not in ("attention", "ffn"):
            raise ValueError(token_mixing)
        if not self.use_global_query_bias:
            raise ValueError("These controls require image-conditioned queries")
        self.global_pool_mode = global_pool_mode
        if token_mixing == "ffn":
            self.slot_mixer.layers = nn.ModuleList(
                [TokenWiseFFN(layer) for layer in self.slot_mixer.layers]
            )

    def _forward_slots(
        self, queries, relation_feat, coarse_prob, coarse_reliability=None
    ):
        if self.global_pool_mode == "ordinary":
            # Replace only the query initialization; dense P/U/R is unchanged.
            queries = self.slot_queries.unsqueeze(0) + relation_feat.mean(
                (-2, -1)
            ).unsqueeze(1)
        return super()._forward_slots(
            queries, relation_feat, coarse_prob, coarse_reliability
        )


@MODELS.register_module()
class TCSVTModuleAblationHead(LiteHOSV2RelationSlotHead):
    def __init__(self, sfsc_mode, **kwargs):
        if sfsc_mode not in ("without_pur", "without_sfsc"):
            raise ValueError(f"Unsupported SFSC ablation: {sfsc_mode}")
        super().__init__(**kwargs)
        # Fail rather than silently discard any unrelated experimental branch.
        for name in (
            "coarse_contact_guidance",
            "coarse_reliability_head",
            "coarse_reliability_aggregator",
            "refine_scse",
            "pid_aggregation",
            "concat_attention",
            "decoder_attention",
            "prototype_extractor",
            "video_memory_adapter",
            "streaming_memory_adapter",
            "reliability_query_proj",
            "query_evidence_update",
            "semantic_group_mixer",
            "relation_topology_mixer",
            "hand_object_refine_update",
            "hand_condition_object_refine",
            "mask_logit_refine",
        ):
            if getattr(self, name, None) is not None:
                raise ValueError(f"{name} must be disabled in module ablations")
        if (
            self.use_depth_prior
            or self.pixel_self_attn_num_layers
            or not self.use_global_query_bias
        ):
            raise ValueError(
                "Require RGB, no pixel attention, and global query bias enabled"
            )
        self.sfsc_mode = sfsc_mode
        if sfsc_mode == "without_sfsc":
            self.coarse_head = None
            self.refine = nn.Identity()
        else:
            self.refine = nn.Sequential(
                ConvBNAct(self.feat_channels, self.feat_channels, 3),
                ConvBNAct(self.feat_channels, self.feat_channels, 3),
            )

    def forward(
        self, feats, depth_prior=None, mask_pilot_targets=None, video_memory_state=None
    ):
        if depth_prior is not None or video_memory_state is not None:
            raise ValueError("Module ablations are RGB single-frame only")
        field = self._fuse_features(feats)
        refined = self.refine(field)
        global_context = refined.mean(dim=(-2, -1))
        queries = self.slot_queries.unsqueeze(0) + global_context.unsqueeze(1)
        # No foreground estimate participates in this computation.
        dummy = field.new_zeros(field.shape[0], 1, *field.shape[-2:])
        masks, classes, tokens, _, _ = self._forward_slots(queries, refined, dummy)
        if not self.return_intermediate:
            masks, classes = masks[-1:], classes[-1:]
        result = self._pack_outputs(masks, classes)
        result.update(
            coarse_logits=(
                self.coarse_head(field) if self.coarse_head is not None else dummy
            ),
            slot_tokens=tokens,
            relation_feat=refined,
            pixel_feat=field,
            global_context=global_context,
            coarse_prior_prob=None,
            coarse_uncertainty=None,
            coarse_reliability=None,
            memory_aux={},
        )
        return result
