"""New module ablation: no SFSC and no image-level token conditioning."""

import torch
from torch import nn
from mmseg.registry import MODELS
from mmseg.models.decode_heads.litehosv2_relation_state_heads import (
    LiteHOSV2RelationSlotHead,
)


@MODELS.register_module()
class NoSFSCZeroContextHead(LiteHOSV2RelationSlotHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.use_depth_prior and not self.pixel_self_attn_num_layers
        for key in (
            "coarse_contact_guidance",
            "coarse_reliability_aggregator",
            "prototype_extractor",
            "video_memory_adapter",
            "streaming_memory_adapter",
            "relation_topology_mixer",
            "decoder_attention",
            "hand_condition_object_refine",
            "mask_logit_refine",
        ):
            assert getattr(self, key, None) is None, key
        self.coarse_head = None
        self.refine = nn.Identity()
        self.use_global_query_bias = False

    def forward(
        self, feats, depth_prior=None, mask_pilot_targets=None, video_memory_state=None
    ):
        assert depth_prior is None and video_memory_state is None
        field = self._fuse_features(feats)
        queries = self.slot_queries.unsqueeze(0).expand(field.shape[0], -1, -1)
        dummy = field.new_zeros(field.shape[0], 1, *field.shape[-2:])
        masks, classes, tokens, _, _ = self._forward_slots(queries, field, dummy)
        if not self.return_intermediate:
            masks, classes = masks[-1:], classes[-1:]
        result = self._pack_outputs(masks, classes)
        result.update(
            coarse_logits=dummy,
            slot_tokens=tokens,
            relation_feat=field,
            pixel_feat=field,
            global_context=field.new_zeros(field.shape[0], self.feat_channels),
            coarse_prior_prob=None,
            coarse_uncertainty=None,
            coarse_reliability=None,
            memory_aux={},
        )
        return result
