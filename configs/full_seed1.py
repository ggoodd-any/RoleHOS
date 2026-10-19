backbone_norm_cfg = dict(requires_grad=True, type='BN')
crop_size = (
    448,
    448,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        106.01075,
        95.40013,
        87.42854,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        448,
        448,
    ),
    std=[
        64.356636,
        60.888744,
        61.41911,
    ],
    type='SegDataPreProcessor')
data_root = 'data/egohos'
dataset_type = 'EgoHOSDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=10000,
        max_keep_ckpts=4,
        save_best=None,
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=False)
launcher = 'none'
load_from = None
log_level = 'INFO'
max_iters = 20000
model = dict(
    backbone=dict(
        arch='tiny',
        checkpoint='',
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        type='ConvNeXtDINOBackbone'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            106.01075,
            95.40013,
            87.42854,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            448,
            448,
        ),
        std=[
            64.356636,
            60.888744,
            61.41911,
        ],
        type='SegDataPreProcessor'),
    fast_encoder=dict(channels=32, in_channels=3, out_channels=64),
    fast_head=dict(
        align_corners=False,
        hidden_channels=64,
        in_channels=64,
        state_channels=256),
    gate_head=dict(hidden_channels=64, in_channels=64, max_age=5),
    semantic_output_mode='fixed5',
    slow_head=dict(
        align_corners=False,
        class_out_channels=6,
        feat_channels=256,
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_decoder_layers=4,
        num_heads=4,
        out_channels=5,
        return_intermediate=True,
        type='LiteHOSV2RelationSlotHead',
        use_coarse_contact_guidance=False,
        use_coarse_uncertainty_prior=True,
        use_hand_conditioned_object_refine=False,
        use_relation_topology_mixer=False),
    test_cfg=dict(align_corners=False, route='slow', threshold=0.5),
    train_cfg=dict(
        loss_bce_weight=5.0,
        loss_cls_no_object_weight=0.1,
        loss_cls_weight=2.0,
        loss_coarse_bce_weight=0.25,
        loss_coarse_dice_weight=0.25,
        loss_dice_weight=5.0,
        loss_lr_exclusivity_weight=0.0),
    type='LiteHOSV2RelationStateSegmentor')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0, norm_type=2),
    optimizer=dict(lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        bypass_duplicate=True, custom_keys=dict(backbone=dict(lr_mult=0.2))),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.1, type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=20000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
randomness = dict(deterministic=True, seed=1)
resume = False
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        448,
        448,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=20000, type='IterBasedTrainLoop', val_interval=10000)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='train/image', seg_map_path='train/label'),
        data_root='data/egohos',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                direction='horizontal',
                prob=0.5,
                swap_seg_labels=[
                    (
                        1,
                        2,
                    ),
                    (
                        3,
                        4,
                    ),
                ],
                type='RandomFlip'),
            dict(keep_ratio=False, scale=(
                448,
                448,
            ), type='Resize'),
            dict(type='PackSegInputs'),
        ],
        type='EgoHOSDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        direction='horizontal',
        prob=0.5,
        swap_seg_labels=[
            (
                1,
                2,
            ),
            (
                3,
                4,
            ),
        ],
        type='RandomFlip'),
    dict(keep_ratio=False, scale=(
        448,
        448,
    ), type='Resize'),
    dict(type='PackSegInputs'),
]
val_cfg = None
val_dataloader = None
val_evaluator = None
work_dir = 'outputs/full/seed1'
