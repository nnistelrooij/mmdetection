_base_ = '../../MaskDINO/configs/maskdino_r50_8xb2-lsj-50e_coco-panoptic.py'

custom_imports = dict(
    imports=[
        'projects.DENTEX.datasets',
        'projects.DENTEX.datasets.dataset_wrappers',
        'projects.DENTEX.datasets.transforms.loading',
        'projects.DENTEX.datasets.transforms.formatting',
        'projects.DENTEX.datasets.transforms.transforms',
        'projects.DENTEX.evaluation',
        'projects.DENTEX.maskdino',
        'projects.DENTEX.visualization',
        'projects.DENTEX.hooks',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='MaskDINOMultilabel',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=None,
    ),
    panoptic_head=dict(
        encoder=dict(
            in_channels=[192, 384, 768, 1536],
            transformer_in_features=['res2', 'res3', 'res4', 'res5'],
            num_feature_levels=4,
            total_num_feature_levels=5,
        ),
        decoder=dict(
            total_num_feature_levels=5,
        )
    )
)

load_from = 'checkpoints/maskdino_swin_mmdet.pth'
