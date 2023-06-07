_base_ = '../../MaskDINO/configs/maskdino_r50_8xb2-lsj-50e_coco-panoptic.py'
# _base_ = './maskdino-5scale_swin-l_8xb2-12e_coco.py'

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

dataset_type = 'CocoMultilabelDataset'

train_pipeline = [
    # dict(type='Mosaic', img_scale=(1024, 1024), pad_val=114.0),
    dict(type='RandomOPGFlip', prob=0.5),
    dict(type='RandomToothFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[{
                'type':
                'RandomChoiceResize',
                'scales': [(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
            [{
                'type': 'RandomChoiceResize',
                'scales': [(400, 1333), (500, 1333),
                            (600, 1333)],
                'keep_ratio': True
            }, {
                'type': 'RandomCrop',
                'crop_type': 'absolute_range',
                'crop_size': (384, 600),
                'allow_negative_crop': True
            }, {
                'type':
                'RandomChoiceResize',
                'scales':
                [(480, 1333), (512, 1333), (544, 1333),
                    (576, 1333), (608, 1333), (640, 1333),
                    (672, 1333), (704, 1333), (736, 1333),
                    (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
        ]),
    dict(type='PackMultilabelDetInputs'),
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True, 
        type='InstanceBalancedDataset',
        oversample_thr=0.1,
        dataset=dict(
            type='MultiImageMixDataset',
            dataset=dict(
                type=dataset_type,
                filter_cfg=dict(filter_empty_gt=False),
                serialize_data=False,
                pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='LoadMultilabelAnnotations', with_bbox=True, with_mask=True),
                ],
            ),
            pipeline=train_pipeline,
        ),
    ),
)

val_pipeline=[
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        imdecode_backend='pillow'),
    dict(
        type='Resize',
        scale=(1333, 800),
        keep_ratio=True,
        backend='pillow'),
    dict(
        type='LoadMultilabelAnnotations',
        file_client_args=dict(backend='disk'),
        with_bbox=True, with_mask=True),
    dict(
        type='PackMultilabelDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

val_dataloader = dict(dataset=dict(type=dataset_type, pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(type=dataset_type, pipeline=val_pipeline))

max_epochs = 200
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[int(max_epochs * 8 / 9), int(max_epochs * 26 / 27)],
        gamma=0.1,
    )
]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = [dict(type='ClassCountsHook')]
model = dict(
    type='MaskDINOMultilabel',
    test_cfg=dict(panoptic_on=False, instance_on=True, max_per_image=100),
    panoptic_head=dict(type='MaskDINOMultilabelHead'),
    panoptic_fusion_head=dict(type='MaskDINOMultilabelFusionHead'),
)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        by_epoch=True,
        save_best='coco/segm_exclude=False',
        rule='greater',
    ),
    visualization=dict(
        draw=True,
        interval=20,
    ),
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='SparseTensorboardVisBackend'),
    ],
)

# load_from = 'checkpoints/maskdino_mmdet.pth'
# load_from = 'work_dirs/opgs_fold_enumeration_0/epoch_50.pth'
