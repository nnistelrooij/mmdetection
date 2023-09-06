_base_ = './maskdino_r50_coco_multilabel.py'
# _base_ = './maskdino_swin-l_coco_multilabel.py'

data_root = '/home/mkaailab/Documents/Promaton/'
data_root = '/home/mkaailab/.darwin/datasets/mucoaid/promaton/'
split = 'promaton_diagnoses_extra'
fold = 0
data_prefix = dict(img=data_root + 'images')
work_dir = f'work_dirs/opgs_fold_{split}_{fold}/'
phase = 'test'

classes = [
    'TOOTH', 'IMPACTED', 'ROOT_REMNANTS', 'CROWN', 'DENTAL_IMPLANT',
]
attributes = [
    'CROWN', 'DENTAL_IMPLANT', 'CARIES', 'PERIAPICAL_RADIOLUCENT',
    'CROWN_FILLING', 'ROOT_CANAL_FILLING',
]
num_classes = 32, len(classes)
num_attributes = 1 + len(attributes)
num_upper_masks = 1 + len(set(classes) & set(attributes))


train_dataloader = dict(
    batch_size=2,
    # num_workers=0,
    # persistent_workers=False,
    dataset=dict(dataset=dict(dataset=dict(
        type='CocoMulticlassDataset',
        ann_file=data_root + f'train_{split}_{fold}.json',
        # ann_file=data_root + f'test_{split}.json',
        data_prefix=data_prefix,
        data_root=data_root,
        metainfo=dict(classes=classes, attributes=attributes),
        pipeline=[
            dict(type='LoadUInt16ImageFromFile'),
            dict(type='LoadMulticlassAnnotations', with_bbox=True, with_mask=True),
        ],
    ))),
)

val_pipeline = [
    dict(
        type='LoadUInt16ImageFromFile',
    ),
    dict(
        type='Resize',
        scale=(1333, 800),
        keep_ratio=True),
    dict(type='LoadMulticlassAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackMultilabelDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]
val_dataloader = dict(dataset=dict(
    type='CocoMulticlassDataset',
    ann_file=data_root + f'val_{split}_{fold}.json',
    data_prefix=data_prefix,
    data_root=data_root,
    metainfo=dict(classes=classes, attributes=attributes),
    pipeline=val_pipeline,
))
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + f'val_{split}_{fold}.json',
    metric=['bbox', 'segm'],
)
val_dataloader = None
val_evaluator = None
val_cfg = None

test_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    dataset=dict(
        type='CocoMulticlassDataset',
        ann_file=data_root + f'val_{split}_{fold}.json',
        # ann_file=data_root + f'test_{split}.json',
        # ann_file=data_root + f'train_{split}_{fold}.json',
        # ann_file=ann_prefix + 'val_dentex_diagnosis_0.json',
        data_prefix=data_prefix,
        data_root=data_root,
        metainfo=dict(classes=classes, attributes=attributes),
        pipeline=val_pipeline,
    ),
)
test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + f'val_{split}_{fold}.json',
        # ann_file=data_root + f'test_{split}.json',
        # ann_file=data_root + f'train_{split}_{fold}.json',
        # ann_file=ann_prefix + 'val_dentex_diagnosis_0.json',
        metric=['bbox', 'segm'],
        class_agnostic=True,
        prefix='class_agnostic',
    ),
    dict(
        type='CocoMetric',
        ann_file=data_root + f'val_{split}_{fold}.json',
        # ann_file=data_root + f'test_{split}.json',
        # ann_file=ann_prefix + 'val_dentex_diagnosis_0.json',
        metric=['bbox', 'segm'],
        class_agnostic=False,
        prefix='fdi_label',
    ),
    dict(
        type='DumpNumpyDetResults',
        out_file_path=(
            'detection_odo_4.pkl'
            if 'dentex' in test_dataloader['dataset']['ann_file'] else
            work_dir + 'detection.pkl'            
        ),
    ),
]
test_evaluator = []

custom_hooks = []
model = dict(
    train_cfg=dict(num_classes=num_classes[0], hnm_samples=2, use_fed_loss=False),
    test_cfg=dict(
        instance_postprocess_cfg=dict(max_per_image=100),
        max_per_image=100,
    ),
    panoptic_head=dict(
        num_things_classes=num_classes[0],
        num_stuff_classes=0,
        decoder=dict(
            num_classes=num_classes,
            num_attributes=num_attributes,
            enable_multilabel=False,
            enable_multiclass=True,
            num_queries=100 if phase == 'train' else 300,
        ),
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_classes[0],
        num_stuff_classes=0,
        enable_multilabel=False,
        enable_multiclass=True,
        num_upper_masks=num_upper_masks,
    ),
)

max_epochs = 100
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
        milestones=[max_epochs - 20, max_epochs - 5],
        gamma=0.1,
    )
]

tta_model = dict(
    type='DENTEXTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)

default_hooks = dict(checkpoint=dict(
    save_best='coco/segm_mAP',
))

visualizer = dict(type='MulticlassDetLocalVisualizer')
