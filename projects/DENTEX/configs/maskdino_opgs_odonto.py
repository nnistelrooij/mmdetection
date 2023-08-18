_base_ = './maskdino_r50_coco_multilabel.py'
# _base_ = './maskdino_swin-l_coco_multilabel.py'

data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
export = 'curated-odonto'
split = 'odonto_enumeration'
fold = 2
data_prefix = dict(img=data_root + 'images')
ann_prefix = data_root + f'releases/{export}/other_formats/coco/'

classes = [
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]
attributes = ['Caries', 'Deep Caries', 'Impacted', 'Periapical Lesion']


train_dataloader = dict(
    dataset=dict(dataset=dict(dataset=dict(
        ann_file=ann_prefix + f'train_{split}_{fold}.json',
        data_prefix=data_prefix,
        data_root=data_root,
        metainfo=dict(classes=classes, attributes=attributes),
    ))),
)

val_dataloader = dict(dataset=dict(
    ann_file=ann_prefix + f'val_{split}_{fold}.json',
    data_prefix=data_prefix,
    data_root=data_root,
    metainfo=dict(classes=classes, attributes=attributes),
))
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=ann_prefix + f'val_{split}_{fold}.json',
    # ann_file=data_root + 'annotations/instances_val2017_onesample_139.json',  # TODO: delete before merging
    metric=['bbox', 'segm'],
)

test_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    dataset=dict(
        # ann_file=ann_prefix + f'test_{split}.json',
        ann_file=ann_prefix + 'val_dentex_diagnosis_0.json',
        data_prefix=data_prefix,
        data_root=data_root,
        metainfo=dict(classes=classes, attributes=attributes),
    ),
)
test_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=ann_prefix + f'test_{split}.json',
    # ann_file=data_root + 'annotations/instances_val2017_onesample_139.json',  # TODO: delete before merging
    metric=['bbox', 'segm'],
)
test_evaluator = dict(
    _delete_=True,
    type='DumpNumpyDetResults',
    out_file_path='detection.pkl',
)

model = dict(
    train_cfg=dict(num_classes=len(classes), hnm_samples=2, use_fed_loss=False),
    test_cfg=dict(
        instance_postprocess_cfg=dict(max_per_image=100),
        max_per_image=100,
    ),
    panoptic_head=dict(
        num_things_classes=len(classes),
        num_stuff_classes=0,
        decoder=dict(
            num_classes=len(classes),
            num_attributes=len(attributes),
            enable_multilabel=False,
        ),
    ),
    panoptic_fusion_head=dict(
        num_things_classes=len(classes),
        num_stuff_classes=0,
        enable_multilabel=False,
    ),
)

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

work_dir = f'work_dirs/opgs_fold_{split}_{fold}_{_base_.model.backbone.type}'
resume = True
