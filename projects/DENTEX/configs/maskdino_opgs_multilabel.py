_base_ = './maskdino_r50_coco_multilabel.py'

data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
export = 'curated-odonto'
fold = '_dentex_diagnoses'
multilabel = True
data_prefix = dict(img=data_root + 'images')
ann_prefix = data_root + f'releases/{export}/other_formats/coco/'
work_dir = f'work_dirs/opgs_fold{fold}/'

classes = [
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]
attributes = ['Caries', 'Deep Caries', 'Impacted', 'Periapical Lesion']


train_dataloader = dict(
    batch_size=2,
    dataset=dict(dataset=dict(dataset=dict(
        ann_file=ann_prefix + f'train{fold}.json',
        data_prefix=data_prefix,
        data_root=data_root,
        metainfo=dict(classes=classes, attributes=attributes),
    ))),
)

val_dataloader = dict(dataset=dict(
    ann_file=ann_prefix + f'val{fold}.json',
    data_prefix=data_prefix,
    data_root=data_root,
    metainfo=dict(classes=classes, attributes=attributes),
))
val_evaluator = dict(
    _delete_=True,
    type='CocoDENTEXMetric',
    ann_file=ann_prefix + f'val{fold}.json',
    # ann_file=data_root + 'annotations/instances_val2017_onesample_139.json',  # TODO: delete before merging
    metric=['bbox', 'segm'],
)

test_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    dataset=dict(
        # ann_file=ann_prefix + f'val{fold}.json',
        ann_file=ann_prefix + f'val{fold}.json',
        data_prefix=data_prefix,
        data_root=data_root,
        metainfo=dict(classes=classes, attributes=attributes),
    ),
)
test_evaluator = dict(
    _delete_=True,
    type='CocoDENTEXMetric',
    ann_file=ann_prefix + f'val{fold}.json',
    # ann_file=data_root + 'annotations/instances_val2017_onesample_139.json',  # TODO: delete before merging
    metric=['bbox', 'segm'],
)
# test_evaluator = [
#     dict(
#         type='DumpNumpyDetResults',
#         out_file_path=work_dir + 'detection.pkl',
#     ),
#     dict(
#         type='CocoMetric',
#         ann_file=ann_prefix + f'val{fold}.json',
#         # ann_file=data_root + 'annotations/instances_val2017_onesample_139.json',  # TODO: delete before merging
#         metric=['bbox', 'segm'],
#     )
# ]

max_per_image = 100
model = dict(
    train_cfg=dict(num_classes=len(classes), hnm_samples=2, use_fed_loss=False),
    test_cfg=dict(
        instance_postprocess_cfg=dict(max_per_image=max_per_image),
        max_per_image=max_per_image,
    ),
    panoptic_head=dict(
        num_things_classes=len(classes),
        num_stuff_classes=0,
        decoder=dict(
            num_classes=len(classes),
            num_attributes=len(attributes),
            enable_multilabel=multilabel,
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

load_from = 'checkpoints/maskdino_mmdet.pth'
load_from = 'work_dirs/opgs_fold_odonto_dentex_enumeration/epoch_50.pth'


train_cfg = dict(max_epochs=24)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1,
    )
]

default_hooks = dict(checkpoint=dict(
    save_best='coco/segm_split-diagnoses=False',
))
