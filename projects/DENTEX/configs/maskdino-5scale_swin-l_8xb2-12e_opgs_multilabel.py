_base_ = './maskdino-5scale_swin-l_8xb2-12e_coco_multilabel.py'

data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
data_prefix = dict(img=data_root + 'images')
classes = [
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]
attributes = ['Caries', 'Deep Caries', 'Impacted', 'Periapical Lesion']

ann_prefix = data_root + 'releases/fdi-checked/other_formats/coco/'
train_dataloader = dict(dataset=dict(dataset=dict(
    ann_file=ann_prefix + 'train4.json',
    data_prefix=data_prefix,
    data_root=data_root,
    metainfo=dict(classes=classes, attributes=attributes),
)))

val_dataloader = dict(dataset=dict(
    ann_file=ann_prefix + 'val4.json',
    data_prefix=data_prefix,
    data_root=data_root,
    metainfo=dict(classes=classes, attributes=attributes),
))
val_evaluator = dict(
    _delete_=True,
    type='CocoOPGMetric',
    ann_file=ann_prefix + 'val4.json',
    # ann_file=data_root + 'annotations/instances_val2017_onesample_139.json',  # TODO: delete before merging
    metric=['bbox', 'segm'],
)

test_dataloader = dict(dataset=dict(
    ann_file=ann_prefix + 'val4.json',
    data_prefix=data_prefix,
    data_root=data_root,
    metainfo=dict(classes=classes, attributes=attributes),
))
test_evaluator = dict(
    _delete_=True,
    type='CocoOPGMetric',
    ann_file=ann_prefix + 'val4.json',
    # ann_file=data_root + 'annotations/instances_val2017_onesample_139.json',  # TODO: delete before merging
    metric=['bbox', 'segm'],
)

model = dict(
    panoptic_head=dict(num_things_classes=len(classes), num_stuff_classes=0),
    panoptic_fusion_head=dict(num_things_classes=len(classes), num_stuff_classes=0),
)
work_dir = 'work_dirs/opgs_fold4'
