_base_ = './maskdino-5scale_swin-l_8xb2-12e_opgs_multilabel.py'

test_dataloader = dict(dataset=dict(
    ann_file='/output/input.json',
    data_prefix=dict(img='/output'),
    data_root='/output',
))
test_evaluator = dict(ann_file='/output/input.json')
