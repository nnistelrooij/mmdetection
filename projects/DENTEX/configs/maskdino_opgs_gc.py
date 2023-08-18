_base_ = './maskdino-5scale_swin-l_8xb2-12e_opgs_multilabel.py'

test_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    dataset=dict(
        type='SingleMHADataset',
        ann_file='/opt/algorithm/test.csv',
        data_prefix=dict(img='/output'),
        data_root='/output',
        pipeline=_base_.val_pipeline[1:],
    ),
)
test_evaluator = dict(
    _delete_=True,
    type='DumpNumpyDetResults',
    out_file_path='detection.pkl',
    score_thr=0.1,
)

model = dict(test_cfg=dict(
    instance_postprocess_cfg=dict(max_per_image=36),
    max_per_image=36,
))

log_level = 'WARN'
