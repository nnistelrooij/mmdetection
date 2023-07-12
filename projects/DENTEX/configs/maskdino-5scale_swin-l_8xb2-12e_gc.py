_base_ = './maskdino-5scale_swin-l_8xb2-12e_opgs_multilabel.py'

test_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    dataset=dict(
        ann_file='/output/input.json',
        data_prefix=dict(img='/output'),
        data_root='/output',
    ),
)
test_evaluator = dict(
    _delete_=True,
    type='DumpNumpyDetResults',
    out_file_path='detection.pkl',
)

# test_evaluator = []
log_level = 'WARN'
