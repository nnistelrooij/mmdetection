_base_ = './maskdino_opgs_mha.py'

test_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    dataset=dict(
        type='PNGsDataset',
        data_prefix=dict(img='/input'),
        data_root='/input',
    ),
)
