from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs


@TRANSFORMS.register_module()
class PackMultilabelDetInputs(PackDetInputs):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_bboxes_multilabels': 'multilabels',
        'gt_masks': 'masks',
    }
