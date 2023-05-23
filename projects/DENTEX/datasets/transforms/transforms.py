import numpy as np

from mmdet.datasets.transforms import RandomFlip
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type


@TRANSFORMS.register_module()
class RandomOPGFlip(RandomFlip):

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, labels, and semantic segmentation."""
        labels = results['gt_bboxes_labels'] 
        results['gt_bboxes_labels'] = np.where((labels % 16) >= 8, labels - 8, labels + 8)

        super()._flip(results)
