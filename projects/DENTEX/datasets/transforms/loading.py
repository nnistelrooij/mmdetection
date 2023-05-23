from mmdet.datasets.transforms import LoadAnnotations
from mmdet.registry import TRANSFORMS
import numpy as np


@TRANSFORMS.register_module()
class LoadMultilabelAnnotations(LoadAnnotations):

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels, gt_bboxes_multilabels = [], []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
            gt_bboxes_multilabels.append(instance['bbox_multilabel'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)
        results['gt_bboxes_multilabels'] = np.array(
            gt_bboxes_multilabels, dtype=np.int64)
