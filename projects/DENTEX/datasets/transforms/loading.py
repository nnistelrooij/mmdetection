import pycocotools.mask as maskUtils
import numpy as np
from typing import Dict, Optional

from mmcv.transforms import LoadImageFromFile
from mmdet.datasets.transforms import LoadAnnotations
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks


@TRANSFORMS.register_module()
class LoadUInt16ImageFromFile(LoadImageFromFile):

    def __init__(
        self, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs, color_type='unchanged')

    def transform(self, results: dict) -> Optional[Dict]:
        results = super().transform(results)

        results['img'] = np.tile(results['img'][..., None], (1, 1, 3)) / 257

        return results
    

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
        

@TRANSFORMS.register_module()
class LoadMulticlassAnnotations(LoadMultilabelAnnotations):

    def __init__(
        self,
        merge_layers: bool,
        *args, **kwargs,
    ):
        self.merge_layers = merge_layers

        super().__init__(*args, **kwargs)

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        if 'instances' not in results or not results['instances']:
            return results

        if isinstance(results['instances'][0]['mask'], list) and not self.merge_layers:
            for instance in results['instances']:
                mask = 0
                for i, rle in enumerate(instance['mask']):
                    mask += 2 ** i * maskUtils.decode(rle)
                instance['mask'] = mask
        elif isinstance(results['instances'][0]['mask'], list) and self.merge_layers:
            for instance in results['instances']:
                mask = 0
                for i, rle in enumerate(instance['mask'], 1):
                    mask = np.maximum(mask, i * maskUtils.decode(rle))
                instance['mask'] = mask

        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)
        gt_masks = BitmapMasks(gt_masks, h, w)
        results['gt_masks'] = gt_masks
