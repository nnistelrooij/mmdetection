from collections import defaultdict
import math

from mmengine.dataset import ClassBalancedDataset
import numpy as np

from mmdet.registry import DATASETS


@DATASETS.register_module()
class InstanceBalancedDataset(ClassBalancedDataset):

    def __init__(
        self,
        key: str='bbox_multilabel',
        *args,
        **kwargs,
    ):
        self.key = key
        
        super().__init__(*args, **kwargs)

    def _get_repeat_factors(self, dataset, repeat_thr):
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        image_freq, bbox_freq = defaultdict(int), defaultdict(int)
        num_images = len(dataset)
        num_bboxes = 0
        for idx in range(num_images):
            instances = self.dataset.get_data_info(idx)['instances']
            if not instances:
                continue

            attributes = np.stack([ann[self.key] for ann in instances])
            if attributes.ndim == 1:
                temp = np.zeros((attributes.shape[0], attributes.max() + 1))
                temp[np.arange(attributes.shape[0]), attributes] = 1
                attributes = temp.astype(int)

            attributes = attributes[np.any(attributes, axis=-1)]

            cat_ids = set()
            for multilabel in attributes:
                cat_id = sum(2**i * label for i, label in enumerate(multilabel))
                bbox_freq[cat_id] = bbox_freq[cat_id] + 1
                num_bboxes += 1

                cat_ids.add(cat_id)

            for cat_id in cat_ids:
                image_freq[cat_id] = image_freq[cat_id] + 1

        for k, v in image_freq.items():
            image_freq[k] = v / num_images
            bbox_freq[k] = bbox_freq[k] / num_bboxes

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / math.sqrt(img_freq * bbox_freq[cat_id])))
            for cat_id, img_freq in image_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            instances = self.dataset.get_data_info(idx)['instances']
            if not instances:
                repeat_factors.append(1.0)
                continue
            
            attributes = np.stack([ann[self.key] for ann in instances])
            if attributes.ndim == 1:
                temp = np.zeros((attributes.shape[0], attributes.max() + 1))
                temp[np.arange(attributes.shape[0]), attributes] = 1
                attributes = temp.astype(int)
            attributes = attributes[np.any(attributes, axis=-1)]

            if not np.any(attributes):
                repeat_factors.append(1.0)
                continue
            
            cat_ids = set()
            for multilabel in attributes:
                cat_id = sum(2**i * label for i, label in enumerate(multilabel))
                cat_ids.add(cat_id)

            repeat_factor = max(category_repeat[cat_id] for cat_id in cat_ids)
            repeat_factors.append(repeat_factor)

        return repeat_factors
