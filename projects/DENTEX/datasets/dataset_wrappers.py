from collections import defaultdict
import math

from mmengine.dataset import ClassBalancedDataset
import numpy as np

from mmdet.registry import DATASETS


@DATASETS.register_module()
class InstanceBalancedDataset(ClassBalancedDataset):

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
            
            attributes = np.stack(ann['bbox_multilabel'] for ann in instances)
            attributes = attributes[np.any(attributes, axis=-1)]

            for multilabel in attributes:
                for cat_id in multilabel.nonzero()[0]:
                    bbox_freq[cat_id] += 1
            for cat_id in np.any(attributes, axis=0).nonzero()[0]:
                image_freq[cat_id] += 1

            num_bboxes += attributes.sum()

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
                repeat_factors.append(1)
                continue
            
            attributes = np.stack(ann['bbox_multilabel'] for ann in instances)
            attributes = attributes[np.any(attributes, axis=-1)]
            attributes = np.any(attributes, axis=0)

            if not np.any(attributes):
                repeat_factors.append(1)
                continue

            repeat_factor = max(
                {category_repeat[cat_id]
                 for cat_id in attributes.nonzero()[0]})
            repeat_factors.append(repeat_factor)

        return repeat_factors
