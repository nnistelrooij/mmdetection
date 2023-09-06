# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List, Union

from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class CocoMultilabelDataset(CocoDataset):
    """Dataset for COCO with multi-label instance attributes."""
    
    def _attrs2multilabel(
        self,
        attributes: List[str],
    ) -> List[int]:
        attr2label = {attr: i for i, attr in enumerate(self._metainfo['attributes'])}
        multilabel = [0]*len(attr2label)
        for attr in attributes:
            multilabel[attr2label[attr]] = 1

        return multilabel

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        cat_ids = [0]*len(self._metainfo['classes'])
        for id, cat in self.coco.cats.items():
            category = cat['name']
            if category not in self._metainfo['classes']:
                continue

            cat_ids[self._metainfo['classes'].index(category)] = id

        cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = Path(self.data_prefix['img']) / img_info['file_name']
        if self.data_prefix.get('seg', None):
            seg_map_path = Path(self.data_prefix['seg']) / (
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix
            )
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in cat2label:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            if 'extra' in ann and 'attributes' in ann['extra']:
                attributes = ann['extra']['attributes']
            else:
                attributes = []
            instance['bbox_multilabel'] = self._attrs2multilabel(attributes)

            instances.append(instance)
        data_info['instances'] = instances
        
        return data_info
