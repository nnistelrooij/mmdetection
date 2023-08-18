# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List, Union

from mmengine.logging import MMLogger
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class CocoBinaryDataset(CocoDataset):
    """Dataset for COCO with instances of single class."""

    def __init__(
        self,
        label: str,
        *args, **kwargs,
    ):
        self.label = label
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        cat_ids = [0]*len(self.metainfo['classes'])
        for id, cat in self.coco.cats.items():
            category = cat['name']
            if category not in self.metainfo['classes']:
                continue

            cat_ids[self.metainfo['classes'].index(category)] = id

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
        for ann in ann_info:
            instance = {}

            if not self.test_mode and (
                'extra' not in ann or
                'attributes' not in ann['extra'] or
                self.label not in ann['extra']['attributes']
            ):
                continue

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
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

            instances.append(instance)
        data_info['instances'] = instances
        
        return data_info
    

    def filter_data(self) -> List[dict]:
        data_infos = super().filter_data()
        data_infos = [data_info for data_info in data_infos if data_info['instances']]

        return data_infos
