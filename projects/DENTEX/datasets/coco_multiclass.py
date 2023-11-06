from collections import defaultdict
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import pycocotools.mask as maskUtils

from mmengine.logging import MMLogger
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class CocoMulticlassDataset(CocoDataset):
    """Dataset for COCO with multi-class masks."""

    FDIs = [
        11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48,
    ]

    def __init__(
        self,
        metainfo: dict,
        strict: bool=True,
        decode_masks: bool=True,
        *args,
        **kwargs,
    ):
        metainfo['fdis'] = CocoMulticlassDataset.FDIs
        self.strict = strict
        self.decode_masks = decode_masks

        super().__init__(metainfo=metainfo, *args, **kwargs)

    def coco_to_rle(
        self,
        mask_ann: Union[list, dict],
        img_h: int,
        img_w: int,
    ):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            np.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        return rle

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        n_classes, n_attributes = len(self._metainfo['classes']), len(self._metainfo['attributes'])
        assert n_attributes <= 7, 'A maximum of 7 attributes are possible, due to uint8 masks'

        cats = np.array(self._metainfo['classes'] + self._metainfo['attributes'])
        _, inverse, counts = np.unique(cats, return_inverse=True, return_counts=True)
        assert np.all(
            cats[:n_classes][(counts == 2)[inverse[:n_classes]]] ==
            cats[n_classes:][(counts == 2)[inverse[n_classes:]]]
        ), 'Please specify duplicate class and attribute in the same order.'

        classes = np.concatenate((
            cats[:n_classes][(counts == 1)[inverse[:n_classes]]],
            cats[:n_classes][(counts == 2)[inverse[:n_classes]]],
        )).tolist()
        attributes = np.concatenate((
            cats[n_classes:][(counts == 2)[inverse[n_classes:]]],
            cats[n_classes:][(counts == 1)[inverse[n_classes:]]],
        )).tolist()
        assert all(c1 == c2 for c1, c2 in zip(cats.tolist(), classes + attributes)), (
            f'Please specify classes as {classes} and attributes as '
            f'{attributes} in configuration file for consistent class labels.'
        )
        n_unique_classes = (counts == 1)[inverse[:n_classes]].sum()


        data_info = {}
        data_info['img_path'] = Path(self.data_prefix['img']) / img_info['file_name']
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = None
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        fdi2anns = defaultdict(list)
        for ann in ann_info:
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if (
                inter_w * inter_h == 0 or
                ann['area'] <= 0 or w < 1 or h < 1
            ):
                if self.strict:
                    MMLogger.get_current_instance().error(
                        f'{img_info["file_name"]} has empty segmentation'
                    )
                continue

            cat_name = self.coco.cats[ann['category_id']]['name']
            if 'CROWN' in cat_name:
                ann['category_name'] = 'CROWN'
            elif 'METAL_FILLING' in cat_name:
                ann['category_name'] = 'CROWN_FILLING'
            else:
                ann['category_name'] = cat_name[:-3]
            fdi = int(cat_name[-2:])
            fdi2anns[fdi] = fdi2anns[fdi] + [ann]


        instances = []
        for fdi, anns in fdi2anns.items():
            instance = {
                'bbox': np.array([data_info['width'], data_info['height'], 0, 0]),
                'bbox_label': CocoMulticlassDataset.FDIs.index(fdi),
                'ignore_flag': anns[0].get('iscrowd', 0),
                'mask': np.zeros((img_info['height'], img_info['width']), dtype=np.uint8),
                'bbox_multilabel': np.zeros(n_unique_classes + n_attributes, dtype=int),
            }
            if self.decode_masks:
                instance['mask'] = np.tile(instance['mask'][None], (1 + n_attributes, 1, 1))
            else:
                instance['mask'] = [
                    maskUtils.encode(np.asfortranarray(instance['mask']))
                    for _ in range(1 + n_attributes)
                ]

            for ann in anns:
                cat_name = ann['category_name']

                if (
                    cat_name not in classes and
                    cat_name not in attributes
                ):
                    msg = (
                        f'Expected category name in {classes + attributes}, '
                        f'but found {ann["category_name"]}.'
                    )
                    if self.strict:
                        logging.warn(msg)
                    
                    continue

                if cat_name in classes:
                    instance['bbox'][:2] = np.minimum(
                        instance['bbox'][:2],
                        ann['bbox'][:2],
                    )
                    instance['bbox'][2:] = np.maximum(
                        instance['bbox'][2:],
                        np.array(ann['bbox'][:2]) + ann['bbox'][2:],
                    )
                    label = 0
                    label_idx = classes.index(cat_name)

                if cat_name in attributes:
                    label = 1 + attributes.index(cat_name)
                    label_idx = n_unique_classes + attributes.index(cat_name)

                rle = self.coco_to_rle(
                    ann['segmentation'], img_info['height'], img_info['width'],
                )

                if self.decode_masks:
                    index_arrays = np.nonzero(maskUtils.decode(rle))
                    index_arrays = (np.full(index_arrays[0].shape[0], label),) + index_arrays
                    instance['mask'][index_arrays] = 1
                else:
                    instance['mask'][label] = maskUtils.merge([instance['mask'][label], rle])

                instance['bbox_multilabel'][label_idx] = 1
          
            if self.decode_masks:
                instance['mask'] = sum(2 ** i * mask for i, mask in enumerate(instance['mask']))
                
            instances.append(instance)

        clean_instances = []
        for instance in instances:
            if np.any(instance['bbox_multilabel'][:len(classes)]):
                clean_instances.append(instance)
                continue
            
            if self.strict:
                MMLogger.get_current_instance().error(
                    f'{img_info["file_name"]} is missing tooth with diagnosis'
                )

        data_info['instances'] = clean_instances
        
        return data_info
