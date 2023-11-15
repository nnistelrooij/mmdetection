import copy
from collections import defaultdict
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Union

import numpy as np
import pycocotools.mask as maskUtils
from tqdm import tqdm

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
        num_workers: int=-1,
        *args,
        **kwargs,
    ):
        metainfo['fdis'] = CocoMulticlassDataset.FDIs
        self.strict = strict
        self.decode_masks = decode_masks
        self.num_workers = num_workers

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
    
    def _validate_classes_attributes(
        self,
    ):
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

        return classes, attributes, n_classes, n_unique_classes, n_attributes
    
    
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with self.file_client.get_local_path(self.ann_file) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        catname2id = {cat['name']: cat['id'] for cat in self.coco.cats.values()}
        self.cat2label = {
            catname2id[cat_name]: i
            for i, cat_name in enumerate(self.metainfo['classes'])
            if cat_name in catname2id
        }
        self.cat_ids = list(self.cat2label.keys())
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        (
            self.classes,
            self.attributes,
            self.n_classes,
            self.n_unique_classes,
            self.n_attributes,
        ) = self._validate_classes_attributes()

        img_ids = self.coco.get_img_ids()
        data_list = []
        if self.num_workers == 0:
            for img_id in tqdm(img_ids):
                data_info = self.parse_data_info(img_id)
                if data_info['instances']:
                    data_list.append(data_info)
        else:            
            chunk_size = 20
            img_ids_list = [img_ids[i:i+chunk_size] for i in range(0, len(img_ids), chunk_size)]
            num_workers = self.num_workers if self.num_workers > 0 else mp.cpu_count()
            with mp.Pool(num_workers) as p:
                for data_infos in tqdm(
                    p.imap_unordered(self.parse_data_batch, img_ids_list),
                    total=len(img_ids_list),
                ):
                    data_list.extend(data_infos)

        del self.coco

        return data_list

    def _fdi2anns(
        self,
        img_info: dict,
        ann_info: list[dict],
    ):
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

        return fdi2anns
    
    def _parse_instance(self, img_info, instance_info):
        fdi, anns = instance_info

        instance = {
            'bbox': np.array([img_info['width'], img_info['height'], 0, 0]),
            'bbox_label': CocoMulticlassDataset.FDIs.index(fdi),
            'ignore_flag': anns[0].get('iscrowd', 0),
            'mask': np.zeros((img_info['height'], img_info['width']), dtype=np.uint8),
            'bbox_multilabel': np.zeros(self.n_unique_classes + self.n_attributes, dtype=int),
        }
        if self.decode_masks:
            instance['mask'] = np.tile(instance['mask'][None], (1 + self.n_attributes, 1, 1))
        else:
            instance['mask'] = [
                maskUtils.encode(np.asfortranarray(instance['mask']))
                for _ in range(1 + self.n_attributes)
            ]
            
        for ann in anns:
            cat_name = ann['category_name']

            if (
                cat_name not in self.classes and
                cat_name not in self.attributes
            ):
                msg = (
                    f'Expected category name in {self.classes + self.attributes}, '
                    f'but found {ann["category_name"]}.'
                )
                if self.strict:
                    logging.warn(msg)
                
                continue

            if cat_name in self.classes:
                instance['bbox'][:2] = np.minimum(
                    instance['bbox'][:2],
                    ann['bbox'][:2],
                )
                instance['bbox'][2:] = np.maximum(
                    instance['bbox'][2:],
                    np.array(ann['bbox'][:2]) + ann['bbox'][2:],
                )
                label = 0
                label_idx = self.classes.index(cat_name)

            if cat_name in self.attributes:
                label = 1 + self.attributes.index(cat_name)
                label_idx = self.n_unique_classes + self.attributes.index(cat_name)

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

        return instance
    
    def parse_data_batch(self, img_ids: list):
        data_list = []
        for img_id in img_ids:
            data_info = self.parse_data_info(img_id)
            if data_info['instances']:
                data_list.append(data_info)

        return data_list

    def parse_data_info(self, img_id: int) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = self.coco.load_imgs([img_id])[0]
        img_info['img_id'] = img_id

        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)

        data_info = {}
        data_info['img_path'] = Path(self.data_prefix['img']) / img_info['file_name']
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = None
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        fdi2anns = self._fdi2anns(img_info, ann_info)

        instances = []
        for fdi, anns in fdi2anns.items():
            instance = self._parse_instance(img_info, (fdi, anns))
                
            instances.append(instance)

        clean_instances = []
        for instance in instances:
            if np.any(instance['bbox_multilabel'][:len(self.classes)]):
                clean_instances.append(instance)
                continue
            
            if self.strict:
                MMLogger.get_current_instance().error(
                    f'{img_info["file_name"]} is missing tooth with diagnosis'
                )

        data_info['instances'] = clean_instances
        
        return data_info
