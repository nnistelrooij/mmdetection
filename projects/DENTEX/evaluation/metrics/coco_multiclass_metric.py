import json
from typing import Sequence

import torch

from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


@METRICS.register_module()
class CocoMulticlassMetric(CocoMetric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cat_ids = list(range(32))

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        converted_json_path = super().gt_to_coco_json(gt_dicts, outfile_prefix)

        with open(converted_json_path, 'r') as f:
            coco_dict = json.load(f)
            coco_dict['categories'] = [
                {'name': i, 'id': cat_id} for i, cat_id in enumerate(self.cat_ids)
            ]

        with open(converted_json_path, 'w') as f:
            json.dump(coco_dict, f)

        return converted_json_path

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                if isinstance(pred['masks'], torch.Tensor) and pred['masks'].ndim == 4:
                    masks = pred['masks'].amax(1)
                    result['masks'] = encode_mask_results(masks.detach().cpu().numpy())
                elif isinstance(pred['masks'], torch.Tensor) and pred['masks'].ndim == 3:
                    masks = (0 < pred['masks']) & (pred['masks'] < 8)
                    result['masks'] = encode_mask_results(masks.detach().cpu().numpy())
                else:
                    result['masks'] = pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            gt['anns'] = []
            for bbox, label, mask in zip(*[
                data_sample['gt_instances'][k] for k in ['bboxes', 'labels', 'masks']
            ]):
                rle = encode_mask_results([(0 < mask) & (mask < 8)])[0]
                instance = {
                    'bbox': bbox.tolist(),
                    'bbox_label': label.item(),
                    'mask': {
                        'size': rle['size'],
                        'counts': rle['counts'].decode(),
                    },
                }
                gt['anns'].append(instance)

            # add converted result to the results list
            self.results.append((gt, result))
