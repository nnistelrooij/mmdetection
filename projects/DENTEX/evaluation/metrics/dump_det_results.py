# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
import warnings
from typing import Sequence

from mmengine.evaluator import DumpResults
from mmengine.evaluator.metric import _to_cpu
from mmengine.logging import MMLogger
import numpy as np
from scipy import ndimage
import torch
from torch_scatter import scatter_max, scatter_min

from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


@METRICS.register_module()
class DumpNumpyDetResults(DumpResults):
    """Dump model predictions to a pickle file for offline evaluation.

    Different from `DumpResults` in MMEngine, it compresses instance
    segmentation masks into RLE format.

    Args:
        out_file_path (str): Path of the dumped file. Must end with '.pkl'
            or '.pickle'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def __init__(
        self,
        score_thr: float=0.1,
        filter_wrong_arch: bool=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.score_thr=score_thr
        self.filter_wrong_arch = filter_wrong_arch

    def means_stds_maxs(
        self,
        upper_coords,
        lower_coords,
        is_upper,
    ):
        upper_y_mean = upper_coords[:, 1].float().mean()
        upper_y_std = upper_coords[:, 1].float().std()
        upper_y_maxs, _ = scatter_max(
            src=upper_coords[:, 1].float(),
            index=upper_coords[:, 0],
            dim_size=is_upper.sum(),
        )
        upper_y_stds = (upper_y_maxs - upper_y_mean) / upper_y_std

        lower_y_mean = lower_coords[:, 1].float().mean()
        lower_y_std = lower_coords[:, 1].float().std()
        lower_y_mins, _ = scatter_min(
            src=lower_coords[:, 1].float(),
            index=lower_coords[:, 0],
            dim_size=is_upper.shape[0] - is_upper.sum(),
        )
        lower_y_stds = (lower_y_mean - lower_y_mins) / lower_y_std

        return upper_y_stds, lower_y_stds
        

    def wrong_arch_idxs(
        self,
        pred,
        keep,
        std_thr: float=3.3,  # expected probability < 0.001
    ):
        is_upper = pred['labels'][keep] < 16

        upper_coords = torch.nonzero(pred['masks'][keep][is_upper])
        lower_coords = torch.nonzero(pred['masks'][keep][~is_upper])

        upper_y_stds, lower_y_stds = self.means_stds_maxs(
            upper_coords, lower_coords, is_upper,
        )

        keep_ = torch.zeros(keep.sum(), dtype=torch.bool, device=keep.device)
        keep_[is_upper] = upper_y_stds < std_thr
        keep_[~is_upper] = lower_y_stds < std_thr

        keep[keep.nonzero()[~keep_, 0]] = False

        max_upper_std = upper_y_stds.max() if upper_y_stds.numel() else 0
        max_lower_std = lower_y_stds.max() if lower_y_stds.numel() else 0
        
        return keep, torch.all(keep_), max(max_upper_std, max_lower_std)


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            data_sample.pop('ignored_instances', None)
            data_sample.pop('gt_panoptic_seg', None)

            if 'gt_instances' in data_sample:
                gt = data_sample['gt_instances']
                # encode mask to RLE
                if 'masks' in gt:
                    gt['masks'] = encode_mask_results(gt['masks'])

            if 'pred_instances' in data_sample:
                pred = data_sample['pred_instances']

                keep1, same, max_std = pred['scores'] >= self.score_thr, True, 0.0
                if self.filter_wrong_arch:
                    keep, same, max_std = self.wrong_arch_idxs(pred, keep1.clone())
                else:
                    keep = keep1

                if not same:
                    MMLogger.get_current_instance().warn(
                        f'Scan {data_sample["img_path"].name} removed bbox from wrong arch!',
                    )

                pred['scores'] = pred['scores'][keep]
                pred['bboxes'] = pred['bboxes'][keep]
                if 'logits' in pred:
                    pred['logits'] = pred['logits'][keep]
                pred['labels'] = pred['labels'][keep]
                pred['masks'] = encode_mask_results(pred['masks'][keep].numpy())

                path = data_sample['img_path']
                path = path.name if isinstance(path, Path) else path
                MMLogger.get_current_instance().warn(
                    f'Scan {path} Teeth: {keep.sum()} STD: {max_std}',
                )
                    
            if 'pred_panoptic_seg' in data_sample:
                warnings.warn(
                    'Panoptic segmentation map will not be compressed. '
                    'The dumped file will be extremely large! '
                    'Suggest using `CocoPanopticMetric` to save the coco '
                    'format json and segmentation png files directly.')
        self.results.extend(data_samples)


@METRICS.register_module()
class DumpMulticlassDetResults(DumpResults):

    def __init__(
        self,
        score_thr: float=0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.score_thr = score_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            data_sample.pop('ignored_instances', None)
            data_sample.pop('gt_panoptic_seg', None)

            if 'gt_instances' in data_sample:
                gt = data_sample['gt_instances']
                if gt['masks'].masks.dtype == np.bool_:
                    gt['masks'] = [
                        encode_mask_results(mask)
                        for mask in gt['masks'].masks
                    ]
                elif gt['masks'].masks.dtype == np.uint8:
                    gt['masks'] = [
                        encode_mask_results([mask == i for i in range(1, 9)])
                        for mask in gt['masks'].masks
                    ]
            
            if 'pred_instances' in data_sample:
                pred = data_sample['pred_instances']

                keep, max_std = pred['scores'] >= self.score_thr, 0.0

                pred['scores'] = pred['scores'][keep]
                pred['bboxes'] = pred['bboxes'][keep]
                pred['logits'] = pred['logits'][keep]
                pred['labels'] = pred['labels'][keep]
                if pred['masks'].dtype == torch.bool:
                    pred['masks'] = [
                        encode_mask_results(mask)
                        for mask in pred['masks'][keep].numpy()
                    ]
                elif pred['masks'].dtype == torch.int64:
                    pred['masks'] = [
                        encode_mask_results([mask == i for i in range(1, 9)])
                        for mask in pred['masks'][keep].numpy()
                    ]

                MMLogger.get_current_instance().warn(
                    f'Scan {data_sample["img_path"].name} Teeth: {keep.sum()} STD: {max_std}',
                )

        self.results.extend(data_samples)
