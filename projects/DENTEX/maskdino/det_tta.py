# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmcv.ops import batched_nms
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
import numpy as np
from scipy import ndimage
import torch

from mmdet.models import DetTTAModel
from mmdet.structures import DetDataSample


@MODELS.register_module()
class DENTEXTTAModel(DetTTAModel):
    """Merge augmented detection results, only bboxes corresponding score under
    flipping and multi-scale resizing can be processed now.

    Examples:
        >>> tta_model = dict(
        >>>     type='DetTTAModel',
        >>>     tta_cfg=dict(nms=dict(
        >>>                     type='nms',
        >>>                     iou_threshold=0.5),
        >>>                     max_per_img=100))
        >>>
        >>> tta_pipeline = [
        >>>     dict(type='LoadImageFromFile',
        >>>          file_client_args=dict(backend='disk')),
        >>>     dict(
        >>>         type='TestTimeAug',
        >>>         transforms=[[
        >>>             dict(type='Resize',
        >>>                  scale=(1333, 800),
        >>>                  keep_ratio=True),
        >>>         ], [
        >>>             dict(type='RandomFlip', prob=1.),
        >>>             dict(type='RandomFlip', prob=0.)
        >>>         ], [
        >>>             dict(
        >>>                 type='PackDetInputs',
        >>>                 meta_keys=('img_id', 'img_path', 'ori_shape',
        >>>                         'img_shape', 'scale_factor', 'flip',
        >>>                         'flip_direction'))
        >>>         ]])]
    """

    def merge_preds(self, data_samples_list: List[List[DetDataSample]]):
        """Merge batch predictions of enhanced data.

        Args:
            data_samples_list (List[List[DetDataSample]]): List of predictions
                of all enhanced data. The outer list indicates images, and the
                inner list corresponds to the different views of one image.
                Each element of the inner list is a ``DetDataSample``.
        Returns:
            List[DetDataSample]: Merged batch prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples    

    def denoise_masks(self, masks, rel_count: float=5.0):
        structure = ndimage.generate_binary_structure(3, 2)
        structure[[0, 2]] = False
        labels, max_label = ndimage.label(masks, structure)

        maxs = np.concatenate(([0], labels.max(axis=(1, 2))))
        max_label = 0
        for i, label in enumerate(maxs):
            if label == 0:
                maxs[i] = max_label
            else:
                max_label = label
        mask_idxs = np.concatenate([
            np.full(max2 - max1, i) if i == 0 or max1 > 0 and max2 > 0 else []
            for i, (max1, max2) in enumerate(zip(maxs[:-1], maxs[1:]))
        ])

        counts = np.bincount(labels.flatten())[1:]
        labels = np.maximum(0, labels - maxs[:-1].reshape(-1, 1, 1))
        for i in range(masks.shape[0]):
            mask_counts = counts[mask_idxs == i]
            mask_counts = np.concatenate(([10_000], mask_counts))
            if mask_counts.shape[0] == 1:
                continue

            keep_mask = mask_counts >= (mask_counts[1:].max() / rel_count)
            masks[i][~keep_mask[labels[i]]] = False

        return masks

    def _merge_single_sample(
            self, data_samples: List[DetDataSample]) -> DetDataSample:
        """Merge predictions which come form the different views of one image
        to one prediction.

        Args:
            data_samples_list (List[DetDataSample]): List of predictions
            of enhanced data which come form one image.
        Returns:
            List[DetDataSample]: Merged prediction.
        """
        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        aug_multilabels = []
        aug_multilogits = []
        aug_logits = []
        aug_masks = []
        img_metas = []
        convert_to_cpu = True
        for data_sample in data_samples:
            masks = data_sample.pred_instances.masks
            # masks = torch.from_numpy(self.denoise_masks(masks.cpu().numpy()))
            x1 = (masks.sum(dim=1) > 0).int().argmax(dim=-1)
            y1 = (masks.sum(dim=2) > 0).int().argmax(dim=-1)
            x2 = masks.shape[2] - (torch.flip(masks, dims=(2,)).sum(dim=1) > 0).int().argmax(dim=-1)
            y2 = masks.shape[1] - (torch.flip(masks, dims=(1,)).sum(dim=2) > 0).int().argmax(dim=-1)
            bboxes = torch.column_stack((x1, y1, x2, y2))

            if bboxes.device == 'cpu':
                convert_to_cpu = True

            aug_bboxes.append(bboxes.float())
            aug_scores.append(data_sample.pred_instances.scores)
            aug_multilabels.append(data_sample.pred_instances.multilabels)
            aug_multilogits.append(data_sample.pred_instances.multilogits)
            img_metas.append(data_sample.metainfo)

            if not data_sample.flip:
                aug_labels.append(data_sample.pred_instances.labels)
                aug_logits.append(data_sample.pred_instances.logits)
                aug_masks.append(data_sample.pred_instances.masks)
                continue

            labels = data_sample.pred_instances.labels
            flipped_labels = torch.where((labels % 16) >= 8, labels - 8, labels + 8)
            aug_labels.append(flipped_labels)

            logits = data_sample.pred_instances.logits
            flipped_logits = torch.column_stack((
                logits[:, 8:16], logits[:, :8],
                logits[:, 24:], logits[:, 16:24],
            ))
            aug_logits.append(flipped_logits)

            masks = data_sample.pred_instances.masks
            flipped_masks = torch.flip(masks, dims=(-1,))
            aug_masks.append(flipped_masks)

        if convert_to_cpu:
            aug_bboxes = [bboxes.cpu() for bboxes in aug_bboxes]
            aug_scores = [scores.cpu() for scores in aug_scores]
            aug_labels = [labels.cpu() for labels in aug_labels]
            aug_multilabels = [labels.cpu() for labels in aug_multilabels]
            aug_multilogits = [labels.cpu() for labels in aug_multilogits]
            aug_logits = [logits.cpu() for logits in aug_logits]
            aug_masks = [masks.cpu() for masks in aug_masks]

        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0)
        merged_multilabels = torch.cat(aug_multilabels, dim=0)
        merged_multilogits = torch.cat(aug_multilogits, dim=0)
        merged_logits = torch.cat(aug_logits, dim=0)
        merged_masks = torch.cat(aug_masks, dim=0)

        if merged_bboxes.numel() == 0:
            det_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
            return [
                (det_bboxes, merged_labels),
            ]

        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.tta_cfg.nms)

        det_bboxes = det_bboxes[:self.tta_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.tta_cfg.max_per_img]
        det_multilabels = merged_multilabels[keep_idxs][:self.tta_cfg.max_per_img]
        det_multilogits = merged_multilogits[keep_idxs][:self.tta_cfg.max_per_img]
        det_logits = merged_logits[keep_idxs][:self.tta_cfg.max_per_img]
        det_masks = merged_masks[keep_idxs][:self.tta_cfg.max_per_img]

        results = InstanceData()
        _det_bboxes = det_bboxes.clone()
        results.bboxes = _det_bboxes[:, :-1]
        results.scores = _det_bboxes[:, -1]
        results.labels = det_labels
        results.multilabels = det_multilabels
        results.multiscores = det_multilogits.sigmoid()
        results.logits = det_logits
        results.masks = det_masks
        det_results = data_samples[0]
        det_results.pred_instances = results
        
        return det_results
