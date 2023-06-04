from typing import List, Optional

from mmengine.structures import InstanceData
import torch
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.structures.mask import mask2bbox
from mmdet.utils import OptConfigType, OptMultiConfig
from mmdet.utils.memory import AvoidCUDAOOM

from projects.MaskDINO.maskdino import MaskDINOFusionHead


@MODELS.register_module()
class MaskDINOMultilabelFusionHead(MaskDINOFusionHead):
    """MaskDINO fusion head which postprocesses results for panoptic
    segmentation, instance segmentation and semantic segmentation."""

    def __init__(self,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 test_cfg: OptConfigType = None,
                 loss_panoptic: OptConfigType = None,  # TODO: not used ?
                 init_cfg: OptMultiConfig = None,
                 enable_multilabel: bool=False,
                 **kwargs):
        super().__init__(
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            test_cfg=test_cfg,
            loss_panoptic=loss_panoptic,
            init_cfg=init_cfg,
            **kwargs)
        
        self.enable_multilabel = enable_multilabel

    def predict(self,
                mask_cls_results: Tensor,
                mask_pred_results: Tensor,
                mask_box_results: Tensor,
                mask_attributes_results: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> List[dict]:
        """ segment without test-time aumengtation.
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]

        results = []
        for mask_cls_result, mask_pred_result, mask_box_result, mask_attributes_result, meta in zip(
                mask_cls_results, mask_pred_results, mask_box_results, mask_attributes_results, batch_img_metas):
            # shape of image before pipeline
            ori_height, ori_width = meta['ori_shape'][:2]
            # shape of image after pipeline and before padding divisibly
            img_height, img_width = meta['img_shape'][:2]
            # shape of image after padding divisibly
            batch_input_height, batch_input_width = meta['batch_input_shape'][:2]

            # remove padding
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()
            
            mask_box_result = bbox_cxcywh_to_xyxy(mask_box_result)
            height_factor = batch_input_height / img_height * ori_height
            width_factor = batch_input_width / img_width * ori_width
            mask_box_result[:, 0::2] = mask_box_result[:, 0::2] * width_factor
            mask_box_result[:, 1::2] = mask_box_result[:, 1::2] * height_factor
            result['ins_results'] = self.instance_postprocess(
                mask_cls_result, mask_pred_result, mask_box_result, mask_attributes_result)

            results.append(result)

        return results

    @AvoidCUDAOOM.retry_if_cuda_oom
    def instance_postprocess(
        self,
        mask_cls: Tensor,
        mask_pred: Tensor,
        mask_box: Optional[Tensor],
        mask_attributes: Tensor,
    ) -> InstanceData:
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.
            mask_box (Tensor): TODO

        Returns:
            :obj:`InstanceData`: Instance segmentation results.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # TODO: merge into MaskFormerFusionHead
        max_per_image = self.test_cfg.get('max_per_image', 100)
        focus_on_box = self.test_cfg.get('focus_on_box', False)

        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = mask_cls.sigmoid()  # TODO: modify MaskFormerFusionHead to add an arg use_sigmoid  # TODO: difference
        scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False)  # TODO：why ？
        
        if self.enable_multilabel:
            scores = mask_attributes.sigmoid()
            scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False)  # TODO：why ？

            attributes = torch.arange(scores.shape[1], device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)  # TODO：why ？
            attributes_per_image = attributes[top_indices]
        else:
            # shape (num_queries * num_class)
            labels = torch.arange(self.num_classes, device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)  # TODO：why ？
            labels_per_image = labels[top_indices]

        query_indices = top_indices // scores.shape[1]  # TODO：why ？
        mask_pred = mask_pred[query_indices]
        mask_box = mask_box[query_indices] if mask_box is not None else None  # TODO: difference

        if self.enable_multilabel:
            mask_cls = mask_cls[query_indices]
            labels_per_image = mask_cls.argmax(-1)

            _, idx, counts = torch.unique(query_indices, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
            unique_indices = ind_sorted[cum_sum]

            mask_pred = mask_pred[unique_indices]
            mask_box = mask_box[unique_indices]
            scores_per_image = scores_per_image[unique_indices]
            labels_per_image = labels_per_image[unique_indices]

            attributes_per_image = torch.zeros(scores.shape[1], device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)  # TODO：why ？
            attributes_per_image[top_indices] = 1
            attributes_per_image = attributes_per_image.reshape(-1, scores.shape[1])
            attributes_per_image = attributes_per_image[query_indices]
            attributes_per_image = attributes_per_image[unique_indices]

        # extract things  # TODO： if self.panoptic_on ?
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        if self.enable_multilabel:
            attributes_per_image = attributes_per_image[is_thing]
        mask_pred = mask_pred[is_thing]
        mask_box = mask_box[is_thing] if mask_box is not None else None  # TODO: difference

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)  # TODO：why ？
        det_scores = scores_per_image * mask_scores_per_image  # TODO：why ？
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask_box if mask_box is not None else mask2bbox(mask_pred_binary)  # TODO: difference

        results = InstanceData()
        results.bboxes = bboxes
        results.labels = labels_per_image
        results.scores = det_scores if not focus_on_box else 1.0
        results.masks = mask_pred_binary
        if self.enable_multilabel:
            results.multilabels = attributes_per_image
        else:
            results.multilabels = mask_attributes[query_indices] > 0
        return results
