import warnings

import torch
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType

from projects.MaskDINO.maskdino.maskdino_head import (
    bbox_xyxy_to_cxcywh,
    MaskDINOHead,
)
from projects.DENTEX.maskdino.criterion import SetMultilabelCriterion
from projects.DENTEX.maskdino.maskdino_multilabel_decoder_layers import MaskDINOMultilabelDecoder


@MODELS.register_module()
class MaskDINOMultilabelHead(MaskDINOHead):

    def __init__(
        self,
        decoder: OptConfigType,
        train_cfg: OptConfigType = None,
        *args, **kwargs,
    ):
        num_attributes = decoder.pop('num_attributes')
        enable_multilabel = decoder.pop('enable_multilabel')
        super().__init__(decoder=decoder, train_cfg=train_cfg, *args, **kwargs)

        decoder['num_attributes'] = num_attributes
        decoder['enable_multilabel'] = enable_multilabel
        self.predictor = MaskDINOMultilabelDecoder(**decoder)
        self.criterion = SetMultilabelCriterion(**train_cfg)

    def loss(self, feats, batch_data_samples):
        if torch.any(torch.isnan(self.predictor.attributes_embed.weight)):
            k = 3


        targets = self.prepare_targets(batch_data_samples)
        outputs, mask_dict = self(feats, mask=None, targets=targets)  # TODO: deal with key_padding_masks ?
        # bipartite matching-based loss
        losses = self.criterion(outputs, targets, mask_dict)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        
        for _, x_ in losses.items():
            if torch.any(torch.isnan(x_) | torch.isinf(x_)):
                k = 3

        return losses

    def predict(self, feats, batch_data_samples):
        outputs, mask_dict = self(feats)
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_box_results = outputs["pred_boxes"]
        mask_attrs_results = outputs["pred_multilabel_logits"]

        # upsample masks
        batch_input_shape = batch_data_samples[0].metainfo['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(batch_input_shape[0], batch_input_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results, mask_box_results, mask_attrs_results

    def prepare_targets(self, batch_data_samples):
        # h_pad, w_pad = images.tensor.shape[-2:]  # TODO: Here is confusing
        h_pad, w_pad = batch_data_samples[0].batch_input_shape  # TODO: make a check
        new_targets = []
        for data_sample in batch_data_samples:
            # pad gt
            device = data_sample.gt_instances.bboxes.device
            h, w = data_sample.img_shape[:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=device)

            gt_masks = torch.from_numpy(data_sample.gt_instances.masks.masks).bool().to(device)
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": data_sample.gt_instances.labels,
                    "masks": padded_masks,
                    "boxes": bbox_xyxy_to_cxcywh(data_sample.gt_instances.bboxes) / image_size_xyxy,
                    "multilabels": data_sample.gt_instances.multilabels,
                }
            )

            warnings.warn(  # TODO: align the lsj pipeline
                'The lsj for MaskDINO and Mask2Former has not been fully aligned '
                'with COCOPanopticNewBaselineDatasetMapper in original repo')

        return new_targets
