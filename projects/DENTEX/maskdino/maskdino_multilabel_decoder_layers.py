import torch
from torch import nn

from projects.MaskDINO.maskdino.maskdino_decoder_layers import (
    bbox_xyxy_to_cxcywh,
    gen_encoder_output_proposals,
    inverse_sigmoid,
    MaskDINODecoder,
)


class MaskDINOMultilabelDecoder(MaskDINODecoder):

    def __init__(
        self,
        num_attributes: int,
        enable_multilabel: bool,
        *args,
        **kwargs,
    ):
        """
        Args:
            num_attribute_classes: number of multi-label classes
        """
        super().__init__(*args, **kwargs)

        if self.mask_classification:
            self.attributes_embed = nn.Linear(self.hidden_dim, num_attributes)
        self.enable_multilabel = enable_multilabel

    def dn_post_process(
        self,
        outputs_class,
        outputs_coord,
        mask_dict,
        outputs_mask,
        outputs_attributes,
    ):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]

        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]

        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]

        outputs_known_attributes = outputs_attributes[:, :, :mask_dict['pad_size'], :]
        outputs_attributes = outputs_attributes[:, :, mask_dict['pad_size']:, :]

        out = {
            'pred_logits': output_known_class[-1],
            'pred_boxes': output_known_coord[-1],
            'pred_masks': output_known_mask[-1],
            'pred_multilabel_logits': outputs_known_attributes[-1],
        }

        out['aux_outputs'] = self._set_aux_loss(
            output_known_class,
            output_known_mask,
            output_known_coord,
            outputs_known_attributes,
        )
        mask_dict['output_known_lbs_bboxes'] = out

        return outputs_class, outputs_coord, outputs_mask, outputs_attributes

    def forward(self, x, mask_features, masks, targets=None):
        """
        :param x: input, a list of multi-scale feature
        :param mask_features: is the per-pixel embeddings with resolution 1/4 of the original image,
        obtained by fusing backbone encoder encoded features. This is used to produce binary masks.
        :param masks: mask in the original image
        :param targets: used for denoising training
        """
        assert len(x) == self.num_feature_levels
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        predictions_attributes = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            # enc_outputs_coord_unselected = self._bbox_embed(
            #     output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            enc_outputs_coord_unselected = self.bbox_embed[-1](
                output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            refpoint_embed = refpoint_embed_undetach.detach()

            tgt_undetach = torch.gather(output_memory, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # unsigmoid

            outputs_class, outputs_mask, outputs_attributes = self.forward_prediction_heads(
                tgt_undetach.transpose(0, 1), mask_features,
            )
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs=dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_multilabel_logits'] = outputs_attributes
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask

            if self.initialize_box_type != 'no':
                # convert masks into boxes to better initialize box in the decoder
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                    raise NotImplementedError()  # TODO: learn to write this
                    # refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.cuda()  # TODO: make a dummy BitMask?
                elif self.initialize_box_type == 'mask2box':  # faster conversion
                    raise NotImplementedError()  # TODO: learn to write this
                    # refpoint_embed = mask2bbox(flaten_mask > 0).cuda()
                else:
                    assert NotImplementedError
                refpoint_embed = bbox_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h], dtype=torch.float).cuda()
                refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, x[0].shape[0])
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)

        # direct prediction from the matching and denoising part in the begining
        if self.initial_pred:
            outputs_class, outputs_mask, outputs_attributes = self.forward_prediction_heads(
                tgt.transpose(0, 1), mask_features, self.training,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_attributes.append(outputs_attributes)
        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed=torch.cat([input_query_bbox,refpoint_embed],dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask,
            bbox_embed=self.bbox_embed
        )
        for i, output in enumerate(hs):
            outputs_class, outputs_mask, outputs_attributes = self.forward_prediction_heads(
                output.transpose(0, 1), mask_features, self.training or (i == len(hs)-1),
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_attributes.append(outputs_attributes)

        # iteratively box prediction
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)
        if mask_dict is not None:
            predictions_mask=torch.stack(predictions_mask)
            predictions_class=torch.stack(predictions_class)
            predictions_attributes = torch.stack(predictions_attributes)

            predictions = self.dn_post_process(
                predictions_class,
                out_boxes,
                mask_dict,
                predictions_mask,
                predictions_attributes,
            )
            (
                predictions_class,
                out_boxes,
                predictions_mask,
                predictions_attributes
            ) = predictions

            predictions_class = list(predictions_class)
            predictions_mask = list(predictions_mask)
            predictions_attributes = list(predictions_attributes)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] += 0.0*self.label_enc.weight.sum()

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_multilabel_logits': predictions_attributes[-1],
            'pred_boxes':out_boxes[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
                out_boxes,
                predictions_attributes if self.mask_classification else None,
            )
        }
        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        return out, mask_dict

    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        outputs_class, outputs_mask = super().forward_prediction_heads(
            output, mask_features, pred_mask,
        )

        decoder_output = self.decoder.norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        
        if not self.enable_multilabel:
            with torch.no_grad():
                outputs_attributes = self.attributes_embed(decoder_output)
        else:
            outputs_attributes = self.attributes_embed(decoder_output)

        
        return outputs_class, outputs_mask, outputs_attributes

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class,
        outputs_seg_masks,
        out_boxes=None,
        out_attributes=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_multilabel_logits": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_attributes[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c, "pred_multilabel_logits": d}
                for a, b, c, d in zip(
                    outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1], out_attributes[:-1],
                )
            ]
