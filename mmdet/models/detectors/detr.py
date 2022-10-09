# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmengine.model import xavier_init
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from ..layers import (DetrTransformerDecoder, DetrTransformerEncoder,
                      SinePositionalEncoding)
from .base_detr import TransformerDetector


@MODELS.register_module()
class DETR(TransformerDetector):
    """Implementation of `DETR: End-to-End Object Detection with Transformers.

    <https://arxiv.org/pdf/2005.12872>`_.

    Code is modified from the `official github repo
    <https://github.com/facebookresearch/detr>`_.
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding_cfg)
        self.encoder = DetrTransformerEncoder(**self.encoder_cfg)
        self.decoder = DetrTransformerDecoder(**self.decoder_cfg)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in [self.encoder, self.decoder]:
            for m in coder.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')

    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        Args:
            img_feats (Tuple[Tensor]): Features output from neck,
                with shape [bs, c, h, w].
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:  # TODO: Doc
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """

        feat = img_feats[-1]  # NOTE img_feats contains only one feature.
        batch_size, feat_dim, _, _ = feat.shape
        # construct binary masks which used for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]

        input_img_h, input_img_w = batch_input_shape
        masks = feat.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        masks = F.interpolate(
            masks.unsqueeze(1), size=feat.shape[-2:]).to(torch.bool).squeeze(1)
        # [batch_size, embed_dim, h, w]
        pos_embed = self.positional_encoding(masks)

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [h*w, bs, c]
        feat = feat.view(batch_size, feat_dim, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(2, 0, 1)
        # [bs, h, w] -> [bs, h*w]
        masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        encoder_inputs_dict = dict(feat=feat, masks=masks, pos_embed=pos_embed)
        decoder_inputs_dict = dict(masks=masks, pos_embed=pos_embed)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, masks: Tensor,
                        pos_embed: Tensor) -> Dict:
        """Forward with Transformer encoder.

        Args:  # TODO: Doc
            feat (Tensor): Sequential features, has shape (num_feat, bs, dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (num_feat, bs).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (num_feat, bs, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, query_pos=pos_embed, query_key_padding_mask=masks)
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat, dim).

        Returns:       # TODO: Doc
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory'.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """

        batch_size = memory.size(1)
        query_embed = self.query_embedding.weight
        # [num_query, dim] -> [num_query, bs, dim]
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        query = torch.zeros_like(query_embed)

        decoder_inputs_dict = dict(
            query_pos=query_embed, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        masks: Tensor, pos_embed: Tensor) -> Dict:
        """Overriding method 'forward_decoder' from 'base_detr.py' Forward with
        Transformer decoder.

        Args:# TODO: Doc
            query (Tensor): The queries of decoder inputs, has shape
                (num_query, bs, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (num_query, bs, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (num_feat, bs, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output.
        """
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=query,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_pos,
            key_padding_mask=masks)
        out_dec = out_dec.transpose(1, 2)
        head_inputs_dict = dict(hidden_states=out_dec)
        return head_inputs_dict
