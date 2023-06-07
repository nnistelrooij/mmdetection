# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import numpy as np
import torch
from mmengine.fileio import load


def convert(src, dst):
    if src.endswith('pth'):
        src_model = torch.load(src)
    else:
        src_model = load(src)

    dst_state_dict = OrderedDict()
    for k, v in src_model['model'].items():
        key_name_split = k.split('.')
        if 'backbone.patch_embed.proj' in k:
            name = f'backbone.patch_embed.projection.{key_name_split[-1]}'
        elif 'backbone' in k:
            name = k.replace('layers', 'stages')
            if 'attn' in k:
                name = name.replace('attn', 'attn.w_msa')
            elif 'mlp.fc1' in k:
                name = name.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                name = name.replace('mlp.fc2', 'ffn.layers.1')
        elif 'sem_seg_head' in k:
            name = k.replace('sem_seg_head', 'panoptic_head')
            if 'adapter' in k or 'layer_1' in k:
                if 'norm' in k:
                    name = '.'.join(['panoptic_head', *key_name_split[1:-2], 'gn', key_name_split[-1]])
                else:
                    name = '.'.join(['panoptic_head', *key_name_split[1:-1], 'conv', key_name_split[-1]])
            elif 'decoder.bbox_embed' in k:
                name = '.'.join(['panoptic_head.predictor', *key_name_split[3:]])
            elif (
                'input_proj' in k or
                'mask_features' in k or
                'level_embed' in k or
                ('output_proj' in k and 'transformer' in k) or
                ('self_attn' in k and 'transformer' in k)
            ):
                name = name.replace('predictor', 'pixel_decoder')
            elif (
                'transformer.encoder.layers' in k and
                'self_attn' not in k
            ):
                name = name.replace('predictor', 'pixel_decoder')
                if 'norm' in k:
                    layer = int(key_name_split[-2][-1]) - 1
                    name = '.'.join(['panoptic_head.pixel_decoder', *key_name_split[2:6], f'norms.{layer}', key_name_split[-1]])
                else:
                    layer = int(key_name_split[-2][-1]) - 1
                    layer = '0.0' if layer == 0 else 1
                    name = '.'.join(['panoptic_head.pixel_decoder', *key_name_split[2:6], f'ffn.layers.{layer}', key_name_split[-1]])
            else:
                name = '.'.join(['panoptic_head.predictor', *key_name_split[2:]])
        elif 'criterion' in k:
            name = 'panoptic_head.criterion.empty_weight'
        else:
            # some base parameters such as beta will not convert
            print(f'{k} is not converted!!')
            continue

        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError(
                'Unsupported type found in checkpoint! {}: {}'.format(
                    k, type(v)))
        if not isinstance(v, torch.Tensor):
            dst_state_dict[name] = torch.from_numpy(v)
        else:
            dst_state_dict[name] = v

    mmdet_model = dict(state_dict=dst_state_dict, meta=dict())
    torch.save(mmdet_model, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
