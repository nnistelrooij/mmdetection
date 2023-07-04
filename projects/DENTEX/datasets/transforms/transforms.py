import numpy as np
import torch

from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import RandomFlip
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type


@TRANSFORMS.register_module()
class RandomOPGFlip(RandomFlip):

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, labels, and semantic segmentation."""
        labels = results['gt_bboxes_labels'] 
        results['gt_bboxes_labels'] = np.where((labels % 16) >= 8, labels - 8, labels + 8)

        super()._flip(results)


@TRANSFORMS.register_module()
class RandomToothFlip(BaseTransform):

    def __init__(
        self,
        prob: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.prob = prob

    @staticmethod
    def _determine_flipped_instance(
        results: dict,
        tooth_idx: int,
        middle_idx: int,
    ):
        middle_bbox = results['gt_bboxes'].tensor[middle_idx].int()

        #image cut at upper right bbox of central right incisor: half_flipped_image
        half_flipped_image = np.fliplr(results['img'][:, :middle_bbox[2]])

        #mask cut at upper right bbox of central right incisor: half_flipped_mask
        tooth_mask = results['gt_masks'].masks[tooth_idx]
        half_flipped_mask = np.fliplr(tooth_mask[:, :middle_bbox[2]])

        #half flipped instance
        half_flipped_instance = half_flipped_image * half_flipped_mask[..., np.newaxis]

        #padding of half flipped instance: flipped_instance
        width_difference = results['width'] - 2 * half_flipped_instance.shape[1]
        if width_difference < 0:
            flipped_instance = np.pad(
                half_flipped_instance,
                pad_width=(
                    (0, 0),
                    (half_flipped_instance.shape[1], 0),
                    (0, 0),
                ),
            )
            flipped_instance = flipped_instance[:, :width_difference]
        else:
            flipped_instance = np.pad(
                half_flipped_instance,
                pad_width=(
                    (0, 0),
                    (half_flipped_instance.shape[1], width_difference),
                    (0, 0),
                ),
            )

        return flipped_instance        

    @staticmethod
    def _add_flipped_instance(
        results: dict,
        tooth_idx: int,
        flipped_instance,
    ):
        image_with_flipped_instance = np.where(flipped_instance == 0, results['img'], flipped_instance)
        results['img'] = image_with_flipped_instance

        results['gt_ignore_flags'] = np.concatenate((
            results['gt_ignore_flags'], [results['gt_ignore_flags'][tooth_idx]],
        ))

        tooth_bbox = results['gt_bboxes'].tensor[tooth_idx].clone()
        results['gt_bboxes'][tooth_idx].flip_(results['img_shape'], 'horizontal')
        results['gt_bboxes'].tensor = torch.cat((
            results['gt_bboxes'].tensor,            
            results['gt_bboxes'][tooth_idx].tensor,
        ))
        results['gt_bboxes'].tensor[tooth_idx] = tooth_bbox

        tooth_label = results['gt_bboxes_labels'][tooth_idx]
        opposite_label = tooth_label + 8 if tooth_label < 16 else tooth_label - 8
        results['gt_bboxes_labels'] = np.concatenate((
            results['gt_bboxes_labels'], [opposite_label],
        ))
        results['gt_bboxes_multilabels'] = np.concatenate((
            results['gt_bboxes_multilabels'], [results['gt_bboxes_multilabels'][tooth_idx]],
        ))
        
        flipped_instance_mask = flipped_instance[..., 0] > 0
        results['gt_masks'].masks = np.concatenate((
            results['gt_masks'].masks, [flipped_instance_mask],
        ))

    @staticmethod
    def _remove_occluded_instance(
        results: dict,
        tooth_idx: int,
    ):
        # remove tooth that is now superimposed by the flipped tooth
        tooth_label = results['gt_bboxes_labels'][tooth_idx]
        opposite_label = tooth_label + 8 if tooth_label < 16 else tooth_label - 8
        opposite_idxs = np.nonzero(results['gt_bboxes_labels'] == opposite_label)[0]
        if opposite_idxs.shape[0] == 1:
            return
        
        opposite_idx = opposite_idxs[0]
        results['gt_ignore_flags'] = np.concatenate((
            results['gt_ignore_flags'][:opposite_idx],
            results['gt_ignore_flags'][opposite_idx + 1:],
        ))
        results['gt_bboxes'].tensor = torch.cat((
            results['gt_bboxes'].tensor[:opposite_idx],
            results['gt_bboxes'].tensor[opposite_idx + 1:],
        ))
        results['gt_bboxes_labels'] = np.concatenate((
            results['gt_bboxes_labels'][:opposite_idx],
            results['gt_bboxes_labels'][opposite_idx + 1:],
        ))
        results['gt_bboxes_multilabels'] = np.concatenate((
            results['gt_bboxes_multilabels'][:opposite_idx],
            results['gt_bboxes_multilabels'][opposite_idx + 1:],
        ))
        results['gt_masks'].masks = np.concatenate((
            results['gt_masks'].masks[:opposite_idx],
            results['gt_masks'].masks[opposite_idx + 1:],
        ))

    def _flip_tooth_right_to_left(self, results: dict) -> None:
        """
        Flip a right tooth (left side of PR) and superimpose it on a left tooth.
        """
        labels = results['gt_bboxes_labels']
        right_teeth = labels[(labels < 8) | (labels >= 24)]

        if right_teeth.shape[0] == 0:
            return
         
        tooth_label = np.random.choice(right_teeth)
        tooth_idx = np.nonzero(labels == tooth_label)[0][0]

        middle = 0 if tooth_label < 16 else 24
        if middle not in labels:
            return
        
        middle_idx = np.nonzero(labels == middle)[0][0]
        flipped_instance = self._determine_flipped_instance(results, tooth_idx, middle_idx)        
        self._add_flipped_instance(results, tooth_idx, flipped_instance)
        self._remove_occluded_instance(results, tooth_idx)

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results
        
        self._flip_tooth_right_to_left(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'

        return repr_str
