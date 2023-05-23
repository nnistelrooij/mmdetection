from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList

from projects.MaskDINO.maskdino import MaskDINO


@MODELS.register_module()
class MaskDINOMultilabel(MaskDINO):
    r"""Implementation of `Mask DINO: Towards A Unified Transformer-based
    Framework for Object Detection and Segmentation
    <https://arxiv.org/abs/2206.02777>`_."""

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        
        feats = self.extract_feat(batch_inputs)
        mask_results = self.panoptic_head.predict(feats, batch_data_samples)
        results_list = self.panoptic_fusion_head.predict(
            *mask_results, batch_data_samples, rescale=rescale,
        )
        
        results = self.add_pred_to_datasample(batch_data_samples, results_list)

        return results
