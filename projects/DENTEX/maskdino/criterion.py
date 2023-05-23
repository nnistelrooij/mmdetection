import copy

import torch

from projects.MaskDINO.maskdino.criterion import (
    SetCriterion,
    sigmoid_focal_loss,
)


class SetMultilabelCriterion(SetCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        attributes_weight=32.0,
        *args,
        **kwargs
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(*args, **kwargs)
        
        self.weight_dict.update({
            k.replace('ce', 'bces'): attributes_weight
            for k in self.weight_dict
            if 'loss_ce' in k
        })
        self.losses.append('multilabels')
        self.dn_losses.append('multilabels')
    
    def loss_multilabels(self, outputs, targets, indices, num_boxes):
        """Multi-label classification loss (Binary focal loss)
        targets dicts must contain the key "multilabels" containing a tensor of dim [nb_target_boxes. nb_attributes]
        """
        assert 'pred_multilabel_logits' in outputs

        src_logits = outputs['pred_multilabel_logits']
        src_logits = torch.cat([p[J] for p, (J, _) in zip(src_logits, indices)])

        target_attributes = torch.cat([t["multilabels"][J] for t, (_, J) in zip(targets, indices)])
        target_attributes = target_attributes.float()
        
        loss_ce = sigmoid_focal_loss(src_logits, target_attributes, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_bces': loss_ce}

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'multilabels': self.loss_multilabels,
            'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes_panoptic if self.panoptic_on else self.loss_boxes,
        }

        if loss == 'multilabels':
            indices = copy.deepcopy(indices)
            for i, (idxs_i, idxs_j) in enumerate(indices):
                keep_mask = torch.any(targets[i]['multilabels'], dim=-1).to(idxs_i.device)
                keep_mask = keep_mask[idxs_j]

                indices[i] = idxs_i[keep_mask], idxs_j[keep_mask]

        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)
