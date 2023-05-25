import copy

import torch
import torch.nn.functional as F

from projects.MaskDINO.maskdino.criterion import (
    SetCriterion,
    calculate_uncertainty,
    dice_loss_jit,
    get_uncertain_point_coords_with_randomness,
    nested_tensor_from_tensor_list,
    point_sample,
    sigmoid_ce_loss_jit,
    sigmoid_focal_loss,
)


def tversky_focal_loss(
    inputs,
    targets,
    num_masks: float,
    alpha: float=0.7,
    beta: float=0.3,
    gamma: float=4 / 3,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = (inputs * targets).sum(-1)
    denominator = (
        numerator +
        alpha * ((1 - inputs) * targets).sum(-1) +
        beta * (inputs * (1 - targets)).sum(-1)
    )
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss ** (1 / gamma)
    loss = loss.sum() / num_masks

    assert not torch.isinf(loss)
    assert not torch.isnan(loss)

    return loss



tversky_focal_loss_jit = torch.jit.script(
    tversky_focal_loss
)  # type: torch.jit.ScriptModule


class SetMultilabelCriterion(SetCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        attributes_weight=8.0,
        tversky_weight=5.0,
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
        self.weight_dict.update({
            k.replace('dice', 'tversky'): tversky_weight
            for k in self.weight_dict
            if 'loss_dice' in k
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
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
            "loss_tversky": tversky_focal_loss(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
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
                # keep all teeth with at least one diagnosis
                keep_mask = torch.any(targets[i]['multilabels'], dim=-1).to(idxs_i.device)
                keep_mask = keep_mask[idxs_j]

                # keep at most two teeth without diagnosis
                remove_idxs = torch.nonzero(~keep_mask)[:, 0]
                if remove_idxs.shape[0] > 0:
                    keep_idxs = torch.multinomial(
                        input=torch.ones(remove_idxs.shape[0]),
                        num_samples=min(remove_idxs.shape[0], 2),
                    )
                    keep_mask[remove_idxs[keep_idxs]] = True
                
                # update indices
                indices[i] = idxs_i[keep_mask], idxs_j[keep_mask]

        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)
