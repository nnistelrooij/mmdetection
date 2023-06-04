import copy

import torch

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

    return loss

tversky_focal_loss_jit = torch.jit.script(
    tversky_focal_loss
)  # type: torch.jit.ScriptModule


def asymmetric_loss(
    inputs: torch.Tensor, targets, num_boxes, weights=None,
    gamma_plus=0, gamma_min=2, m=0.2, reduction='sum',
):
    if inputs.numel() == 0:
        loss = torch.zeros_like(inputs)
    else:
        prob = inputs.sigmoid()
        prob_m = torch.maximum(prob - m, torch.zeros_like(prob.flatten()[0]))

        L_plus = (1 - prob) ** gamma_plus * torch.log(prob)
        L_min = prob_m ** gamma_min * torch.log(1 - prob_m)

        loss = -targets * L_plus - (1 - targets) * L_min

    if weights is not None:
        loss = weights * loss

    if reduction == 'mean':
        return loss.mean(1).mean() / num_boxes
    elif reduction == 'sum':
        return loss.mean(1).sum() / num_boxes
    else:
        return loss.mean(1) / num_boxes
    

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
        hnm_samples: int=-1,
        use_fed_loss: bool=True,
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

        self.hnm_samples = hnm_samples
        self.use_fed_loss = use_fed_loss

    def set_federated_class_weights(
        self,
        class_counts,
        attribute_counts,
    ):
        self.fdi_weights = torch.tensor(class_counts).float() ** 0.5
        self.attribute_weights = torch.tensor(attribute_counts).float() ** 0.5

    def get_federated_classes(
        self,
        gt_classes_onehot,
        weight,
        num_classes: int,
        max_fed_classes: int=4,
    ):
        gt_classes_onehot = gt_classes_onehot.flatten(0, 1)
        gt_classes = torch.any(gt_classes_onehot, dim=0)

        unique_gt_classes = torch.nonzero(gt_classes)[:, 0]

        if unique_gt_classes.shape[0] >= max_fed_classes:
            fed_loss_classes = unique_gt_classes
        else:        
            prob = unique_gt_classes.new_ones(num_classes + 1).float()
            prob[:num_classes] = weight.float().clone()
            prob[-1] = 0
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, max_fed_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])

        fed_classes_mask = torch.zeros(num_classes + 1, dtype=torch.bool, device=fed_loss_classes.device)
        fed_classes_mask[fed_loss_classes] = True

        fed_classes = torch.nonzero(fed_classes_mask[:num_classes])[:, 0]

        return fed_classes

    def loss_labels(self, outputs, targets, indices, num_boxes, no_object: bool=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        num_classes = src_logits.shape[2]

        if no_object:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
        else:
            src_logits = torch.cat([p[J] for p, (J, _) in zip(src_logits, indices)])
            target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes_onehot = torch.zeros(src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1,
                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)

        if self.use_fed_loss:
            fed_classes = self.get_federated_classes(target_classes_onehot, self.fdi_weights, num_classes)
            weights = torch.zeros_like(src_logits)
            weights[..., fed_classes] = 1
        else:
            weights = None

        target_classes_onehot = target_classes_onehot[..., :-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, weights, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        return losses
    
    def loss_multilabels(self, outputs, targets, indices, num_boxes, reduction: str='sum', no_object: bool=True):
        """Multi-label classification loss (Asymmetric loss)
        targets dicts must contain the key "multilabels" containing a tensor of dim [nb_target_boxes. nb_attributes]
        """
        assert 'pred_multilabel_logits' in outputs

        src_logits = outputs['pred_multilabel_logits']
        num_classes = src_logits.shape[2]

        if no_object:
            idx = self._get_src_permutation_idx(indices)
            target_attributes_o = torch.cat([t["multilabels"][J] for t, (_, J) in zip(targets, indices)])
            target_attributes = torch.zeros(src_logits.shape, dtype=target_attributes_o.dtype, device=src_logits.device)
            target_attributes[idx] = target_attributes_o.reshape(-1, num_classes)
        else:
            src_logits = torch.cat([p[J] for p, (J, _) in zip(src_logits, indices)])

            target_attributes = torch.cat([t["multilabels"][J] for t, (_, J) in zip(targets, indices)])


        if self.use_fed_loss:
            target_attributes = torch.dstack((
                target_attributes, ~target_attributes.any(dim=-1),
            ))
            non_fed_classes = self.get_federated_classes(target_attributes, self.attribute_weights, num_classes)
            weights = torch.ones_like(src_logits)
            weights[..., non_fed_classes] = 0
            target_attributes = target_attributes[..., :-1]
        else:
            weights = None
        
        target_attributes = target_attributes.float()
        loss_asl = src_logits.shape[1] * asymmetric_loss(
            src_logits, target_attributes, num_boxes, weights, reduction=reduction,
        )        
        # loss_asl = sigmoid_focal_loss(
        #     src_logits, target_attributes, num_boxes, alpha=self.focal_alpha, gamma=2, reduction=reduction,
        # ) * src_logits.shape[1]

        losses = {'loss_bces': loss_asl}

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
            # "loss_tversky": tversky_focal_loss(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses
    
    @torch.no_grad()
    def hard_negative_mining(
        self, outputs, targets, indices, num_masks,
    ):
        losses = self.loss_multilabels(outputs, targets, indices, num_masks, reduction='none')
        losses = losses['loss_bces'].split([len(idxs[0]) for idxs in indices])

        out = []
        for target, (idxs_i, idxs_j), loss in zip(targets, indices, losses):
            keep_mask = torch.any(target['multilabels'], dim=-1).to(loss.device)
            keep_mask = keep_mask[idxs_j]

            neg_idxs = torch.nonzero(~keep_mask)[:, 0]

            _, idxs = loss[neg_idxs].topk(min(neg_idxs.shape[0], self.hnm_samples))
            keep_mask[neg_idxs[idxs]] = True
            keep_mask = keep_mask.cpu()

            out.append((idxs_i[keep_mask], idxs_j[keep_mask]))

        return out
    
    def remove_normal_teeth(self, targets, indices):
        out = []
        for target, (idxs_i, idxs_j) in zip(targets, indices):
            keep_mask = torch.any(target['multilabels'], dim=-1).to(idxs_i.device)

            if target['labels'].shape[0] == 0 or torch.all(keep_mask):
                out.append((idxs_i, idxs_j))
                continue

            keep_mask = keep_mask[idxs_j]
            out.append((idxs_i[keep_mask], idxs_j[keep_mask]))

        return out

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'multilabels': self.loss_multilabels,
            'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes_panoptic if self.panoptic_on else self.loss_boxes,
        }

        indices = self.remove_normal_teeth(targets, indices)
        # if self.hnm_samples > 0 and loss != 'labels':
        #     indices = self.hard_negative_mining(outputs, targets, indices, num_masks)

        assert loss in loss_map, f"do you really want to compute {loss} loss?"

        return loss_map[loss](outputs, targets, indices, num_masks)
