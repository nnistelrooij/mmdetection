from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
from mmengine.visualization import Visualizer
import numpy as np
import pycocotools.mask as maskUtils
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)
import torch

from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


def draw_roc_curve(
    results,
    prefix,
    steps=0,
    specificity_range=[0.0, 1.0],
    thr_range=[0.0, 1.0],
    thr_method: str='f1',
) -> Dict[str, float]:
    if 'pred_score' not in results[0]:
        return None

    pred_scores = torch.stack([r['pred_score'][1] for r in results]).cpu().numpy()
    gt_labels = torch.cat([r['gt_label'] for r in results]).cpu().numpy()

    # determine statistics at thresholds
    fpr, tpr, thrs = roc_curve(gt_labels, pred_scores)
    sensitivity, specificity = np.stack((tpr, 1 - fpr))

    # at least specificity of 90% and reasonable thresholds
    mask = (
        (specificity >= specificity_range[0]) &
        (specificity <= specificity_range[1]) &
        (thrs >= thr_range[0]) &
        (thrs <= thr_range[1])
    )
    if np.any(mask):
        fpr, tpr, thrs = fpr[mask], tpr[mask], thrs[mask]
        sensitivity, specificity = sensitivity[mask], specificity[mask]

    # compute criteria to determine optimal threshold
    if thr_method == 'iu':
        auc = roc_auc_score(gt_labels, pred_scores)
        criteria = np.abs(sensitivity - auc) + np.abs(specificity - auc)
    elif thr_method == 'er':
        criteria = np.sqrt((1 - sensitivity) ** 2 + (1 - specificity) ** 2)
    elif thr_method == 'f1':
        criteria = np.zeros_like(thrs)
        for i, thr in enumerate(thrs):
            criteria[i] = -f1_score(gt_labels, pred_scores >= thr)
    else:
        raise TypeError('thr_method must be one of ["iu", "er", "f1"].')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    RocCurveDisplay.from_predictions(gt_labels, pred_scores, ax=ax)
    ax.scatter([fpr[criteria.argmin()]], [tpr[criteria.argmin()]], s=48, c='r')
    ax.grid()

    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)

    vis = Visualizer.get_current_instance()
    vis.add_image(f'{prefix}/roc_curve', image, step=steps)
    plt.close(fig)

    try:
        return {
            'auc': roc_auc_score(gt_labels, pred_scores),
            'optimal_thr': thrs[criteria.argmin()],
        }
    except ValueError:
        return None


def draw_confusion_matrix(
    results,
    thr,
    prefix,
    steps=0,
):
    if thr is not None:
        pred_labels = torch.stack([r['pred_score'][1] >= thr for r in results]).cpu().numpy()
    else:
        pred_labels = torch.cat([r['pred_label'] for r in results]).cpu().numpy()
    gt_labels = torch.cat([r['gt_label'] for r in results]).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(gt_labels, pred_labels, ax=ax)

    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)

    vis = Visualizer.get_current_instance()
    vis.add_image(f'{prefix}/confusion_matrix', image, step=steps)
    plt.close(fig)

    return {
        'precision': precision_score(gt_labels, pred_labels),
        'recall': recall_score(gt_labels, pred_labels),
        'f1-score': f1_score(gt_labels, pred_labels),
    }


@METRICS.register_module()
class SingleLabelMetric(BaseMetric):
    default_prefix: Optional[str] = 'single-label'

    def __init__(self,
                 label_idx: int=0,
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 items: Sequence[str] = ('precision', 'recall', 'f1-score'),
                 average: Optional[str] = 'macro',
                 num_classes: Optional[int] = None,
                 iou_type: str='segm',
                 score_thr: float=0.3,
                 remove_irrelevant: bool=False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:

        super().__init__(collect_device=collect_device, prefix=prefix)

        self.idx = label_idx
        self.steps = 0

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs, )
        else:
            self.thrs = tuple(thrs)

        for item in items:
            assert item in ['precision', 'recall', 'f1-score', 'support'], \
                f'The metric {item} is not supported by `SingleLabelMetric`,' \
                ' please specify from "precision", "recall", "f1-score" and ' \
                '"support".'
        self.items = tuple(items)
        self.average = average
        self.num_classes = num_classes
        assert iou_type in ['bbox', 'segm'], 'iou_type must be "bbox" or "segm"'
        self.iou_type = iou_type
        self.score_thr = score_thr
        self.remove_irrelevant = remove_irrelevant

    def match_instances(self, pred_instances, gt_instances, iou_thr: float=0.5):
        if self.iou_type == 'bbox':
            preds = pred_instances['bboxes'].cpu().numpy()
            preds[:, 2:] -= preds[:, :2]
            gts = gt_instances['bboxes'].cpu().numpy()
            gts[:, 2:] -= gts[:, :2]
        else:
            if pred_instances['masks'].dtype == torch.bool:
                pred_masks = pred_instances['masks'].amax(1).cpu().numpy()
                gt_masks = gt_instances['masks'].masks > 0
            else:
                pred_masks = (pred_instances['masks'] > 0).cpu().numpy()
                gt_masks = gt_instances['masks'].masks > 0

            preds = [maskUtils.encode(np.asfortranarray(mask)) for mask in pred_masks]
            gts = [maskUtils.encode(np.asfortranarray(mask)) for mask in gt_masks]
        
        ious = maskUtils.iou(preds, gts, [0]*len(gts))
        
        return np.column_stack(np.nonzero(ious >= iou_thr))

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            gt = data_sample['gt_instances']

            keep_pred = pred['scores'] >= self.score_thr
            for pred_idx, gt_idx in self.match_instances(pred, gt):
                if not keep_pred[pred_idx]:
                    continue

                result = {
                    'gt_label': gt['multilabels'][gt_idx, self.idx].reshape(-1),
                    'pred_label': pred['multilabels'][pred_idx, self.idx].reshape(-1),
                    **({'pred_score': torch.stack((
                        1 - pred['multilogits'][pred_idx][self.idx],
                        pred['multilogits'][pred_idx][self.idx],
                    ))} if 'multilogits' in pred else {}),
                    'num_classes': 2,
                }
                self.results.append(result)
                
    def compute_metrics(self, results: List):
        if not results:
            return {}
        
        # determine and draw ROC curve of results
        self.steps += 1
        roc_metrics = draw_roc_curve(self.results, self.prefix, self.steps)

        thr = roc_metrics['optimal_thr'] if roc_metrics is not None else None
        cm_metrics = draw_confusion_matrix(self.results, thr, self.prefix, self.steps)

        self.results.clear()


        if self.average is not None:
            metrics = {**(roc_metrics if roc_metrics is not None else {}), **cm_metrics}
            return {f'{k}': v for k, v in metrics.items()}

        return {}


@METRICS.register_module()
class AggregateLabelMetric(BaseMetric):

    def __init__(
        self,
        label_idxs: List[int],
        prefixes: List[str],
        **kwargs,
    ):
        super().__init__(prefix='aggregate')

        self.single_metrics = [
            SingleLabelMetric(label_idx, prefix=prefix) for
            label_idx, prefix in zip(label_idxs, prefixes)
        ]

    def process(self, data_batch, data_samples: Sequence[dict]):
        for metric in self.single_metrics:
            metric.process(data_batch, data_samples)

    def compute_metrics(self, results: List):
        metrics = {}
        agg_metrics = defaultdict(list)
        for metric in self.single_metrics:
            metric_dict = metric.compute_metrics(metric.results)

            metrics.update({f'{metric.prefix}/{k}': v for k, v in metric_dict.items()})

            for k, v in metric_dict.items():
                agg_metrics[k] = agg_metrics[k] + [v]

        agg_metrics = {f'aggregate/{k}': np.mean(v) for k, v in agg_metrics.items()}
        metrics.update(agg_metrics)
        
        return metrics
