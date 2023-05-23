import copy
import json
from pathlib import Path
import tempfile
from typing import Callable, Dict, List, Sequence

from mmengine.fileio import dump
import numpy as np

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
from mmdet.evaluation import CocoMetric


@METRICS.register_module()
class CocoOPGMetric(CocoMetric):

    QUADRANT_NAME2CAT = {
        f'{q}{e}': {'id': q - 1, 'name': str(q)}
        for q in range(1, 5) for e in range(1, 9)
    }
    ENUMERATION_NAME2CAT = {
        f'{q}{e}': {'id': e - 1, 'name': str(e)}
        for q in range(1, 5) for e in range(1, 9)
    }
    DIAGNOSIS_NAME2CAT = {
        diag: {'id': i, 'name': diag} for i, diag in enumerate(
            ['Caries', 'Deep Caries', 'Impacted', 'Periapical Lesion']
        )
    }

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.metric_items = [
            'mAP', 'mAP_50', 'mAP_75', 'AR@100',
        ]
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        super().process(data_batch, data_samples)

        for data_sample, (_, result) in zip(data_samples[::-1], self.results[::-1]):
            pred = data_sample['pred_instances']
            result['multilabels'] = pred['multilabels'].cpu().numpy()

    def ann2cats(
        self,
        ann,
        cat_map,
    ) -> List[int]:
        if (
            'extra' not in ann or
            'attributes' not in ann['extra']
        ):
            return []
        
        cat_name = self._coco_api.cats[ann['category_id']]['name']
        cat_ids = [cat_map[cat_name]['id']] if cat_name in cat_map else []

        for cat_name in ann['extra']['attributes']:
            if cat_name in cat_map:
                cat_ids.append(cat_map[cat_name]['id'])

        return cat_ids
    
    def pred2cats(
        self,
        pred,
        cat_map,
    ) -> List[int]:
        bboxes, scores, labels, masks = [], [], [], []
        for bbox, score, label, mask, multilabel in zip(*(pred[key] for key in [
            'bboxes', 'scores', 'labels', 'masks', 'multilabels',
        ])):
            if not np.any(multilabel):
                continue

            for label in np.nonzero(multilabel)[0]:
                cat_name = self.dataset_meta['classes'][label]
                if cat_name not in cat_map:
                    cat_name = self.dataset_meta['attributes'][label]

                bboxes.append(bbox)
                scores.append(score)
                labels.append(cat_map[cat_name]['id'])
                masks.append(mask)

        pred = {
            'img_id': pred['img_id'],
            'bboxes': np.stack(bboxes) if bboxes else np.zeros((0, 4), dtype=np.float32),
            'scores': np.array(scores),
            'labels': np.array(labels),
            'masks': masks,
        }

        return pred

    def gt_to_coco_json(
        self,
        gt_dicts: Sequence[dict],
        cat_map: Dict[str, Dict],
        outfile_prefix: str,
    ) -> str:
        coco_json_path = super().gt_to_coco_json(gt_dicts, outfile_prefix)

        with open(coco_json_path, 'r') as f:
            coco_json_dict = json.load(f)

        coco_json_dict['categories'] = list(cat_map.values())

        dump(coco_json_dict, coco_json_path)

        return coco_json_path
    
    def prepare_gts(
        self,
        gts: list,
        cat_map: Dict[str, Dict],
    ) -> COCO:
        gt_dicts = []
        for gt_dict in gts:
            image_id = gt_dict['img_id']
            anns = self._coco_api.imgToAnns[image_id]

            gt_dict['anns'] = []
            for ann in copy.deepcopy(anns):
                ann['bbox'] = [
                    ann['bbox'][0],
                    ann['bbox'][1],
                    ann['bbox'][0] + ann['bbox'][2],
                    ann['bbox'][1] + ann['bbox'][3],
                ]

                ann['mask'] = ann['segmentation']

                cat_ids = self.ann2cats(ann, cat_map)
                for cat_id in cat_ids:
                    ann = copy.deepcopy(ann)
                    ann['bbox_label'] = cat_id
                    gt_dict['anns'].append(ann)

            gt_dicts.append(gt_dict)       

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = Path(tmp_dir.name) / 'results'
        else:
            outfile_prefix = self.outfile_prefix

        coco_json_path = self.gt_to_coco_json(gt_dicts, cat_map, outfile_prefix)
        coco = COCO(coco_json_path)

        return coco
    
    def prepare_preds(
        self,
        preds: list,
        cat_map: Dict[str, Dict],
    ) -> list:
        preds_list = []
        for pred in preds:
            pred = self.pred2cats(pred, cat_map)
            preds_list.append(pred)

        return preds_list

    def compute_metrics_single(
        self,
        results: list,
        cat_map: Callable[[Dict], List[int]],
    ) -> Dict[str, float]:
        results = copy.deepcopy(results)
        gts, preds = zip(*results)

        coco = self.prepare_gts(gts, cat_map)
        preds = self.prepare_preds(preds, cat_map)

        self._orig_coco_api = self._coco_api
        self._coco_api = coco
        self.cat_ids = np.unique([c['id'] for c in cat_map.values()])

        results = zip(gts, preds)
        metrics = super().compute_metrics(results)

        self._coco_api = self._orig_coco_api

        return metrics

    def compute_metrics(self, results: list) -> Dict[str, float]:
        metrics = {metric: [] for metric in self.metrics}
        for cat_map in [
            self.QUADRANT_NAME2CAT,
            self.ENUMERATION_NAME2CAT,
            self.DIAGNOSIS_NAME2CAT,
        ]:
            task_metrics = self.compute_metrics_single(results, cat_map)

            for metric, value in task_metrics.items():
                metrics[metric.split('_')[0]].append(value)
            
        return {metric: np.mean(values) for metric, values in metrics.items()}
