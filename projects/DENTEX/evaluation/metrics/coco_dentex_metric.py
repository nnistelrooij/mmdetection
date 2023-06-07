import copy
import json
from pathlib import Path
import tempfile
from typing import Dict, List, Sequence

from mmengine.fileio import dump
from mmengine.logging import MMLogger
import numpy as np

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
from mmdet.evaluation import CocoMetric


@METRICS.register_module()
class CocoDENTEXMetric(CocoMetric):

    CAT_MAPS = {
        'quadrants':{
            f'{q}{e}': {'id': q - 1, 'name': str(q)}
            for q in range(1, 5) for e in range(1, 9)
        },
        'enumeration': {
            f'{q}{e}': {'id': e - 1, 'name': str(e)}
            for q in range(1, 5) for e in range(1, 9)
        },
        'diagnosis': {
            diag: {'id': i, 'name': diag} for i, diag in enumerate(
                ['Caries', 'Deep Caries', 'Impacted', 'Periapical Lesion']
            )
        },
    }

    def __init__(
        self,
        proposal_nums: Sequence[int]=(1, 10, 100),
        *args,
        **kwargs,
    ):
        super().__init__(proposal_nums=proposal_nums, *args, **kwargs)

        self.metric_items = [
            'mAP', 'mAP_50', 'mAP_75', 'AR@1000',  # AR@1000 == AR@100
        ]
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        super().process(data_batch, data_samples)

        for data_sample, (_, result) in zip(data_samples[::-1], self.results[::-1]):
            pred = data_sample['pred_instances']
            result['multilabels'] = pred['multilabels'].cpu().numpy()

    def ann2cats(
        self,
        ann,
        cat_map: Dict[str, Dict],
        split_diagnoses: bool,
    ) -> List[int]:
        cat_name = self._coco_api.cats[ann['category_id']]['name']
        if not split_diagnoses and cat_name in cat_map:
            return [cat_map[cat_name]['id']]            
        
        if (
            'extra' not in ann or
            'attributes' not in ann['extra']
        ):
            return []

        cat_ids = []
        for attr in ann['extra']['attributes']:
            if cat_name not in cat_map:
                cat_name = attr

            cat_ids.append(cat_map[cat_name]['id'])

        return cat_ids
    
    def labels2cats(
        self,
        label: int,
        multilabel: List[int],
        cat_map: Dict[str, Dict],
        split_diagnoses: bool,
    ) -> List[int]:
        cat_name = self.dataset_meta['classes'][label]
        if not split_diagnoses and cat_name in cat_map:
            return [cat_map[cat_name]['id']]
        

        if not np.any(multilabel):
            return []

        cat_ids = []
        for attr_idx in np.nonzero(multilabel)[0]:
            if cat_name not in cat_map:
                cat_name = self.dataset_meta['attributes'][attr_idx]

            cat_ids.append(cat_map[cat_name]['id'])

        return cat_ids
    
    def pred2cats(
        self,
        pred,
        cat_map: Dict[str, Dict],
        split_diagnoses: bool
    ) -> List[int]:
        bboxes, scores, labels, masks = [], [], [], []
        for bbox, score, label, mask, multilabel in zip(*(pred[key] for key in [
            'bboxes', 'scores', 'labels', 'masks', 'multilabels',
        ])):
            for cat_id in self.labels2cats(label, multilabel, cat_map, split_diagnoses):
                bboxes.append(bbox)
                scores.append(score)
                labels.append(cat_id)
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

        categories = {cat['id']: cat for cat in cat_map.values()}
        coco_json_dict['categories'] = list(categories.values())

        dump(coco_json_dict, coco_json_path)

        return coco_json_path
    
    def prepare_gts(
        self,
        gts: list,
        cat_map: Dict[str, Dict],
        split_diagnoses: bool,
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

                cat_ids = self.ann2cats(ann, cat_map, split_diagnoses)
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
        split_diagnoses: bool,
    ) -> list:
        preds_list = []
        for pred in preds:
            pred = self.pred2cats(pred, cat_map, split_diagnoses)
            preds_list.append(pred)

        return preds_list

    def compute_metrics_single(
        self,
        results: list,
        cat_map: str,
    ) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        
        task = cat_map
        cat_map = self.CAT_MAPS[cat_map]
        task_metrics = {}
        for split_diagnoses in [False, True]:
            if task == 'diagnosis' and not split_diagnoses:
                continue

            results = copy.deepcopy(results)
            gts, preds = zip(*results)

            coco = self.prepare_gts(gts, cat_map, split_diagnoses)
            preds = self.prepare_preds(preds, cat_map, split_diagnoses)

            self._orig_coco_api = self._coco_api
            self._coco_api = coco
            self.cat_ids = np.unique([c['id'] for c in cat_map.values()])

            logger.info('Evaluating {}, {} splitting diagnoses'.format(
                task, '' if split_diagnoses else 'not',
            ))
            metrics = super().compute_metrics(zip(gts, preds))

            task_metrics.update({
                k.replace('_', f'_split-diagnoses={split_diagnoses}_', 1): v
                for k, v in metrics.items()
            })

            self._coco_api = self._orig_coco_api

        return task_metrics

    def compute_metrics(self, results: list) -> Dict[str, float]:
        metrics = {}
        for cat_map in self.CAT_MAPS:
            task_metrics = self.compute_metrics_single(results, cat_map)

            for metric, value in task_metrics.items():
                metric_name = '_'.join(metric.split('_')[:2])
                metrics.setdefault(metric_name, []).append(value)

        return {metric: np.mean(values) for metric, values in metrics.items()}
