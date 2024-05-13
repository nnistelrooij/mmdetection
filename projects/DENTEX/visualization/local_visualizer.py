from typing import List, Optional

import cv2
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from mmdet.registry import VISUALIZERS
from mmdet.visualization.local_visualizer import (
    _get_adaptive_scales,
    bitmap_to_polygon,
    get_palette,
    jitter_color,
    BitmapMasks,
    DetLocalVisualizer,
    PolygonMasks,
)


@VISUALIZERS.register_module()
class MultilabelDetLocalVisualizer(DetLocalVisualizer):

    def _draw_instances(self, image: np.ndarray, instances: InstanceData,
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            if 'masks' not in instances:
                self.draw_bboxes(
                    bboxes,
                    edge_colors=colors,
                    alpha=self.alpha,
                    line_widths=self.line_width)

            positions = bboxes[:, :2] + self.line_width
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'
                if 'multilabels' in instances:
                    attrs = []
                    for attr_idx in torch.nonzero(instances.multilabels[i])[:, 0]:
                        attrs.append(self.dataset_meta['attributes'][attr_idx][0])

                    if attrs:
                        label_text += '-' + ''.join(attrs)
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'


                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(26 * scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

            if len(labels) > 0 and \
                    ('bboxes' not in instances or
                     instances.bboxes.sum() == 0):
                # instances.bboxes.sum()==0 represent dummy bboxes.
                # A typical example of SOLO does not exist bbox branch.
                areas = []
                positions = []
                for mask in masks:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8)
                    if stats.shape[0] > 1:
                        largest_id = np.argmax(stats[1:, -1]) + 1
                        positions.append(centroids[largest_id])
                        areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)

                for i, (pos, label) in enumerate(zip(positions, labels)):
                    label_text = classes[
                        label] if classes is not None else f'class {label}'
                    if 'multilabels' in instances:
                        attrs = []
                        for attr_idx in torch.nonzero(instances.multilabels[i])[:, 0]:
                            attrs.append(self.dataset_meta['attributes'][attr_idx][0])

                        if attrs:
                            label_text += '-' + ''.join(attrs)
                    if 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        label_text += f': {score}'

                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(26 * scales[i]),
                        horizontal_alignments='center',
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])
                
        return self.get_image()


@VISUALIZERS.register_module()
class MulticlassDetLocalVisualizer(DetLocalVisualizer):

    FDIs = list(map(str, [
        11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48,
    ]))

    def _draw_instances(self, image: np.ndarray, instances: InstanceData,
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)
        instances.pop('multilabels', None)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)

            positions = bboxes[:, :2] + self.line_width
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            fdis = MulticlassDetLocalVisualizer.FDIs
            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = fdis[label] if label < len(fdis) else 'Irrelevant'
                if 'multilabels' in instances:
                    attrs = []
                    for attr_idx in torch.nonzero(instances.multilabels[i])[:, 0]:
                        attrs.append(self.dataset_meta['classes'][attr_idx][0])

                    if attrs:
                        label_text += '-' + ''.join(attrs)
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'


                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(26 * scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            counts = []
            binary_masks = []
            for i, mask in enumerate(masks):
                counts.append(0)
                if mask.dtype == np.uint8:
                    for i in range(8):
                        binary_masks.append((mask & (2**i)) > 0)
                        counts[-1] += 1
                elif mask.dtype == np.int64:
                    for i in range(1, 9):
                        binary_masks.append(mask == i)
                        counts[-1] += 1
                elif mask.dtype == np.bool_:
                    binary_masks.extend(mask)
                    counts[-1] += len(mask)
                else:
                    raise ValueError(f'Expected np.uint8 or np.bool_ dtype, but found {mask.dtype}')
            binary_masks = np.stack(binary_masks) if binary_masks else (
                np.zeros((1, image.shape[0], image.shape[1]), dtype=bool)
            )

            mask_colors = []
            for i in range(binary_masks.shape[0]):
                if (i % 8) in [1, 2]:
                    color = mask_palette[6 + (i % 8)]
                else:
                    color = colors[i // 8]
                mask_colors.append(color)

            polygons = []
            for binary_mask in binary_masks:                        
                contours, _ = bitmap_to_polygon(binary_mask)
                polygons.extend(contours)            

            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(binary_masks, colors=mask_colors, alphas=self.alpha)

            if len(labels) > 0 and \
                    ('bboxes' not in instances or
                     instances.bboxes.sum() == 0):
                # instances.bboxes.sum()==0 represent dummy bboxes.
                # A typical example of SOLO does not exist bbox branch.
                areas = []
                positions = []
                for mask in binary_mask:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8)
                    if stats.shape[0] > 1:
                        largest_id = np.argmax(stats[1:, -1]) + 1
                        positions.append(centroids[largest_id])
                        areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)

                for i, (pos, label) in enumerate(zip(positions, labels)):
                    label_text = classes[
                        label] if classes is not None else f'class {label}'
                    if 'multilabels' in instances:
                        attrs = []
                        for attr_idx in torch.nonzero(instances.multilabels[i])[:, 0]:
                            attrs.append(self.dataset_meta['attributes'][attr_idx][0])

                        if attrs:
                            label_text += '-' + ''.join(attrs)
                    if 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        label_text += f': {score}'

                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(26 * scales[i]),
                        horizontal_alignments='center',
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])
                
        return self.get_image()

    @master_only
    def add_datasample(
        self,
        name: str,
        *args,
        **kwargs,
    ):
        MMLogger.get_current_instance().info(name)

        super().add_datasample(name, *args, **kwargs)
