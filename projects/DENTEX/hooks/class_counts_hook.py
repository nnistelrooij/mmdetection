from mmengine.hooks import Hook
from mmengine.runner import Runner
import torch

from mmdet.registry import HOOKS


@HOOKS.register_module()
class ClassCountsHook(Hook):

    def before_train(self, runner: Runner) -> None:        
        dataset = runner.train_dataloader.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        class_counts = torch.zeros(len(dataset.metainfo['classes']))
        attribute_counts = torch.zeros(len(dataset.metainfo['attributes']))
        for img_anns in dataset.data_list:
            for ann in img_anns['instances']:
                label = ann['bbox_label']
                attributes_onehot = ann['bbox_multilabel']

                class_counts[label] += 1
                attribute_counts += torch.tensor(attributes_onehot).float()

        criterion = runner.model.panoptic_head.criterion
        criterion.set_federated_class_weights(
            class_counts, attribute_counts,
        )
        