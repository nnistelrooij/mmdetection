import torch.nn as nn

from mmdet.models import ResNet
from mmdet.registry import MODELS


@MODELS.register_module()
class GrayscaleResNet(ResNet):

    def __init__(
        self, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
