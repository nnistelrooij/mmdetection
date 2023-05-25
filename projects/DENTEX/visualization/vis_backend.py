from pathlib import Path

from mmengine.registry import VISBACKENDS
from mmengine.visualization.vis_backend import (
    TensorboardVisBackend,
    force_init_env,
)
import numpy as np


@VISBACKENDS.register_module()
class SparseTensorboardVisBackend(TensorboardVisBackend):
    
    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to tensorboard.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Global step value to record. Defaults to 0.
        """
        # only keep five most recent images
        img_dir = Path(self._save_dir) / 'vis_image'
        img_files = sorted(img_dir.glob('*'), key=lambda f: int(f.stem.split('_')[-1]))
        for img_file in img_files[:-5]:
            img_file.unlink()

        super().add_image(name, image, step, **kwargs)
