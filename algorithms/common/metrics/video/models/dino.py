import sys
import torch
from torch import nn, Tensor
from torchvision.transforms import Compose, Resize, Normalize, Lambda
from algorithms.common.metrics.video.utils import videos_as_images


class DINO(nn.Module):
    def __init__(self):
        super().__init__()
        # NOTE: this is a workaround to avoid the error from module collision
        # https://pytorch.org/docs/stable/hub.html
        # https://github.com/pytorch/hub/issues/243#issuecomment-942403391
        sys.modules.pop("utils")
        self.model = torch.hub.load(
            repo_or_dir="facebookresearch/dino:main",
            source="github",
            model="dino_vitb16",
            verbose=False,
        )
        self.transform = Compose(
            [
                Resize(size=224),
                Lambda(lambda x: x.float() / 255.0),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def train(self, mode: bool) -> "DINO":
        super().train(False)

    @videos_as_images
    def forward(self, images: Tensor) -> Tensor:
        return self.model(self.transform(images))
