from torch import nn, Tensor
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    Lambda,
    InterpolationMode,
)
import clip
from algorithms.common.metrics.video.utils import videos_as_images


class CLIP(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.model = clip.load(name, device="cpu")[0]
        self.transform = Compose(
            [
                Resize(size=224, interpolation=InterpolationMode.BICUBIC),
                Lambda(lambda x: x.float() / 255.0),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def train(self, mode: bool) -> "CLIP":
        super().train(False)

    @videos_as_images
    def forward(self, images: Tensor) -> Tensor:
        return self.model.encode_image(self.transform(images))
