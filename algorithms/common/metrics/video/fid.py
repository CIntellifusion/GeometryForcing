from torch import Tensor
from .shared_registry import SharedVideoMetricModelRegistry
from .types import VideoMetricModelType
from .base_fid import BaseFrechetDistance


class FrechetInceptionDistance(BaseFrechetDistance):
    """
    Calculates Fréchet inception distance (FID) to quantify the similarity between two distributions of images.
    Adapted from `torchmetrics.image.FrechetInceptionDistance`.
    Requires a batch of images of shape (B, C, H, W) and range [0, 1] (normalize=True) or [0, 255] (normalize=False).
    """

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        reset_real_features=True,
        normalize=False,
        **kwargs,
    ):
        self.normalize = normalize

        super().__init__(
            registry=registry,
            features=2048,
            reset_real_features=reset_real_features,
            **kwargs,
        )

    def extract_features(self, x: Tensor) -> Tensor:
        x = (x * 255).byte() if self.normalize else x
        features = self.registry(VideoMetricModelType.INCEPTION_V3, x)
        return features
