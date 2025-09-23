from typing import Any
from torch import Tensor
from algorithms.common.metrics.video.types import VideoMetricModelType
from algorithms.common.metrics.video.shared_registry import (
    SharedVideoMetricModelRegistry,
)
from algorithms.common.metrics.video.models.amt import InputPadder
from .dimension import Dimension
from .utils import mae_score


class MotionSmoothness(Dimension):
    """
    Motion smoothness dimension.
    """

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        resolution: int = 224,
        **kwargs: Any
    ) -> None:
        self.padder = InputPadder(dims=(resolution, resolution), divisor=16)
        super().__init__(registry, **kwargs)

    def _interpolate(self, videos1: Tensor, videos2: Tensor) -> Tensor:
        """
        Interpolate the corresponding frames of videos1 and videos2.
        """
        return self.registry(VideoMetricModelType.AMT_S, videos1, videos2)

    def forward(self, videos: Tensor) -> Tensor:
        """
        Compute the motion smoothness, which is computed by:
        1. Drop odd frames and interpolate them by passing the even frames through AMT-S model.
        2. Compute MAE between the interpolated frames and the original frames.
        3. Normalize the MAE from [0, 255] to [0, 1], and average them across all frames.

        Args:
            videos: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        Returns:
            score: The computed motion smoothness score of shape (B,), range [0, 1].
        """
        odd_videos = videos[:, 1::2]
        even_videos = videos[:, ::2]
        even_videos = self.padder.pad(even_videos)
        interpolated = self._interpolate(even_videos[:, :-1], even_videos[:, 1:])
        interpolated = self.padder.unpad(interpolated)
        odd_videos = odd_videos[:, : interpolated.shape[1]]
        return mae_score(odd_videos, interpolated)
