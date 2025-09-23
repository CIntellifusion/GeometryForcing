from torch import Tensor
from algorithms.common.metrics.video.types import VideoMetricModelType
from .dimension import Dimension


class ImagingQuality(Dimension):
    """
    Imaging quality dimension.
    """

    def forward(self, videos: Tensor) -> Tensor:
        """
        Compute the imaging quality score.
        A 0-100 rating is assigned to each frame using MUSIQ image quality predictor, then normalized to [0, 1], and averaged over all frames.

        Args:
            videos: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        Returns:
            score: The computed imaging quality score of shape (B,), range [0, 1].
        """
        videos = videos.float() / 255.0
        image_quality_scores = self.registry(VideoMetricModelType.MUSIQ, videos)
        return image_quality_scores.mean(dim=1) / 100.0
