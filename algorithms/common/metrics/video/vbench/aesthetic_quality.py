from torch import Tensor
import torch.nn.functional as F
from algorithms.common.metrics.video.types import VideoMetricModelType
from .dimension import Dimension


class AestheticQuality(Dimension):
    """
    Aesthetic quality dimension.
    """

    def forward(self, videos: Tensor) -> Tensor:
        """
        Compute the aesthetic quality score.
        A 0-10 rating is assigned to each frame by applying LAION aesthetic predictor to CLIP features, then normalized to [0, 1], and averaged over all frames.

        Args:
            videos: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        Returns:
            score: The computed aesthetic quality score of shape (B,), range [0, 1].
        """
        clip_features = self.registry(VideoMetricModelType.CLIP_L_14, videos)
        clip_features = F.normalize(clip_features, p=2, dim=-1)
        aesthetic_scores = self.registry(VideoMetricModelType.LAION, clip_features)
        return aesthetic_scores.mean(dim=1) / 10.0
