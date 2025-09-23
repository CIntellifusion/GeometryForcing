from torch import Tensor
from algorithms.common.metrics.video.types import VideoMetricModelType
from .cosine_similarity_dimension import CosineSimilarityDimension


class BackgroundConsistency(CosineSimilarityDimension):
    """
    Background consistency dimension.
    """

    def extract_features(self, videos: Tensor) -> Tensor:
        """
        Extract CLIP features from videos.
        """
        return self.registry(VideoMetricModelType.CLIP_B_32, videos)
