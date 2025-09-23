from abc import ABC, abstractmethod
from torch import Tensor
import torch.nn.functional as F
from .dimension import Dimension


class CosineSimilarityDimension(Dimension, ABC):
    @abstractmethod
    def extract_features(self, videos: Tensor) -> Tensor:
        """
        Extract image features from the videos.
        Args:
            videos: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        Returns:
            The extracted features of shape (B, T, num_features).
        """
        raise NotImplementedError

    def forward(self, videos: Tensor) -> Tensor:
        """
        Compute the cosine similarity score.
        It is the average of:
        - the cosine similarity between consecutive frames.
        - the cosine similarity between the first and each frame.
        Cosine similarity is clamped to be non-negative.
        Args:
            videos: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        Returns:
            The computed cosine similarity score of shape (B,) of range [0, 1].
        """
        # pylint: disable=not-callable
        features = self.extract_features(videos)
        cos_sim_consecutive = F.cosine_similarity(
            features[:, :-1], features[:, 1:], dim=-1
        )
        cos_sim_first = F.cosine_similarity(features[:, 0:1], features[:, 1:], dim=-1)
        cos_sim_consecutive, cos_sim_first = map(
            lambda x: x.clamp(min=0), (cos_sim_consecutive, cos_sim_first)
        )
        return (cos_sim_consecutive + cos_sim_first).mean(dim=1) / 2.0
