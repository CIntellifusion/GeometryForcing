from typing import Any
import torch
from torch import Tensor
from einops import rearrange
from algorithms.common.metrics.video.types import VideoMetricModelType
from algorithms.common.metrics.video.shared_registry import (
    SharedVideoMetricModelRegistry,
)
from algorithms.common.metrics.video.models.raft import InputPadder
from algorithms.common.metrics.video.utils import videos_as_images
from .dimension import Dimension


class DynamicDegree(Dimension):
    """
    Dynamic degree dimension.
    """

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        resolution: int = 224,
        **kwargs: Any
    ) -> None:
        self.padder = InputPadder(dims=(resolution, resolution))
        self._score_threshold = 6.0 * (resolution / 256.0)
        super().__init__(registry, **kwargs)

    def _get_flow(self, videos1: Tensor, videos2: Tensor) -> Tensor:
        """
        Computes the optical flow between the corresponding frames of two video batches.
        """
        return self.registry(
            VideoMetricModelType.RAFT,
            videos1,
            videos2,
            iters=20,
        )

    @videos_as_images
    def _get_score(self, flow: Tensor) -> Tensor:
        """
        Computes a static score given the optical flow.
        Args:
            flow: Optical flow of shape (B, 2, H, W).
        Returns:
            score: The computed static score of shape (B,)
        """
        rad = torch.sqrt(torch.sum(flow**2, dim=1))  # (B, H, W)
        rad = rearrange(rad, "b h w -> b (h w)")
        max_rad, _ = torch.topk(rad, int(rad.shape[1] * 0.05), dim=1)
        return max_rad.abs().mean(dim=1)

    def forward(self, videos: Tensor) -> Tensor:
        """
        Compute the dynamic degree score, which is 0 or 1 depending on whether the video is static or non-static (decided by the optical flow from RAFT).

        Args:
            videos: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        Returns:
            score: The computed dynamic degree score of shape (B,), range [0, 1] (either 0 or 1).
        """
        count_threshold = round(4 * (videos.shape[1] / 16.0))
        videos = self.padder.pad(videos)
        flow = self._get_flow(videos[:, :-1], videos[:, 1:])
        score = self._get_score(flow)
        return ((score > self._score_threshold).sum(dim=1) >= count_threshold).float()
