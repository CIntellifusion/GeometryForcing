from abc import ABC, abstractmethod
from typing import Any
from torch import Tensor
from torch import nn
from algorithms.common.metrics.video.shared_registry import (
    SharedVideoMetricModelRegistry,
)


class Dimension(nn.Module, ABC):
    """
    Base class for evaluation dimensions in VBench.
    """

    def __init__(self, registry: SharedVideoMetricModelRegistry, **kwargs: Any):
        super().__init__(**kwargs)
        self.registry = registry

    @abstractmethod
    def forward(self, videos: Tensor) -> Tensor:
        """
        Compute the dimension score.
        Args:
            videos: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        Returns:
            The computed dimension score of shape (B,) of range [0, 1].
        """
        raise NotImplementedError
