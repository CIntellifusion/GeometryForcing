from typing import Any, Dict, List
from dataclasses import dataclass
from enum import Enum
import torch
from torch import Tensor
from torchmetrics import Metric
from torchvision.transforms import Resize
from algorithms.common.metrics.video.shared_registry import (
    SharedVideoMetricModelRegistry,
)
from algorithms.common.metrics.video.utils import videos_as_images
from .subject_consistency import SubjectConsistency
from .background_consistency import BackgroundConsistency
from .temporal_flickering import TemporalFlickering
from .aesthetic_quality import AestheticQuality
from .imaging_quality import ImagingQuality
from .dynamic_degree import DynamicDegree
from .motion_smoothness import MotionSmoothness


class VBenchDimensionType(str, Enum):
    # order: dimension, dim_weight, min_val, max_val
    SUBJECT_CONSISTENCY = ("subject_consistency", 1, 0.1462, 1.0)
    BACKGROUND_CONSISTENCY = ("background_consistency", 1, 0.2615, 1.0)
    TEMPORAL_FLICKERING = ("temporal_flickering", 1, 0.6293, 1.0)
    MOTION_SMOOTHNESS = ("motion_smoothness", 1, 0.706, 0.9975)
    DYNAMIC_DEGREE = ("dynamic_degree", 0.5, 0.0, 1.0)
    AESTHETIC_QUALITY = ("aesthetic_quality", 1, 0.0, 1.0)
    IMAGING_QUALITY = ("imaging_quality", 1, 0.0, 1.0)

    def __new__(cls, value, dim_weight, min_val, max_val):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.dim_weight = dim_weight
        obj.min = min_val
        obj.max = max_val
        return obj

    def normalize_score(self, score: Tensor) -> Tensor:
        """Normalize the score (shape: (,)) with the min and max values."""
        return (score - self.min) / (self.max - self.min)

    def weight_score(self, score: Tensor) -> Tensor:
        """Weight the score (shape: (,)) with the dimension weight."""
        return score * self.dim_weight


class VBench(Metric):
    """
    Calculates VBench score.
    - Only quality metrics (subject consistency, background consistency, temporal flickering, motion smoothness, dynamic degree, aesthetic quality, imaging quality) are supported, as the other metrics require text prompts.
    - Requires a batch of videos of shape (B, T, C, H, W) and range [0, 1].

    References:
    - https://arxiv.org/abs/2311.17982
    - https://github.com/Vchitect/VBench
    """

    ALL_DIMENSIONS: List[VBenchDimensionType] = list(VBenchDimensionType)

    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        resolution: int = 224,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            registry: The shared video metric model registry.
            resolution: The resolution to resize the videos to before computing the VBench score.
        """
        super().__init__(**kwargs)

        self.dimensions = torch.nn.ModuleDict(
            {
                VBenchDimensionType.SUBJECT_CONSISTENCY: SubjectConsistency(registry),
                VBenchDimensionType.BACKGROUND_CONSISTENCY: BackgroundConsistency(
                    registry
                ),
                VBenchDimensionType.TEMPORAL_FLICKERING: TemporalFlickering(registry),
                VBenchDimensionType.AESTHETIC_QUALITY: AestheticQuality(registry),
                VBenchDimensionType.IMAGING_QUALITY: ImagingQuality(registry),
                VBenchDimensionType.DYNAMIC_DEGREE: DynamicDegree(registry, resolution),
                VBenchDimensionType.MOTION_SMOOTHNESS: MotionSmoothness(
                    registry, resolution
                ),
            }
        )

        self.transform = videos_as_images(Resize(size=resolution))

        for dimension in self.dimensions.keys():
            self.add_state(f"{dimension}_scores", [], dist_reduce_fx=None)

    def update(self, videos: Tensor) -> None:
        """
        Update the state with the video.
        Args:
            videos: Videos of shape (B, T, C, H, W) and range [0, 1].
        """
        # Convert to uint8, resize to resolutionxresolution (e.g. 224x224)
        videos = (videos.clamp(0, 1) * 255).to(torch.uint8)
        videos = self.transform(videos)

        for dimension in self.ALL_DIMENSIONS:
            getattr(self, f"{dimension}_scores").append(
                self.dimensions[dimension](videos)
            )

    def compute(self) -> Dict[str, Tensor]:
        """
        Compute the VBench score.
        Returns:
            - The computed VBench scores for each dimension.
            - The normalized VBench scores for each dimension.
            - The final VBench score - weighted sum of the normalized scores.
        """
        scores = {
            dimension: torch.cat(getattr(self, f"{dimension}_scores"), dim=0).mean()
            for dimension in self.ALL_DIMENSIONS
        }
        normalized_scores = {
            dimension: dimension.normalize_score(score)
            for dimension, score in scores.items()
        }
        final_score = torch.stack(
            [
                dimension.weight_score(score)
                for dimension, score in normalized_scores.items()
            ]
        ).sum() / sum(dimension.dim_weight for dimension in normalized_scores)

        scores.update(
            {
                f"{dimension}_normalized": normalized_score
                for dimension, normalized_score in normalized_scores.items()
            }
        )
        scores["final"] = final_score
        return scores
