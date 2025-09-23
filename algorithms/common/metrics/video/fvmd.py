from typing import Any
from torch import Tensor
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print
from .base_fid import BaseFrechetDistance
from .shared_registry import SharedVideoMetricModelRegistry
from .types import VideoMetricModelType


class FrechetVideoMotionDistance(BaseFrechetDistance):
    """
    Calculates Fréchet video motion distance (FVMD) to quantify the similarity between two distributions of videos, focusing on motion consistency.
    - https://arxiv.org/abs/2407.16124
    """

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        reset_real_features: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(
            registry=registry,
            features=1024,
            reset_real_features=reset_real_features,
            **kwargs
        )

    def extract_features(self, x: Tensor) -> Tensor:
        x = 2.0 * x - 1.0
        return self.registry(
            VideoMetricModelType.PIPS,
            x,
        )

    def _check_input(self, fake: Tensor, real: Tensor) -> bool:
        is_valid = fake.shape[1] >= 16 and real.shape[1] >= 16
        if not is_valid:
            rank_zero_print(cyan("FVMD requires at least 16 frames, skipping FVMD."))
        return is_valid
