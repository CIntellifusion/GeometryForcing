from typing import Any
from torch import Tensor
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print_once
from .base_fid import BaseFrechetDistance


class FrechetVideoDistance(BaseFrechetDistance):
    """
    Calculates Fréchet video distance (FVD) to quantify the similarity between two distributions of videos.
    """

    def __init__(self, reset_real_features: bool = True, **kwargs: Any) -> None:
        super().__init__(
            registry=None,
            features=400,
            reset_real_features=reset_real_features,
            **kwargs,
        )

    def extract_features(self, x: Tensor) -> Tensor:
        return x

    @staticmethod
    def _check_input(fake: Tensor, real: Tensor, skip_message: str = "FVD") -> bool:
        is_valid = fake.shape[1] >= 9 and real.shape[1] >= 9
        if not is_valid:
            rank_zero_print_once(
                cyan(f"I3D requires at least 9 frames, skipping {skip_message}.")
            )
        return is_valid
