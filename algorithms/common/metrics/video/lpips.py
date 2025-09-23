from typing import Any, Literal
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image.lpip import (
    LearnedPerceptualImagePatchSimilarity as _LearnedPerceptualImagePatchSimilarity,
    _valid_img,
)
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE
from .shared_registry import SharedVideoMetricModelRegistry
from .types import VideoMetricModelType


class LearnedPerceptualImagePatchSimilarity(_LearnedPerceptualImagePatchSimilarity):
    """
    Calculates Learned Perceptual Image Patch Similarity (LPIPS) metric.
    Requires a batch of images of shape (B, C, H, W) and range [0, 1] (normalize=True) or [-1, 1] (normalize=False).
    """

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        reduction: Literal["sum", "mean"] = "mean",
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        Metric.__init__(self, **kwargs)  # pylint: disable=non-parent-init-called

        if not _LPIPS_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS metric requires that lpips is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install lpips`."
            )

        self.registry = registry

        valid_reduction = ("mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(
                f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}"
            )
        self.reduction = reduction

        if not isinstance(normalize, bool):
            raise ValueError(
                f"Argument `normalize` should be an bool but got {normalize}"
            )
        self.normalize = normalize

        self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:  # type: ignore
        """Update internal states with lpips score."""
        if not (_valid_img(img1, self.normalize) and _valid_img(img2, self.normalize)):
            raise ValueError(
                "Expected both input arguments to be normalized tensors with shape [N, 3, H, W]."
                f" Got input with shape {img1.shape} and {img2.shape} and values in range"
                f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are"
                f" expected to be in the {[0,1] if self.normalize else [-1,1]} range."
            )
        loss = self.registry(
            VideoMetricModelType.LPIPS, img1, img2, normalize=self.normalize
        ).squeeze()
        # pylint: disable=no-member
        self.sum_scores += loss.sum()
        self.total += img1.shape[0]
