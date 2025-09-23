from typing import Any
import torch
from torch import Tensor
from torchmetrics import Metric


class InceptionScore(Metric):
    """
    Calculates Inception Score (IS) to evaluate how realistic the videos are.
    Requires a batch of videos of shape (B, num_classes).
    Adapted from `torchmetrics.image.InceptionScore` to work with videos.
    """

    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False
    orig_dtype: torch.dtype

    def __init__(self, num_classes: int = 400, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state(
            "prob_sum", torch.zeros(num_classes).double(), dist_reduce_fx="sum"
        )
        self.add_state("num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")
        self.add_state(
            "neg_entropy_sum", torch.tensor(0.0).double(), dist_reduce_fx="sum"
        )

    @property
    def is_empty(self) -> bool:
        # pylint: disable=no-member
        return (self.num_samples == 0).item()

    def update(self, features: Tensor) -> None:
        """
        Update the state with extracted features.
        Args:
            features: Features of shape (B, num_classes). The features are logits extracted from a video classifier (e.g. I3D, C3D).
        """
        # pylint: disable=no-member
        self.orig_dtype = features.dtype
        features = features.double()
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)
        self.num_samples += features.size(0)
        self.prob_sum += prob.sum(dim=0)
        self.neg_entropy_sum += (prob * log_prob).sum()

    def compute(self) -> Tensor:
        """
        Compute the Inception Score (IS).
        Returns:
            The computed IS.
        """
        # pylint: disable=no-member
        mean_prob = self.prob_sum / self.num_samples
        # calculate KL divergence
        kl = (
            self.neg_entropy_sum / self.num_samples
            - (mean_prob * mean_prob.log()).sum()
        )
        # calculate IS
        return kl.exp().to(self.orig_dtype)
