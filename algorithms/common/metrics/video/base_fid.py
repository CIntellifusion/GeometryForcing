from typing import Optional
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image import FrechetInceptionDistance as _FrechetInceptionDistance
from .shared_registry import SharedVideoMetricModelRegistry


class BaseFrechetDistance(_FrechetInceptionDistance, ABC):
    """
    Base class for Fréchet distance metrics (e.g. FID, FVD).
    AAdapted from `torchmetrics.image.FrechetInceptionDistance` to work with shared model registry and support different feature extractors and modalities (e.g. images, videos).
    """

    orig_dtype: torch.dtype

    def __init__(
        self,
        registry: Optional[SharedVideoMetricModelRegistry],
        features: int,
        reset_real_features=True,
        **kwargs,
    ):
        # pylint: disable=non-parent-init-called
        Metric.__init__(self, **kwargs)

        self.registry = registry
        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        num_features = features
        mx_nb_feets = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_nb_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_nb_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    @property
    def is_empty(self) -> bool:
        # pylint: disable=no-member
        return (
            self.real_features_num_samples == 0 or self.fake_features_num_samples == 0
        )

    @abstractmethod
    def extract_features(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _check_input(fake: Tensor, real: Tensor) -> bool:
        return True

    def _update(self, x: Tensor, real: bool) -> None:
        # pylint: disable=no-member
        features = self.extract_features(x)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.size(0)
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.size(0)

    def update(self, fake: Tensor, real: Tensor) -> None:
        if not self._check_input(fake, real):
            return
        self._update(fake, real=False)
        self._update(real, real=True)
