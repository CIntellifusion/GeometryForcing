from typing import Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .distribution import DiagonalGaussianDistribution


class VAE(ABC, nn.Module):
    """
    Base VAE class.
    """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encode input tensor x into latent distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor z into original space.
        """
        raise NotImplementedError

    def forward(
        self, sample: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """
        Forward pass.
        Returns:
            - dec: reconstructed input, uses mode of latent distribution if sample_posterior is False, otherwise sample from it
            - posterior: latent distribution
        """
        posterior = self.encode(sample)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "VAE":
        """
        Load pretrained model from path, with additional kwargs.
        """
        raise NotImplementedError
