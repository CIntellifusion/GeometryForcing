from typing import Optional, Tuple
import math
import torch
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
from diffusers.models.embeddings import TimestepEmbedding
from rotary_embedding_torch.rotary_embedding_torch import rotate_half
from timm.models.vision_transformer import PatchEmbed


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class StochasticUnknownTimesteps(Timesteps):
    def __init__(
        self,
        num_channels: int,
        p: float = 1.0,
    ):
        super().__init__(num_channels)
        self.unknown_token = (
            nn.Parameter(torch.randn(1, num_channels)) if p > 0.0 else None
        )
        self.p = p

    def forward(self, timesteps: torch.Tensor, mask: Optional[torch.Tensor] = None):
        t_emb = super().forward(timesteps)
        # if p == 0.0 - return original embeddings both during training and inference
        if self.p == 0.0:
            return t_emb

        # training or mask is None - randomly replace embeddings with unknown token with probability p
        # (mask can only be None for logging training visualization when using latents)
        # or if p == 1.0 - always replace embeddings with unknown token even during inference)
        if self.training or self.p == 1.0 or mask is None:
            mask = torch.rand(t_emb.shape[:-1], device=t_emb.device) < self.p
            mask = mask[..., None].expand_as(t_emb)
            return torch.where(mask, self.unknown_token, t_emb)

        # # inference with p < 1.0 - replace embeddings with unknown token only for masked timesteps
        # if mask is None:
        #     assert False, "mask should be provided when 0.0 < p < 1.0"
        mask = mask[..., None].expand_as(t_emb)
        return torch.where(mask, self.unknown_token, t_emb)


class StochasticTimeEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_embed_dim: int,
        use_fourier: bool = False,
        p: float = 0.0,
    ):
        super().__init__()
        self.use_fourier = use_fourier
        if self.use_fourier:
            assert p == 0.0, "Fourier embeddings do not support stochastic timesteps"
        self.timesteps = (
            FourierEmbedding(dim, bandwidth=1)
            if use_fourier
            else StochasticUnknownTimesteps(dim, p)
        )
        self.embedding = TimestepEmbedding(dim, time_embed_dim)

    def forward(self, timesteps: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.embedding(
            self.timesteps(timesteps)
            if self.use_fourier
            else self.timesteps(timesteps, mask)
        )


class FourierEmbedding(torch.nn.Module):
    """
    Adapted from EDM2 - https://github.com/NVlabs/edm2/blob/38d5a70fe338edc8b3aac4da8a0cefbc4a057fb8/training/networks_edm2.py#L73
    """

    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y[..., None] * self.freqs.to(torch.float32)
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D or 2-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] or [N x M x dim] Tensor of positional embeddings.
    """
    if len(timesteps.shape) not in [1, 2]:
        raise ValueError("Timesteps should be a 1D or 2D tensor")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[..., None].float() * emb

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[..., half_dim:], emb[..., :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class RotaryEmbeddingND(nn.Module):
    """
    Minimal Axial RoPE generalized to N dimensions.
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        sizes: Tuple[int, ...],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        """
        Args:
            dims: the number of dimensions for each axis.
            sizes: the maximum length for each axis.
        """
        super().__init__()
        self.n_dims = len(dims)
        self.dims = dims
        self.theta = theta
        self.flatten = flatten

        Colon = slice(None)
        all_freqs = []
        for i, (dim, seq_len) in enumerate(zip(dims, sizes)):
            freqs = self.get_freqs(dim, seq_len)
            all_axis = [None] * len(dims)
            all_axis[i] = Colon
            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice].expand(*sizes, dim))
        all_freqs = torch.cat(all_freqs, dim=-1)
        if flatten:  # flatten all but the last dimension
            all_freqs = rearrange(all_freqs, "... d -> (...) d")
        self.register_buffer("freqs", all_freqs, persistent=False)

    def get_freqs(self, dim: int, seq_len: int) -> torch.Tensor:
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        pos = torch.arange(seq_len, dtype=freqs.dtype)
        freqs = einsum("..., f -> ... f", pos, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a [... x N x ... x D] if flatten=False, [... x (N x ...) x D] if flatten=True tensor of queries or keys.
        Returns:
            a tensor of rotated queries or keys. (same shape as x)
        """
        # slice the freqs to match the input shape
        seq_shape = x.shape[-2:-1] if self.flatten else x.shape[-self.n_dims - 1 : -1]
        slice_tuple = tuple(slice(0, seq_len) for seq_len in seq_shape)
        freqs = self.freqs[slice_tuple]
        return x * freqs.cos() + rotate_half(x) * freqs.sin()


class RotaryEmbedding1D(RotaryEmbeddingND):
    """
    RoPE1D for Time Series Transformer.
    Handles tensors of shape [B x T x C] or [B x (T x C)].
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        super().__init__((dim,), (seq_len,), theta, flatten)


class RotaryEmbedding2D(RotaryEmbeddingND):
    """
    RoPE2D for Image Transformer.
    Handles tensors of shape [B x H x W x C] or [B x (H x W) x C].
    """

    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        assert dim % 2 == 0, "RotaryEmbedding2D requires even dim"
        super().__init__((dim // 2,) * 2, sizes, theta, flatten)


class RotaryEmbedding3D(RotaryEmbeddingND):
    """
    RoPE3D for Video Transformer.
    Handles tensors of shape [B x T x H x W x C] or [B x (T x H x W) x C].
    """

    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        assert dim % 2 == 0, "RotaryEmbedding3D requires even dim"
        dim //= 2

        # if dim is not divisible by 3,
        # split into 3 dimensions such that height and width have the same number of frequencies
        match dim % 3:
            case 0:
                dims = (dim // 3,) * 3
            case 1:
                dims = (dim // 3 + 1, dim // 3, dim // 3)
            case 2:
                dims = (dim // 3, dim // 3 + 1, dim // 3 + 1)

        super().__init__(tuple(d * 2 for d in dims), sizes, theta, flatten)


class RandomEmbeddingDropout(nn.Module):
    """
    Randomly nullify the input embeddings with a given probability.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Randomly nullify the input embeddings with a probability p during training. For inference, the embeddings are nullified only if mask is provided.
        Args:
            emb: input embeddings of shape (B, ...)
            mask: mask tensor of shape (B, ). Only allowed during inference. If provided, embeddings for masked batches will be zeroed.
        """
        if mask is not None:
            assert not self.training, "embedding mask is only allowed during inference"
            assert mask.ndim == 1, "embedding mask should be of shape (B,)"

        if self.training and self.p > 0:
            mask = torch.rand(emb.shape[:1], device=emb.device) < self.p
        if mask is not None:
            mask = rearrange(mask, "... -> ..." + " 1" * (emb.ndim - 1))
            emb = torch.where(mask, torch.zeros_like(emb), emb)
        return emb


class RandomDropoutCondEmbedding(TimestepEmbedding):
    """
    A layer for processing conditions into embeddings, randomly dropping embeddings of each frame during training.
    NOTE: If dropout_prob is 0, it will fall back to `TimestepEmbedding`. We use this trick to ensure the backward compatibility with our previous checkpoints.
    """

    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        dropout_prob: float = 0.0,
    ):
        self.dropout_prob = dropout_prob
        if dropout_prob == 0:
            super().__init__(cond_dim, cond_emb_dim)
        else:
            nn.Module.__init__(self)
            self.dropout = RandomEmbeddingDropout(p=dropout_prob)
            self.embedding = TimestepEmbedding(cond_dim, cond_emb_dim)

    def forward(self, cond: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.dropout_prob == 0:
            return super().forward(cond)
        return self.dropout(self.embedding(cond), mask)


class RandomDropoutPatchEmbed(nn.Module):
    def __init__(
        self,
        dropout_prob: float = 0.1,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
        flatten: bool = True,
        **patch_embed_kwargs,
    ):
        super().__init__()
        self.dropout = RandomEmbeddingDropout(p=dropout_prob)
        self.patch_embedder = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=bias,
            flatten=flatten,
            **patch_embed_kwargs,
        )
        self.ndim = 3 if flatten else 4

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: tensor to be patchified of shape (*B, C, H, W)
        Returns:
            patchified tensor of shape (*B, num_patches, embed_dim)
        """
        orig_shape = x.shape
        x = rearrange(x, "... c h w -> (...) c h w")
        x = self.patch_embedder(x)
        x = x.reshape(*orig_shape[:-3], *x.shape[-self.ndim + 1 :])
        return self.dropout(x, mask)
