from typing import Optional, Tuple
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange
from ..modules.normalization import (
    RMSNorm as Normalize,
)
from ..modules.embeddings import (
    RotaryEmbedding1D,
    RotaryEmbedding2D,
    RotaryEmbeddingND,
)
from ..modules.zero_module import zero_module


class EmbedInput(nn.Module):
    """
    Initial downsampling layer for U-ViT.
    One shall replace this with 5/3 DWT, which is fully invertible and may slightly improve performance, according to the Simple Diffusion paper.
    """

    def __init__(self, in_channels: int, dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x


class ProjectOutput(nn.Module):
    """
    Final upsampling layer for U-ViT.
    One shall replace this with IDWT, which is an inverse operation of DWT.
    """

    def __init__(self, dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.proj = zero_module(
            nn.ConvTranspose2d(
                dim, out_channels, kernel_size=patch_size, stride=patch_size
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x


# pylint: disable-next=invalid-name
def NormalizeWithBias(num_channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    """
    Standard ResNet block.
    """

    def __init__(self, channels: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        assert dropout == 0.0, "Dropout is not supported in ResBlock."
        self.emb_layer = nn.Conv2d(emb_dim, channels * 2, kernel_size=(1, 1))
        self.in_layers = nn.Sequential(
            NormalizeWithBias(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.out_norm = NormalizeWithBias(channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            zero_module(
                nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
            ),
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the ResNet block.
        Args:
            x: Input tensor of shape (B, C, H, W).
            emb: Embedding tensor of shape (B, C) or (B, C, H, W).
        Returns:
            Output tensor of shape (B, C, H, W).
        """
        h = self.in_layers(x)
        emb_out = self.emb_layer(emb if emb.dim() == 4 else emb[:, :, None, None])
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_rest(h)
        return x + h


class NormalizeWithCond(nn.Module):
    """
    Conditioning block for U-ViT, that injects external conditions into the network using FiLM.
    """

    def __init__(self, dim: int, emb_dim: int):
        super().__init__()
        self.emb_layer = nn.Linear(emb_dim, dim * 2)
        self.norm = Normalize(dim)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the conditioning block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """
        scale, shift = self.emb_layer(emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AttentionBlock(nn.Module):
    """
    Simple Attention block for axial attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        rope: Optional[RotaryEmbeddingND] = None,
    ):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.rope = rope
        self.norm = NormalizeWithCond(dim, emb_dim)
        self.proj = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm, self.k_norm = Normalize(dim_head), Normalize(dim_head)
        self.out = zero_module(nn.Linear(dim, dim, bias=False))

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the attention block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """
        x = self.norm(x, emb)
        qkv = self.proj(x)
        q, k, v = rearrange(
            qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        ).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # pylint: disable-next=not-callable
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        return x + self.out(x)


class AxialRotaryEmbedding(nn.Module):
    """
    Axial rotary embedding for axial attention.
    Composed of two rotary embeddings for each axis.
    """

    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int] | Tuple[int, int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        """
        If len(sizes) == 2, each axis corresponds to each dimension.
        If len(sizes) == 3, the first dimension corresponds to the first axis, and the rest corresponds to the second axis.
        This enables to be compatible with the initializations of `.embeddings.RotaryEmbedding2D` and `.embeddings.RotaryEmbedding3D`.
        """
        super().__init__()
        self.ax1 = RotaryEmbedding1D(dim, sizes[0], theta, flatten)
        self.ax2 = (
            RotaryEmbedding1D(dim, sizes[1], theta, flatten)
            if len(sizes) == 2
            else RotaryEmbedding2D(dim, sizes[1:], theta, flatten)
        )


class TransformerBlock(nn.Module):
    """
    Efficient transformer block with parallel attention + MLP and Query-Key normalization,
    following https://arxiv.org/abs/2302.05442

    Supports axial attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        dropout: float,
        use_axial: bool = False,
        ax1_len: Optional[int] = None,
        rope: Optional[AxialRotaryEmbedding | RotaryEmbeddingND] = None,
    ):
        super().__init__()
        self.rope = rope.ax2 if (rope is not None and use_axial) else rope
        self.norm = NormalizeWithCond(dim, emb_dim)

        self.heads = heads
        dim_head = dim // heads
        self.use_axial = use_axial
        self.ax1_len = ax1_len
        mlp_dim = 4 * dim
        self.fused_dims = (3 * dim, mlp_dim)
        self.fused_attn_mlp_proj = nn.Linear(dim, sum(self.fused_dims), bias=True)
        self.q_norm, self.k_norm = Normalize(dim_head), Normalize(dim_head)

        self.attn_out = zero_module(nn.Linear(dim, dim, bias=True))

        if self.use_axial:
            self.another_attn = AttentionBlock(
                dim, heads, emb_dim, rope.ax1 if rope is not None else None
            )

        self.mlp_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(mlp_dim, dim, bias=True)),
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the transformer block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """
        if self.use_axial:
            x, emb = map(
                lambda y: rearrange(
                    y, "b (ax1 ax2) d -> (b ax1) ax2 d", ax1=self.ax1_len
                ),
                (x, emb),
            )
        _x = x
        x = self.norm(x, emb)
        qkv, mlp_h = self.fused_attn_mlp_proj(x).split(self.fused_dims, dim=-1)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # pylint: disable-next=not-callable
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = _x + self.attn_out(x)

        if self.use_axial:
            ax2_len = x.shape[1]
            x, emb = map(
                lambda y: rearrange(
                    y, "(b ax1) ax2 d -> (b ax2) ax1 d", ax1=self.ax1_len
                ),
                (x, emb),
            )
            x = self.another_attn(x, emb)
            x = rearrange(x, "(b ax2) ax1 d -> (b ax1) ax2 d", ax2=ax2_len)

        x = x + self.mlp_out(mlp_h)

        if self.use_axial:
            x = rearrange(x, "(b ax1) ax2 d -> b (ax1 ax2) d", ax1=self.ax1_len)
        return x


class Downsample(nn.Module):
    """
    Downsample block for U-ViT.
    Done by average pooling + conv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # pylint: disable-next=not-callable
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """
    Upsample block for U-ViT.
    Done by conv + nearest neighbor upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x
