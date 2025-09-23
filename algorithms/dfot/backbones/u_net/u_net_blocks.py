from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import repeat, rearrange, parse_shape
from rotary_embedding_torch import RotaryEmbedding
from ..modules.embeddings import (
    TimestepEmbedding,
    Timesteps,
)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        emb_dim: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps),
            nn.SiLU(),
            nn.Conv3d(dim, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=dim_out, eps=eps),
            nn.SiLU(),
            nn.Conv3d(dim_out, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )

        self.emb_layers = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim_out * 2),
            )
            if emb_dim is not None
            else None
        )

        self.skip_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None):
        h = self.in_layers(x)

        if self.emb_layers is not None:
            assert (
                emb is not None
            ), "Noise level embedding is required for this ResnetBlock"
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            emb = self.emb_layers(emb)
            emb = rearrange(emb, "b f c -> b c f 1 1")
            scale, shift = emb.chunk(2, dim=1)

            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layers(h)

        return self.skip_conv(x) + h


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(
            dim, dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        return self.conv(x)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 4,
        dim_head: int = 32,
        bias: bool = False,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(query_dim, self.inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(self.inner_dim, query_dim)

        # determine efficient attention configs for cuda and cpu

        self.cpu_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.MATH,
            SDPBackend.EFFICIENT_ATTENTION,
        ]

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major >= 8 and device_properties.minor == 0:
            # A100 GPU
            self.cuda_backends = [SDPBackend.FLASH_ATTENTION]
        else:
            # Non-A100 GPU
            self.cuda_backends = [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
            ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if is_causal and attn_mask is not None:
            is_causal = False
            attn_mask = attn_mask.tril()
        if attn_mask is not None:
            attn_mask = repeat(attn_mask, "b t1 t2 -> b h t1 t2", h=self.heads)

        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # Flash / Memory efficient attention leads to cuda errors for large batch sizes
        backends = (
            (
                [SDPBackend.MATH]
                if q.shape[0] >= 65536
                else (
                    self.cuda_backends
                    if attn_mask is None
                    else [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                )
            )
            if q.is_cuda
            else self.cpu_backends
        )
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        with sdpa_kernel(backends=backends):
            # pylint: disable=E1102
            hidden_states = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                is_causal=is_causal,
                attn_mask=attn_mask,
            )

        hidden_states = rearrange(hidden_states, "b h n d -> b n (h d)")
        hidden_states = hidden_states.to(q.dtype)

        # linear proj
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        use_linear: bool = False,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        if dim_head == -1:
            dim_head = dim // heads
        elif heads == -1:
            heads = dim // dim_head

        self.norm = nn.LayerNorm(dim)
        attn_klass = LinearAttention if use_linear else Attention
        self.attn = attn_klass(
            query_dim=dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb
        )

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        return x + self.attn(self.norm(x), is_causal=is_causal, attn_mask=attn_mask)


class LinearAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(query_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, query_dim)

        if rotary_emb is not None:
            raise NotImplementedError(
                "Rotary embeddings not implemented for linear attention"
            )

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if is_causal:
            raise NotImplementedError(
                "Causal masking not implemented for linear attention"
            )
        if attn_mask is not None:
            raise NotImplementedError(
                "Attention masking not implemented for linear attention"
            )
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h d n", h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h d n -> b n (h d)")
        return self.to_out(out)


class TemporalAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        is_causal: bool = True,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.attn_block = AttentionBlock(dim, heads, dim_head, rotary_emb=rotary_emb)
        self.time_pos_embedding = (
            nn.Sequential(
                Timesteps(dim),
                TimestepEmbedding(in_channels=dim, time_embed_dim=dim * 4, out_dim=dim),
            )
            if rotary_emb is None
            else None
        )
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        x ~ (b t c)
        attn_mask ~ (b t t)
        """

        if self.time_pos_embedding is not None:
            num_frames = x.shape[1]
            time_emb = self.time_pos_embedding(
                torch.arange(num_frames, device=x.device)
            )
            x = x + time_emb
        x = self.attn_block(x, is_causal=self.is_causal, attn_mask=attn_mask)
        return x


class EinopsWrapper(nn.Module):
    def __init__(self, from_shape: str, to_shape: str, module: nn.Module):
        super().__init__()
        self.module = module
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x: torch.Tensor, *args, **kwargs):
        axes_lengths = parse_shape(x, pattern=self.from_shape)
        x = rearrange(x, f"{self.from_shape} -> {self.to_shape}")
        x = self.module(x, *args, **kwargs)
        x = rearrange(x, f"{self.to_shape} -> {self.from_shape}", **axes_lengths)
        return x


def get_einops_wrapped_module(module, from_shape: str, to_shape: str):
    class WrappedModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.wrapper = EinopsWrapper(from_shape, to_shape, module(*args, **kwargs))

        def forward(self, x: torch.Tensor, *args, **kwargs):
            return self.wrapper(x, *args, **kwargs)

    return WrappedModule


UnetSpatialAttentionBlock = get_einops_wrapped_module(
    AttentionBlock, "b c t h w", "(b t) (h w) c"
)

_UnetTemporalAttentionBlock = get_einops_wrapped_module(
    TemporalAttentionBlock, "b c t h w", "(b h w) t c"
)


class UnetTemporalAttentionBlock(_UnetTemporalAttentionBlock):
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if attn_mask is not None:
            attn_mask = repeat(
                attn_mask, "b t1 t2 -> (b h w) t1 t2", h=x.shape[-2], w=x.shape[-1]
            )
        return super().forward(x, attn_mask)


class UnetSequential(torch.nn.Sequential):
    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ):
        for module in self:
            if isinstance(module, ResnetBlock):
                x = module(x, emb)
            else:
                x = module(x)
        return x
