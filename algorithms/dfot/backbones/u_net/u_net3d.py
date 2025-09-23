from functools import partial
from typing import Optional
import torch
from torch import nn
from einops import rearrange
from omegaconf import DictConfig
from rotary_embedding_torch import RotaryEmbedding
from ..base_backbone import BaseBackbone
from .u_net_blocks import (
    ResnetBlock,
    Downsample,
    Upsample,
    UnetSpatialAttentionBlock,
    UnetTemporalAttentionBlock,
    UnetSequential,
)


class IdentityWithExtraArgs(nn.Identity):
    def forward(self, x, *args, **kwargs):
        return x


class Unet3D(BaseBackbone):

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            external_cond_dim,
            use_causal_mask,
        )

        dim = cfg.network_size
        init_dim = dim
        channels, resolution, *_ = x_shape
        out_dim = channels
        num_res_blocks = cfg.num_res_blocks
        resnet_block_groups = cfg.resnet_block_groups
        dim_mults = cfg.dim_mults
        attn_resolutions = [resolution // res for res in list(cfg.attn_resolutions)]
        attn_dim_head = cfg.attn_dim_head
        attn_heads = cfg.attn_heads
        use_linear_attn = cfg.use_linear_attn
        use_init_temporal_attn = cfg.use_init_temporal_attn
        init_kernel_size = cfg.init_kernel_size
        dropout = cfg.dropout

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        emb_dim = self.noise_level_emb_dim + self.external_cond_emb_dim

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels,
            init_dim,
            kernel_size=(1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.rotary_time_pos_embedding = RotaryEmbedding(dim=attn_dim_head)

        self.init_temporal_attn = (
            UnetTemporalAttentionBlock(
                dim=init_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                is_causal=use_causal_mask,
                rotary_emb=self.rotary_time_pos_embedding,
            )
            if use_init_temporal_attn
            else IdentityWithExtraArgs()
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        block_klass_noise = partial(
            ResnetBlock, groups=resnet_block_groups, emb_dim=emb_dim
        )
        spatial_attn_klass = partial(
            UnetSpatialAttentionBlock, heads=attn_heads, dim_head=attn_dim_head
        )
        temporal_attn_klass = partial(
            UnetTemporalAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
            is_causal=use_causal_mask,
            rotary_emb=self.rotary_time_pos_embedding,
        )

        curr_resolution = 1

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in attn_resolutions

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        UnetSequential(
                            block_klass_noise(dim_in, dim_out),
                            *(
                                block_klass_noise(dim_out, dim_out)
                                for _ in range(num_res_blocks - 1)
                            ),
                            (
                                spatial_attn_klass(
                                    dim_out,
                                    use_linear=use_linear_attn and not is_last,
                                )
                                if use_attn
                                else nn.Identity()
                            ),
                            temporal_attn_klass(dim_out) if use_attn else nn.Identity(),
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            curr_resolution *= 2 if not is_last else 1

        self.mid_block = UnetSequential(
            block_klass_noise(mid_dim, mid_dim),
            spatial_attn_klass(mid_dim),
            temporal_attn_klass(mid_dim),
            block_klass_noise(mid_dim, mid_dim),
        )

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in attn_resolutions

            self.up_blocks.append(
                UnetSequential(
                    block_klass_noise(dim_out * 2, dim_in),
                    *(
                        block_klass_noise(dim_in, dim_in)
                        for _ in range(num_res_blocks - 1)
                    ),
                    (
                        spatial_attn_klass(
                            dim_in, use_linear=use_linear_attn and idx > 0
                        )
                        if use_attn
                        else nn.Identity()
                    ),
                    temporal_attn_klass(dim_in) if use_attn else nn.Identity(),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                )
            )

            curr_resolution //= 2 if not is_last else 1

        self.out = nn.Sequential(block_klass(dim * 2, dim), nn.Conv3d(dim, out_dim, 1))

    @property
    def noise_level_emb_dim(self):
        return self.cfg.network_size * 4

    @property
    def external_cond_emb_dim(self):
        return self.cfg.network_size * 2 if self.external_cond_dim else 0

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ):
        x = rearrange(x, "b t c h w -> b c t h w").contiguous()

        emb = self.noise_level_pos_embedding(noise_levels)
        if self.external_cond_embedding is not None:
            if external_cond is None:
                raise ValueError("External condition is required, but not provided.")
            external_cond_emb = self.external_cond_embedding(
                external_cond, external_cond_mask
            )
            emb = torch.cat([emb, external_cond_emb], dim=-1)

        x = self.init_conv(x)
        x = self.init_temporal_attn(x)
        h = x.clone()

        hs = []

        for block, downsample in self.down_blocks:
            h = block(h, emb)
            hs.append(h)
            h = downsample(h)

        h = self.mid_block(h, emb)

        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, emb)

        h = torch.cat([h, x], dim=1)
        x = self.out(h)
        x = rearrange(x, " b c t h w -> b t c h w")
        return x
