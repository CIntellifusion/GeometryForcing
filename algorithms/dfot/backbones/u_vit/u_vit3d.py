from typing import Optional, Tuple
from functools import partial
from omegaconf import DictConfig
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from ..base_backbone import BaseBackbone
from ..modules.embeddings import RotaryEmbedding3D
from ..dit.dit_base import SinusoidalPositionalEmbedding
from .u_vit_blocks import (
    EmbedInput,
    ProjectOutput,
    ResBlock,
    TransformerBlock,
    Upsample,
    Downsample,
    AxialRotaryEmbedding,
)


class UViT3D(BaseBackbone):
    """
    A U-ViT backbone from the following papers:
    - Simple diffusion: End-to-end diffusion for high resolution images (https://arxiv.org/abs/2301.11093)
    - Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion (https://arxiv.org/abs/2410.19324)
    - We more closely follow SiD2's Residual U-ViT, where blockwise skip-connections are removed, and only a single skip-connection is used per downsampling operation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        # ------------------------------- Configuration --------------------------------
        # these configurations closely follow the notation in the SiD2 paper
        channels = cfg.channels
        self.emb_dim = cfg.emb_channels
        patch_size = cfg.patch_size
        block_types = cfg.block_types
        block_dropouts = cfg.block_dropouts
        num_updown_blocks = cfg.num_updown_blocks
        num_mid_blocks = cfg.num_mid_blocks
        num_heads = cfg.num_heads
        self.pos_emb_type = cfg.pos_emb_type
        self.num_levels = len(channels)
        resolution = x_shape[-1]
        self.is_transformers = [block_type != "ResBlock" for block_type in block_types]
        self.use_checkpointing = list(cfg.use_checkpointing)
        self.temporal_length = max_tokens

        # ------------------------------ Initialization ---------------------------------

        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            external_cond_dim,
            use_causal_mask,
        )

        # -------------- Initial downsampling and final upsampling layers --------------
        # This enables avoiding high-resolution feature maps and speeds up the network
        self.embed_input = EmbedInput(
            in_channels=x_shape[0],
            dim=channels[0],
            patch_size=patch_size,
        )
        self.project_output = ProjectOutput(
            dim=channels[0],
            out_channels=x_shape[0],
            patch_size=patch_size,
        )

        # --------------------------- Positional embeddings ----------------------------
        # We use a 1D learnable positional embedding or RoPE for every level with transformers
        assert self.pos_emb_type in [
            "learned_1d",
            "rope",
        ], f"Positional embedding type {self.pos_emb_type} not supported."

        self.pos_embs = nn.ModuleDict({})
        for i_level, channel in enumerate(channels):
            if not self.is_transformers[i_level]:
                continue
            pos_emb_cls, dim = None, None
            if self.pos_emb_type == "rope":
                pos_emb_cls = (
                    RotaryEmbedding3D
                    if block_types[i_level] == "TransformerBlock"
                    else AxialRotaryEmbedding
                )
                dim = channel // num_heads
            else:
                pos_emb_cls = partial(SinusoidalPositionalEmbedding, learnable=True)
                dim = channel
            level_resolution = resolution // patch_size // (2**i_level)
            self.pos_embs[f"{i_level}"] = pos_emb_cls(
                dim,
                (self.temporal_length, level_resolution, level_resolution),
            )
        # import pdb; pdb.set_trace() 
        def _rope_kwargs(i_level: int):
            return (
                {"rope": self.pos_embs[f"{i_level}"]}
                if self.pos_emb_type == "rope" and self.is_transformers[i_level]
                else {}
            )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_type_to_cls = {
            "ResBlock": partial(ResBlock, emb_dim=self.emb_dim),
            "TransformerBlock": partial(
                TransformerBlock, emb_dim=self.emb_dim, heads=num_heads
            ),
            "AxialTransformerBlock": partial(
                TransformerBlock,
                emb_dim=self.emb_dim,
                heads=num_heads,
                use_axial=True,
                ax1_len=self.temporal_length,
            ),
        }

        # ---------------------------- Down-sampling blocks ----------------------------
        for i_level, (num_blocks, ch, block_type, block_dropout) in enumerate(
            zip(
                num_updown_blocks,
                channels[:-1],
                block_types[:-1],
                block_dropouts[:-1],
            )
        ):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        block_type_to_cls[block_type](
                            ch, dropout=block_dropout, **_rope_kwargs(i_level)
                        )
                        for _ in range(num_blocks)
                    ]
                    + [Downsample(ch, channels[i_level + 1])],
                )
            )

        # ------------------------------ Middle blocks ---------------------------------
        self.mid_blocks = nn.ModuleList(
            [
                block_type_to_cls[block_types[-1]](
                    channels[-1],
                    dropout=block_dropouts[-1],
                    **_rope_kwargs(self.num_levels - 1),
                )
                for _ in range(num_mid_blocks)
            ]
        )

        # ---------------------------- Up-sampling blocks ------------------------------
        for _i_level, (num_blocks, ch, block_type, block_dropout) in enumerate(
            zip(
                reversed(num_updown_blocks),
                reversed(channels[:-1]),
                reversed(block_types[:-1]),
                reversed(block_dropouts[:-1]),
            )
        ):
            i_level = self.num_levels - 2 - _i_level
            self.up_blocks.append(
                nn.ModuleList(
                    [Upsample(channels[i_level + 1], ch)]
                    + [
                        block_type_to_cls[block_type](
                            ch,
                            dropout=block_dropout,
                            **_rope_kwargs(i_level),
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.emb_dim

    @property
    def external_cond_emb_dim(self) -> int:
        return self.emb_dim

    def _rearrange_and_add_pos_emb_if_transformer(
        self, x: Tensor, emb: Tensor, i_level: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Rearrange input tensor to be compatible with transformer blocks, if necessary.
        Args:
            x: Input tensor of shape (B * T, C, H, W).
            emb: Embedding tensor of shape (B * T, C).
            i_level: Index of the current level.
        Returns:
            x and emb of shape (B, T * H * W, C).
        """
        is_transformer = self.is_transformers[i_level]
        if not is_transformer:
            return x, emb
        h, w = x.shape[-2:]
        x = rearrange(x, "(b t) c h w -> b (t h w) c", t=self.temporal_length)
        if self.pos_emb_type == "learned_1d":
            x = self.pos_embs[f"{i_level}"](x)
        emb = repeat(emb, "(b t) c -> b (t h w) c", t=self.temporal_length, h=h, w=w)
        return x, emb

    def _unrearrange_if_transformer(self, x: Tensor, i_level: int) -> Tensor:
        """
        Rearrange input tensor back to its original shape, if necessary.
        Args:
            x: Input tensor of shape (B, T * H * W, C).
            i_level: Index of the current level.
        Returns:
            x of shape (B, T, C, H, W).
        """
        is_transformer = self.is_transformers[i_level]
        if not is_transformer:
            return x
        h = w = int((x.shape[1] / self.temporal_length) ** 0.5)
        x = rearrange(x, "b (t h w) c -> (b t) c h w", t=self.temporal_length, h=h, w=w)
        return x

    @staticmethod
    def _checkpointed_forward(
        module: nn.Module, *args, use_checkpointing: bool = False
    ) -> Tensor:
        if use_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _run_level_blocks(
        self, x: Tensor, emb: Tensor, i_level: int, is_up: bool = False
    ) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level.
        Gradient checkpointing is used optionally, with self.checkpoints[i_level] segments.
        """
        use_checkpointing = self.use_checkpointing[i_level]

        blocks = (
            self.mid_blocks
            if i_level == self.num_levels - 1
            else (
                self.up_blocks[self.num_levels - 2 - i_level][1:]
                if is_up
                else self.down_blocks[i_level][:-1]
            )
        )

        for block in blocks:
            x = self._checkpointed_forward(
                block,
                x,
                emb,
                use_checkpointing=use_checkpointing,
            )
        return x

    def _run_level(
        self, x: Tensor, emb: Tensor, i_level: int, is_up: bool = False
    ) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level, accompanied by reshaping operations before and after.
        """
        # print(f"i_level: {i_level}, x.shape: {x.shape}, emb.shape: {emb.shape}")
        x, emb = self._rearrange_and_add_pos_emb_if_transformer(x, emb, i_level)
        # print(f"i_level: {i_level}, x.shape: {x.shape}, emb.shape: {emb.shape}")
        x = self._run_level_blocks(x, emb, i_level, is_up)
        # print(f"i_level: {i_level}, x.shape: {x.shape}, emb.shape: {emb.shape}")
        x = self._unrearrange_if_transformer(x, i_level)
        # print(f"i_level: {i_level}, x.shape: {x.shape}, emb.shape: {emb.shape}")
        return x

    def forward(
        self,
        x: Tensor,
        noise_levels: Tensor,
        external_cond: Optional[Tensor] = None,
        external_cond_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the U-ViT backbone.
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            noise_levels: Noise level tensor of shape (B, T).
            external_cond: External conditioning tensor of shape (B, T, C).
        Returns:
            Output tensor of shape (B, T, C, H, W).
        """
        assert (
            x.shape[1] == self.temporal_length
        ), f"Temporal length of U-ViT is set to {self.temporal_length}, but input has temporal length {x.shape[1]}."
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed_input(x)

        # Embeddings
        emb = self.noise_level_pos_embedding(noise_levels)
        if external_cond is not None:
            emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)
        emb = rearrange(emb, "b t c -> (b t) c")

        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling

        # Down-sampling blocks
        for i_level, down_block in enumerate(
            self.down_blocks,
        ):
            x = self._run_level(x, emb, i_level)
            hs_before.append(x)
            x = down_block[-1](x)
            hs_after.append(x)

        # Middle blocks
        x = self._run_level(x, emb, self.num_levels - 1)

        # import pdb; pdb.set_trace()
        # Up-sampling blocks
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, emb, i_level, is_up=True)

        x = self.project_output(x)
        return rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)
