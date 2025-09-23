from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange, repeat
from ..modules.embeddings import (
    RandomDropoutPatchEmbed,
    RandomEmbeddingDropout,
)
from .dit3d import DiT3D


class DiT3DPose(DiT3D):

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        self.conditioning_modeling = cfg.conditioning.modeling
        self.conditioning_type = cfg.conditioning.type
        self.conditioning_dropout = cfg.external_cond_dropout
        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            cfg.conditioning.dim,
            use_causal_mask,
        )

    @property
    def in_channels(self) -> int:
        return (
            self.x_shape[0] + self.external_cond_dim
            if self.conditioning_modeling == "concat"
            else self.x_shape[0]
        )

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        if self.conditioning_type == "global":
            return super()._build_external_cond_embedding()
        match self.conditioning_modeling:
            case "concat":
                return RandomEmbeddingDropout(
                    p=self.conditioning_dropout,
                )
            case "film":
                return RandomDropoutPatchEmbed(
                    dropout_prob=self.conditioning_dropout,
                    img_size=self.x_shape[1],
                    patch_size=self.cfg.patch_size,
                    in_chans=self.external_cond_dim,
                    embed_dim=self.external_cond_emb_dim,
                    bias=True,
                )
            case _:
                raise ValueError(
                    f"Unknown external condition modeling: {self.conditioning_modeling}"
                )

    def initialize_weights(self) -> None:
        super().initialize_weights()
        if self.conditioning_type != "global" and self.conditioning_modeling == "film":
            self._patch_embedder_init(self.external_cond_embedding.patch_embedder)

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            external_cond is not None
        ), "External condition (camera pose) is required for DiT3DPose model."
        input_batch_size = x.shape[0]
        external_cond_emb = self.external_cond_embedding(
            external_cond, external_cond_mask
        )
        if self.conditioning_modeling == "concat":
            x = torch.cat(
                [x, external_cond_emb],
                dim=2,
            )
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.patch_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)

        emb = self.noise_level_pos_embedding(noise_levels)
        emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)

        if self.conditioning_modeling == "film":
            if self.conditioning_type == "global":
                external_cond_emb = repeat(
                    external_cond_emb, "b t c -> b (t p) c", p=self.num_patches
                )
            else:
                external_cond_emb = rearrange(external_cond_emb, "b t p c -> b (t p) c")
            emb = emb + external_cond_emb

        x = self.dit_base(x, emb)  # (B, N, C)
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B * T, H, W, C)
        x = rearrange(
            x, "(b t) h w c -> b t c h w", b=input_batch_size
        )  # (B, T, C, H, W)
        return x
