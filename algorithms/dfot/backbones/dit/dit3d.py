from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from ..base_backbone import BaseBackbone
from .dit_base import DiTBase


class DiT3D(BaseBackbone):

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        if use_causal_mask:
            raise NotImplementedError(
                "Causal masking is not yet implemented for DiT3D backbone"
            )

        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            external_cond_dim,
            use_causal_mask,
        )

        hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size
        channels, resolution, *_ = x_shape
        assert (
            resolution % self.patch_size == 0
        ), "Resolution must be divisible by patch size."
        self.num_patches = (resolution // self.patch_size) ** 2
        out_channels = self.patch_size**2 * channels

        self.patch_embedder = PatchEmbed(
            img_size=resolution,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        self.dit_base = DiTBase(
            num_patches=self.num_patches,
            max_temporal_length=max_tokens,
            out_channels=out_channels,
            variant=cfg.variant,
            pos_emb_type=cfg.pos_emb_type,
            hidden_size=hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            learn_sigma=False,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        )
        self.initialize_weights()

    @property
    def in_channels(self) -> int:
        return self.x_shape[0]

    @staticmethod
    def _patch_embedder_init(embedder: PatchEmbed) -> None:
        # Initialize patch_embedder like nn.Linear (instead of nn.Conv2d):
        w = embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(embedder.proj.bias)

    def initialize_weights(self) -> None:
        self._patch_embedder_init(self.patch_embedder)

        # Initialize noise level embedding and external condition embedding MLPs:
        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patchified tensor of shape (B, num_patches, patch_size**2 * C)
        Returns:
            unpatchified tensor of shape (B, H, W, C)
        """
        return rearrange(
            x,
            "b (h w) (p q c) -> b (h p) (w q) c",
            h=int(self.num_patches**0.5),
            p=self.patch_size,
            q=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_batch_size = x.shape[0]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.patch_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)

        emb = self.noise_level_pos_embedding(noise_levels)
        import pdb; pdb.set_trace()
        if external_cond is not None:
            emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)
        emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)

        x = self.dit_base(x, emb)  # (B, N, C)
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B * T, H, W, C)
        x = rearrange(
            x, "(b t) h w c -> b t c h w", b=input_batch_size
        )  # (B, T, C, H, W)
        return x
