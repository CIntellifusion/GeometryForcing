"""
Adapted from Open-Sora-Plan-v1.2.0
https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/causal_vae/modeling_causalvae.py
"""

from typing import Tuple, Optional, Literal
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from utils.ckpt_utils import (
    is_wandb_run_path,
    is_hf_path,
    wandb_to_local_path,
    download_pretrained as hf_to_local_path,
)
from ..common.modules import Normalize, nonlinearity
from ..common.modules.utils import (
    resolve_str_to_module as _resolve_str_to_module,
    Module,
)
from ..common.distribution import DiagonalGaussianDistribution
from ..common.base_vae import VAE


class Encoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: Module = "Conv2d",
        conv_out: Module = "PaddedConv3D",
        attention: Module = "AttnBlock",
        resnet_blocks: Tuple[Module] = (
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock3D",
        ),
        spatial_downsample: Tuple[Module] = (
            "Downsample",
            "Downsample",
            "Downsample",
            "",
        ),
        temporal_downsample: Tuple[Module] = ("", "", "TimeDownsampleRes2x", ""),
        mid_resnet: Module = "ResnetBlock3D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        double_z: bool = True,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        assert len(resnet_blocks) == len(hidden_size_mult), print(
            hidden_size_mult, resnet_blocks
        )
        resolve_str_to_module = partial(_resolve_str_to_module, is_causal=is_causal)
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        # ---- In ----
        self.conv_in = resolve_str_to_module(conv_in)(
            3, hidden_size, kernel_size=3, stride=1, padding=1
        )

        # ---- Downsample ----
        curr_res = resolution
        in_ch_mult = (1,) + tuple(hidden_size_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden_size * in_ch_mult[i_level]
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    resolve_str_to_module(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(resolve_str_to_module(attention)(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if spatial_downsample[i_level]:
                down.downsample = resolve_str_to_module(spatial_downsample[i_level])(
                    block_in, block_in
                )
                curr_res = curr_res // 2
            if temporal_downsample[i_level]:
                down.time_downsample = resolve_str_to_module(
                    temporal_downsample[i_level]
                )(block_in, block_in)
            self.down.append(down)

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = resolve_str_to_module(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = resolve_str_to_module(attention)(block_in)
        self.mid.block_2 = resolve_str_to_module(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        # ---- Out ----
        self.norm_out = Normalize(block_in)
        self.conv_out = resolve_str_to_module(conv_out)(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))
            if hasattr(self.down[i_level], "time_downsample"):
                hs_down = self.down[i_level].time_downsample(hs[-1])
                hs.append(hs_down)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: Module = "Conv2d",
        conv_out: Module = "PaddedConv3D",
        attention: Module = "AttnBlock",
        resnet_blocks: Tuple[Module] = (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        spatial_upsample: Tuple[Module] = (
            "",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
        ),
        temporal_upsample: Tuple[Module] = ("", "", "", "TimeUpsampleRes2x"),
        mid_resnet: Module = "ResnetBlock3D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        is_causal: bool = True,
    ):
        super().__init__()
        resolve_str_to_module = partial(_resolve_str_to_module, is_causal=is_causal)
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        # ---- In ----
        block_in = hidden_size * hidden_size_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = resolve_str_to_module(conv_in)(
            z_channels, block_in, kernel_size=3, padding=1
        )

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = resolve_str_to_module(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = resolve_str_to_module(attention)(block_in)
        self.mid.block_2 = resolve_str_to_module(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # ---- Upsample ----
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    resolve_str_to_module(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(resolve_str_to_module(attention)(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if spatial_upsample[i_level]:
                upsample_kwargs = {}
                if spatial_upsample[i_level] == "Spatial2xTime2x3DUpsample":
                    upsample_kwargs["is_first"] = (
                        i_level == len(range(self.num_resolutions)) - 1
                    )
                up.upsample = resolve_str_to_module(spatial_upsample[i_level])(
                    block_in, block_in, **upsample_kwargs
                )
                curr_res = curr_res * 2
            if temporal_upsample[i_level]:
                up.time_upsample = resolve_str_to_module(temporal_upsample[i_level])(
                    block_in, block_in
                )
            self.up.insert(0, up)

        # ---- Out ----
        self.norm_out = Normalize(block_in)
        self.conv_out = resolve_str_to_module(conv_out)(
            block_in, 3, kernel_size=3, padding=1
        )

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
            if hasattr(self.up[i_level], "time_upsample"):
                h = self.up[i_level].time_upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VideoVAE(VAE):
    """
    Generic Video VAE model with two options: causal and non-causal
    (Temporal compression factor of `f_t = 2 ** num_temporal_downsample`).
    - Causal: Latents are only dependent on past frames, requires the input to be of length `f_t * k + 1`, which will be encoded to `k + 1` latents. Maps the first frame to the first latent. Causal model allows input to be f_t * l + 1, where 0 <= l <= k.
    - Noncausal: Latents are dependent on both past and future frames, requires the input to be of length `f_t * k`, which will be encoded to `k` latents. Maps the first frame to the first latent. Noncausal model allows input to be exactly f_t * k.
    TODO: Implement encoding and decoding with tiling for very long videos.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        z_channels: int = 4,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (),
        dropout: float = 0.0,
        resolution: int = 256,
        temporal_length: int = 17,
        double_z: bool = True,
        embed_dim: int = 4,
        num_res_blocks: int = 2,
        q_conv: Module = "PaddedConv3D",
        encoder_conv_in: Module = "Conv2d",
        encoder_conv_out: Module = "PaddedConv3D",
        encoder_attention: Module = "AttnBlock3D",
        encoder_resnet_blocks: Tuple[Module] = (
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        encoder_spatial_downsample: Tuple[Module] = (
            "Downsample",
            "Spatial2xTime2x3DDownsample",
            "Spatial2xTime2x3DDownsample",
            "",
        ),
        encoder_temporal_downsample: Tuple[Module] = (
            "",
            "",
            "",
            "",
        ),
        encoder_mid_resnet: Module = "ResnetBlock3D",
        decoder_conv_in: Module = "PaddedConv3D",
        decoder_conv_out: Module = "PaddedConv3D",
        decoder_attention: Module = "AttnBlock3D",
        decoder_resnet_blocks: Tuple[Module] = (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        decoder_spatial_upsample: Tuple[Module] = (
            "",
            "SpatialUpsample2x",
            "Spatial2xTime2x3DUpsample",
            "Spatial2xTime2x3DUpsample",
        ),
        decoder_temporal_upsample: Tuple[Module] = (
            "",
            "",
            "",
            "",
        ),
        decoder_mid_resnet: Module = "ResnetBlock3D",
        use_quant_layer: bool = True,
        is_causal: bool = True,
        first_padding_mode: Literal["zero", "same"] = "same",
    ) -> None:
        super().__init__()

        self.is_causal = is_causal
        self.temporal_pixel_length = temporal_length
        self.temporal_downsampling_factor = 2 ** (
            len([d for d in encoder_spatial_downsample if "Time" in d])
            + len([d for d in encoder_temporal_downsample if d != ""])
        )
        if is_causal:
            assert (
                self.temporal_pixel_length % self.temporal_downsampling_factor == 1
            ), f"For causal model, temporal length must be {self.temporal_downsampling_factor} * k + 1"
        else:
            assert (
                self.temporal_pixel_length % self.temporal_downsampling_factor == 0
            ), f"For non-causal model, temporal length must be {self.temporal_downsampling_factor} * k"
        self.temporal_latent_length = (
            self.temporal_pixel_length // self.temporal_downsampling_factor
            + (1 if is_causal else 0)
        )

        self.use_quant_layer = use_quant_layer
        self.first_padding_mode = first_padding_mode

        self.encoder = Encoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=encoder_conv_in,
            conv_out=encoder_conv_out,
            attention=encoder_attention,
            resnet_blocks=encoder_resnet_blocks,
            spatial_downsample=encoder_spatial_downsample,
            temporal_downsample=encoder_temporal_downsample,
            mid_resnet=encoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            double_z=double_z,
            is_causal=is_causal,
        )

        self.decoder = Decoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=decoder_conv_in,
            conv_out=decoder_conv_out,
            attention=decoder_attention,
            resnet_blocks=decoder_resnet_blocks,
            spatial_upsample=decoder_spatial_upsample,
            temporal_upsample=decoder_temporal_upsample,
            mid_resnet=decoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            is_causal=is_causal,
        )
        if self.use_quant_layer:
            quant_conv_cls = _resolve_str_to_module(q_conv, is_causal)
            self.quant_conv = quant_conv_cls(2 * z_channels, 2 * embed_dim, 1)
            self.post_quant_conv = quant_conv_cls(embed_dim, z_channels, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        if self.use_quant_layer:
            h = self.quant_conv(h)
        return h

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encode a batch of videos to a batch of DiagonalGaussianDistributions.
        """
        batch_size, _, temporal_length = x.shape[:3]
        if self.is_causal:
            assert (
                temporal_length <= self.temporal_pixel_length
                and temporal_length % self.temporal_downsampling_factor == 1
            ), f"Temporal length must be {self.temporal_downsampling_factor} * k + 1 where 0 <= k <= {self.temporal_latent_length - 1}, got {temporal_length}"
        else:
            if temporal_length % self.temporal_pixel_length != 0:
                pad = repeat(
                    (
                        x[:, :, :1]
                        if self.first_padding_mode == "same"
                        else torch.zeros_like(x[:, :, :1])
                    ),
                    "b c 1 h w -> b c t h w",
                    t=self.temporal_pixel_length
                    - temporal_length % self.temporal_pixel_length,
                )
                x = torch.cat([pad, x], dim=2)
            x = rearrange(
                x, "b c (m t) h w -> (b m) c t h w", t=self.temporal_pixel_length
            )
        h = self._encode(x)
        if h.shape[0] != batch_size:
            h = rearrange(
                h,
                "(b m) c t h w -> b c (m t) h w",
                b=batch_size,
            )
        return DiagonalGaussianDistribution(h)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_quant_layer:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode(
        self, z: torch.Tensor, desired_length: Optional[int] = None
    ) -> torch.Tensor:
        batch_size, _, temporal_latent_length = z.shape[:3]

        if not self.is_causal:
            assert (
                temporal_latent_length % self.temporal_latent_length == 0
            ), f"Temporal latent length must be a multiple of {self.temporal_latent_length}, got {temporal_latent_length}"
            z = rearrange(
                z,
                "b c (m t) h w -> (b m) c t h w",
                t=self.temporal_latent_length,
            )
        dec = self._decode(z)
        if dec.shape[0] != batch_size:
            dec = rearrange(
                dec,
                "(b m) c t h w -> b c (m t) h w",
                b=batch_size,
            )
        if desired_length is not None:
            dec = dec[:, :, -desired_length:]
            assert (
                dec.shape[2] == desired_length
            ), f"Desired length {desired_length} does not match decoded length {dec.shape[2]}"
        return dec

    def forward(
        self, sample: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(sample)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, desired_length=sample.shape[2])
        return dec, posterior

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "VideoVAE":
        if is_wandb_run_path(path):
            path = wandb_to_local_path(path)
        elif is_hf_path(path):
            path = hf_to_local_path(path)
        checkpoint = torch.load(path, map_location="cpu")
        model_cfg = checkpoint["model_cfg"]
        for key, value in model_cfg.items():
            if isinstance(value, list):
                model_cfg[key] = tuple(value)
        model = cls(**model_cfg)

        if (
            len(checkpoint["optimizer_states"]) > 0
            and "ema" in checkpoint["optimizer_states"][0]
        ):
            state_dict = dict(
                zip(
                    [name for name, _ in model.named_parameters()],
                    checkpoint["optimizer_states"][0]["ema"],
                )
            )
        else:
            state_dict = {
                key.replace("vae.", ""): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("vae.")
            }
        model.load_state_dict(state_dict)
        return model
