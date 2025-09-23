from pathlib import Path
from omegaconf import DictConfig
import torch
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from utils.storage_utils import safe_torch_save
from utils.logging_utils import log_video
from ..common.distribution import DiagonalGaussianDistribution
from .trainer import ImageVAE


class ImageVAEPreprocessor(BasePytorchAlgo):
    """
    An algorithm for preprocessing videos to latents using a pretrained ImageVAE model.
    """

    def __init__(self, cfg: DictConfig):
        self.pretrained_path = cfg.pretrained_path
        self.pretrained_kwargs = cfg.pretrained_kwargs
        self.use_fp16 = cfg.precision == "16-true"
        self.max_encode_length = cfg.max_encode_length
        self.max_decode_length = cfg.logging.max_video_length
        self.log_every_n_batch = cfg.logging.every_n_batch
        super().__init__(cfg)

    def _build_model(self):
        self.vae = ImageVAE.from_pretrained(
            path=self.pretrained_path,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            **self.pretrained_kwargs,
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError(
            "Training not implemented for VAEVideo. Only used for validation"
        )

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError(
            "Testing not implemented for VAEVideo. Only used for validation"
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        videos, latent_paths = batch
        latent_paths = [Path(path) for path in latent_paths]
        batch_size = videos.shape[0]
        videos = self._rearrange_and_normalize(videos)

        # Encode the video data into a latent space
        # always convert to float16 (as they will be saved as float16 tensors)
        latent_dist = self._encode_videos(videos)
        latents = latent_dist.sample().to(torch.float16)

        # just to see the progress in wandb
        if batch_idx % 100 == 0:
            self.log("dummy", 0.0)

        # log gt vs reconstructed video to wandb
        if batch_idx % self.log_every_n_batch == 0 and self.logger:
            videos = videos.detach().cpu()[: self.max_decode_length]
            reconstructed_videos = self.vae.decode(latents[: self.max_decode_length])
            reconstructed_videos = reconstructed_videos.detach().cpu()
            videos = self._rearrange_and_unnormalize(videos, batch_size)
            reconstructed_videos = self._rearrange_and_unnormalize(
                reconstructed_videos, batch_size
            )
            log_video(
                reconstructed_videos,
                videos,
                step=self.global_step,
                namespace="reconstruction_vis",
                logger=self.logger.experiment,
                captions=[
                    f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"
                    for p in latent_paths
                ],
            )

        # save the latent to disk
        latents_to_save = (
            rearrange(
                latents,
                "(b f) c h w -> b f c h w",
                b=batch_size,
            )
            .detach()
            .cpu()
        )
        for latent, latent_path in zip(latents_to_save, latent_paths):
            # should clone latent to avoid having large file size
            safe_torch_save(latent.clone(), latent_path)
        return None

    def _encode_videos(self, video: torch.Tensor) -> DiagonalGaussianDistribution:
        chunks = video.chunk(
            (len(video) + self.max_encode_length - 1) // self.max_encode_length, dim=0
        )
        latent_dist_list = []
        for chunk in chunks:
            latent_dist_list.append(self.vae.encode(chunk))
        return DiagonalGaussianDistribution.cat(latent_dist_list)

    def _rearrange_and_normalize(self, videos: torch.Tensor) -> torch.Tensor:
        videos = rearrange(videos, "b f c h w -> (b f) c h w")
        videos = 2.0 * videos - 1.0
        return videos

    def _rearrange_and_unnormalize(
        self, videos: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        videos = 0.5 * videos + 0.5
        videos = rearrange(videos, "(b f) c h w -> b f c h w", b=batch_size)
        return videos
