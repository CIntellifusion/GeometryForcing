from typing import Any, Dict, Tuple, Optional
from itertools import accumulate
import random
from omegaconf import DictConfig, open_dict
import torch
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from algorithms.common.metrics.video import (
    VideoMetric,
    SharedVideoMetricModelRegistry,
)
from utils.distributed_utils import is_rank_zero, broadcast_from_zero
from utils.logging_utils import log_video
from ..common.losses import LPIPSWithDiscriminator3D, warmup
from .model import VideoVAE


class VideoVAETrainer(BasePytorchAlgo):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.lr = cfg.lr
        self.disc_start = cfg.loss.disc_start
        self.warmup_steps = cfg.training.warmup_steps
        self.gradient_clip_val = cfg.training.gradient_clip_val
        self.video_length_probs = list(accumulate(cfg.training.video_length_probs))
        assert self.video_length_probs[-1] == 1.0, "video_length_probs must sum to 1"
        self.video_lengths = cfg.training.video_lengths
        self.validation_video_lengths = cfg.validation.video_lengths
        self.num_logged_videos = [0] * len(self.validation_video_lengths)
        super().__init__(cfg)

    def _build_model(self):
        with open_dict(self.cfg):
            for key, value in self.cfg.model.items():
                if isinstance(value, list):
                    self.cfg.model[key] = tuple(value)
        self.vae = VideoVAE(**self.cfg.model)
        self.loss = LPIPSWithDiscriminator3D(**self.cfg.loss)
        self.metrics_registry = SharedVideoMetricModelRegistry()
        self.metrics = torch.nn.ModuleList(
            [
                VideoMetric(
                    registry=self.metrics_registry,
                    metric_types=self.cfg.logging.metrics,
                )
                for video_length in self.validation_video_lengths
            ]
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)

        new_state_dict = {}
        for key, value in self.state_dict().items():
            if key.startswith("metrics"):
                new_state_dict[key] = value
            else:
                new_state_dict[key] = checkpoint["state_dict"][key]
        checkpoint["state_dict"] = new_state_dict

        for state in checkpoint["optimizer_states"]:
            if "opt" in state:
                state = state["opt"]
            for pg in state["param_groups"]:
                pg["lr"] = self.cfg.lr

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # save model config to enable loading the model from checkpoint only
        checkpoint["model_cfg"] = self.cfg.model

    def _load_ema_weights_to_state_dict(self, checkpoint: dict) -> None:
        if (
            checkpoint.get("pretrained_ema", False)
            and len(checkpoint["optimizer_states"]) == 0
        ):
            # NOTE: for lightweight EMA-only ckpts for releasing pretrained models,
            # we already have EMA weights in the state_dict
            return

        vae_ema_weights = checkpoint["optimizer_states"][0]["ema"]
        vae_parameter_keys = ["vae." + k for k, _ in self.vae.named_parameters()]
        assert len(vae_ema_weights) == len(vae_parameter_keys)
        for key, weight in zip(vae_parameter_keys, vae_ema_weights):
            checkpoint["state_dict"][key] = weight

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.automatic_optimization = False
        optimizer_vae = torch.optim.Adam(
            self.vae.parameters(),
            lr=self.lr,
            betas=self.cfg.training.optimizer_beta,
        )
        optimizer_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=self.lr,
            betas=self.cfg.training.optimizer_beta,
        )
        return [optimizer_vae, optimizer_disc], []

    def on_after_batch_transfer(
        self, batch: Dict[str, torch.Tensor], dataloader_idx: int = 0
    ) -> torch.Tensor:
        x = batch["videos"]
        return self._rearrange_and_normalize(x)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        namespace: str = "training",
        video_length: Optional[int] = None,
    ):
        is_training = namespace == "training"
        batch = self._randomly_crop_video(
            batch, video_length=video_length, random_start=is_training
        )
        recons, posterior = self.vae(batch)

        if is_training:
            optimizer_vae, optimizer_disc = self.optimizers()
            warmup_info = self._compute_warmup()

        # Optimize VAE
        vae_loss, vae_loss_dict = self.loss(
            inputs=batch,
            reconstructions=recons,
            posteriors=posterior,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.vae.get_last_layer(),
            namespace=f"{namespace}_vae",
        )
        if is_training:
            self._optimizer_step(optimizer_vae, vae_loss, warmup_info)
        self._log_losses(f"{namespace}_vae", vae_loss, vae_loss_dict, is_training)
        # Optimize Discriminator
        disc_loss, disc_loss_dict = self.loss(
            inputs=batch,
            reconstructions=recons,
            posteriors=posterior,
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=None,
            namespace=f"{namespace}_disc",
        )
        if is_training:
            self._optimizer_step(optimizer_disc, disc_loss, warmup_info)
        self._log_losses(f"{namespace}_disc", disc_loss, disc_loss_dict, is_training)

        return {
            "gts": self._rearrange_and_unnormalize(batch),
            "recons": self._rearrange_and_unnormalize(recons),
        }

    def on_validation_epoch_start(self) -> None:
        self.num_logged_videos = [0] * len(self.validation_video_lengths)

    def on_validation_epoch_end(self, namespace: str = "validation") -> None:
        # Log metrics
        for video_length, metrics in zip(self.validation_video_lengths, self.metrics):
            self.log_dict(
                metrics.log(f"{namespace}_{video_length}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        for video_length_idx, video_length in enumerate(self.validation_video_lengths):
            new_namespace = f"{namespace}_{video_length}"
            metrics = self.metrics[video_length_idx]
            num_logged_videos = self.num_logged_videos[video_length_idx]
            output_dict = self.training_step(
                batch, batch_idx, new_namespace, video_length
            )
            # Update metrics
            gts, recons = output_dict["gts"], output_dict["recons"]
            metrics(recons, gts)

            # Log ground truth and reconstruction videos
            gts, recons = self.gather_data((gts, recons))
            if not (
                is_rank_zero
                and self.logger
                and num_logged_videos < self.cfg.logging.max_num_videos
            ):
                continue
            num_videos_to_log = min(
                self.cfg.logging.max_num_videos - num_logged_videos,
                gts.shape[1],
            )
            gts, recons = map(
                lambda x: x[:num_videos_to_log],
                (gts, recons),
            )
            log_video(
                recons,
                gts,
                step=None if new_namespace.startswith("test") else self.global_step,
                namespace=f"{new_namespace}_vis",
                logger=self.logger.experiment,
                indent=num_logged_videos,
            )
            self.num_logged_videos[video_length_idx] += num_videos_to_log

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        return self.validation_step(batch, batch_idx, namespace="test")

    def _log_losses(
        self,
        namespace: str,
        loss: torch.Tensor,
        loss_dict: Dict[str, torch.Tensor],
        on_step: bool = True,
    ):
        if self.global_step % self.cfg.logging.loss_freq > 1:
            return
        loss_dict = {
            k: v.to(self.device) for k, v in loss_dict.items()
        }  # to enable gathering across devices
        self.log(
            f"{namespace}/loss",
            loss,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(
            loss_dict,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=False,
            sync_dist=True,
        )

    def _optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        warmup_info: Tuple[bool, float],
    ) -> None:
        should_warmup, lr_scale = warmup_info
        optimizer.zero_grad()
        self.manual_backward(loss)
        if self.gradient_clip_val is not None:
            self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val)
        if should_warmup:
            optimizer = warmup(optimizer, self.lr, lr_scale)
        optimizer.step()

    def _compute_warmup(self) -> Tuple[bool, float]:
        should_warmup, lr_scale = False, 1.0
        if self.global_step < self.warmup_steps:
            should_warmup = True
            lr_scale = float(self.global_step + 1) / self.warmup_steps
        elif (
            self.global_step >= self.disc_start - 1
            and self.global_step < self.disc_start + self.warmup_steps
        ):
            should_warmup = True
            lr_scale = float(self.global_step - self.disc_start + 1) / self.warmup_steps
        return should_warmup, min(lr_scale, 1.0)

    def _rearrange_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t c h w -> b c t h w")
        return 2.0 * x - 1.0

    def _rearrange_and_unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = 0.5 * x + 0.5
        return rearrange(x, "b c t h w -> b t c h w")

    def _randomly_crop_video(
        self,
        x: torch.Tensor,
        video_length: Optional[int] = None,
        random_start: bool = True,
    ) -> torch.Tensor:
        """
        Randomly crop the video to a random temporal length, if not provided.
        Same length across all GPUs.
        """
        if video_length is None:
            rand = broadcast_from_zero(
                lambda: torch.zeros(1, device=self.device),
                lambda: torch.rand(1, device=self.device),
            ).item()
            for i, prob in enumerate(self.video_length_probs):
                if rand < prob:
                    video_length = self.video_lengths[i]
                    break
        crop_start = random.randint(0, x.size(2) - video_length) if random_start else 0
        x = x[:, :, crop_start : crop_start + video_length]
        assert x.size(2) == video_length, "Cropped video length does not match"
        return x
