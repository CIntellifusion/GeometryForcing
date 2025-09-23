"""
A very minimal implementation of continuous-time diffusion models. For compatibility with other modules,
sampling schedules are still implemented in discrete time.
"""

from abc import ABC, abstractmethod
from typing import Optional
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from .discrete_diffusion import DiscreteDiffusion, ModelPrediction


class ContinuousNoiseSchedule(nn.Module, ABC):
    """
    An abstract class for continuous noise schedule that is compatible with continuous-time diffusion models.
    """

    @classmethod
    def from_config(cls, cfg: DictConfig):
        match cfg.name:
            case "cosine":
                return CosineNoiseSchedule(cfg)
            case _:
                raise ValueError(f"unknown noise schedule {cfg.name}")

    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Given the timestep t within [0, 1], return the logSNR value at that timestep."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_logsnr(self) -> torch.Tensor:
        """Return the maximum logSNR value."""
        raise NotImplementedError

    @property
    @abstractmethod
    def min_logsnr(self) -> torch.Tensor:
        """Return the minimum logSNR value."""
        raise NotImplementedError


class CosineNoiseSchedule(ContinuousNoiseSchedule):
    """
    Cosine noise schedule that can be shifted from base resolution to target resolution,
    proposed in Simple Diffusion (2023, https://arxiv.org/abs/2301.11093).
    Here, `shift` should be set to `base_resolution / target_resolution`.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        logsnr_min, logsnr_max = cfg.get("logsnr_min", -15.0), cfg.get(
            "logsnr_max", 15.0
        )
        shift = cfg.get("shift", 1.0)
        self.register_buffer(
            "t_min",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "t_max",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "shift",
            2 * torch.log(torch.tensor(shift, dtype=torch.float32)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return (
            -2 * torch.log(torch.tan(self.t_min + t * (self.t_max - self.t_min)))
            + self.shift
        )

    @property
    def max_logsnr(self) -> torch.Tensor:
        return self.forward(
            torch.tensor(0.0, dtype=torch.float32, device=self.shift.device)
        )

    @property
    def min_logsnr(self) -> torch.Tensor:
        return self.forward(
            torch.tensor(1.0, dtype=torch.float32, device=self.shift.device)
        )


class ContinuousDiffusion(DiscreteDiffusion):
    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
    ):
        super().__init__(cfg, backbone_cfg, x_shape, max_tokens, external_cond_dim)
        assert (
            self.objective == "pred_v" and self.loss_weighting.strategy == "sigmoid"
        ), "ContinuousDiffusion only supports 'pred_v' objective and 'sigmoid' loss weighting"
        self.precond_scale = cfg.precond_scale
        self.sigmoid_bias = cfg.loss_weighting.sigmoid_bias

    def _build_buffer(self):
        super()._build_buffer()
        self.training_schedule = ContinuousNoiseSchedule.from_config(
            self.cfg.training_schedule
        )

    def model_predictions(self, x, k, external_cond=None, external_cond_mask=None,**kwargs):
        model_output = self.model(
            x, self.precond_scale * self.logsnr[k], external_cond, external_cond_mask,**kwargs
        )

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, k, v)
            pred_noise = self.predict_noise_from_v(x, k, v)
    
        model_pred = ModelPrediction(pred_noise, x_start, model_output)

        return model_pred

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
        **kwargs
    ):
        logsnr = self.training_schedule(k)
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
        x_t = alpha_t * x + sigma_t * noise

        # v-prediction 
        v_pred = self.model(x_t, self.precond_scale * logsnr, external_cond,**kwargs)
        noise_pred = alpha_t * v_pred + sigma_t * x_t
        x_pred = alpha_t * x_t - sigma_t * v_pred

        loss = F.mse_loss(noise_pred, noise.detach(), reduction="none")

        # sigmoid loss weighting
        # proposed by Kingma & Gao (2023, https://arxiv.org/abs/2303.00848)
        # further studied in Simple Diffusion 2 (2024, https://arxiv.org/abs/2410.19324)
        loss_weight = torch.sigmoid(self.sigmoid_bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss

    def forward_with_alignment_loss(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
 
    ):        
        logsnr = self.training_schedule(k)
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
        x_t = alpha_t * x + sigma_t * noise

        # v-prediction 
        # import pdb; pdb.set_trace() 
        ## START modification for alignment loss 
        v_pred,latents_list = self.model(x_t, self.precond_scale * logsnr, external_cond,return_latents=True)
        noise_pred = alpha_t * v_pred + sigma_t * x_t
        x_pred = alpha_t * x_t - sigma_t * v_pred
        loss = F.mse_loss(noise_pred, noise.detach(), reduction="none")

        # sigmoid loss weighting
        # proposed by Kingma & Gao (2023, https://arxiv.org/abs/2303.00848)
        # further studied in Simple Diffusion 2 (2024, https://arxiv.org/abs/2410.19324)
        loss_weight = torch.sigmoid(self.sigmoid_bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss , latents_list 