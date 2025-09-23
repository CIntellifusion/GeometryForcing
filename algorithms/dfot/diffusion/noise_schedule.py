from typing import Literal
import math
import torch


def make_beta_schedule(
    schedule: Literal["cosine", "sigmoid", "sd", "linear", "alphas_cumprod_linear"],
    shift: float = 1.0,
    clip_min: float = 1e-9,
    zero_terminal_snr: bool = True,
    **kwargs,
):
    schedule_fn = {
        "alphas_cumprod_linear": alphas_cumprod_linear_schedule,
        "cosine": cosine_schedule,
        "cosine_simple_diffusion": cosine_simple_diffusion_schedule,
        "sigmoid": sigmoid_schedule,
        "sd": sd_schedule,
        "linear": beta_linear_schedule,
    }[schedule]
    alphas_cumprod = schedule_fn(**kwargs)
    if schedule not in ["cosine", "cosine_simple_diffusion"] and zero_terminal_snr:
        # cosine schedule already enforces zero terminal SNR
        # simple diffusion's cosine schedule shall not enforce zero terminal SNR
        alphas_cumprod = enforce_zero_terminal_snr(alphas_cumprod)
    if (
        shift != 1.0 and schedule != "cosine_simple_diffusion"
    ):  # cosine_simple_diffusion already has shift built in
        alphas_cumprod = shift_beta_schedule(alphas_cumprod, shift)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1 - alphas
    return torch.clip(betas, clip_min, 1.0)


def cosine_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[1:]


def cosine_simple_diffusion_schedule(
    timesteps,
    logsnr_min=-15.0,
    logsnr_max=15.0,
    shifted: float = 1.0,
    interpolated: bool = False,
):
    """
    cosine schedule with different parameterization
    following Simple Diffusion - https://arxiv.org/abs/2301.11093
    Supports "shifted cosine schedule" and "interpolated cosine schedule"

    Args:
        timesteps: number of timesteps
        logsnr_min: minimum log SNR
        logsnr_max: maximum log SNR
        shifted: shift the schedule by a factor. Should be base_resolution / current_resolution
        interpolated: interpolate between the original and the shifted schedule, requires shifted != 1.0
    """
    t_min = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float64)))
    t_max = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float64)))
    t = torch.linspace(0, 1, timesteps, dtype=torch.float64)
    logsnr = -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))
    if shifted != 1.0:
        shifted_logsnr = logsnr + 2 * torch.log(
            torch.tensor(shifted, dtype=torch.float64)
        )
        if interpolated:
            logsnr = t * logsnr + (1 - t) * shifted_logsnr
        else:
            logsnr = shifted_logsnr

    alphas_cumprod = 1 / (1 + torch.exp(-logsnr))
    return alphas_cumprod


def alphas_cumprod_linear_schedule(timesteps: int) -> torch.Tensor:
    """
    linear schedule
    as proposed in https://arxiv.org/abs/2301.10972
    """
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    return (1 - t)[1:]


def beta_linear_schedule(
    timesteps: int, start: float = 0.0001, end: float = 0.02
) -> torch.Tensor:
    """
    linear schedule
    as proposed in https://arxiv.org/abs/2006.11239 (original DDPM paper)
    """
    betas = torch.linspace(start, end, timesteps, dtype=torch.float64)
    return (1 - betas).cumprod(dim=0)


def sigmoid_schedule(timesteps, start=-3, end=3, tau=1):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[1:]


def sd_schedule(timesteps, start=0.00085, end=0.0120):
    """
    stable diffusion's noise schedule
    https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/util.py#L21
    """
    betas = torch.linspace(start**0.5, end**0.5, timesteps, dtype=torch.float64) ** 2
    alphas_cumprod = (1 - betas).cumprod(dim=0)
    return alphas_cumprod


def shift_beta_schedule(alphas_cumprod: torch.Tensor, shift: float):
    """
    scale alphas_cumprod so that SNR is multiplied by shift ** 2
    """
    snr_scale = shift**2

    return (snr_scale * alphas_cumprod) / (
        snr_scale * alphas_cumprod + 1 - alphas_cumprod
    )


def enforce_zero_terminal_snr(alphas_cumprod):
    """
    enforce zero terminal SNR following https://arxiv.org/abs/2305.08891
    returns betas
    """
    alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)

    # store old values
    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
    # shift so last timestep is zero
    alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
    # scale so first timestep is back to original value
    alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / alphas_cumprod_sqrt[0]
    # convert to betas
    alphas_cumprod = alphas_cumprod_sqrt**2
    assert alphas_cumprod[-1] == 0, "terminal SNR not zero"
    return alphas_cumprod
