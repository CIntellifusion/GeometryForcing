from typing import Callable
import torch
from einops import rearrange


def video_to_image(func: Callable) -> Callable:
    def wrapper(self, x: torch.Tensor, *args, **kwargs):
        if x.dim() == 5:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = func(self, x, *args, **kwargs)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        else:
            x = func(self, x, *args, **kwargs)
        return x

    return wrapper


def nonlinearity(x):
    return x * torch.sigmoid(x)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) or isinstance(t, list) else ((t,) * length)
