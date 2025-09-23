from typing import Callable, Any
from torch import Tensor
from einops import rearrange


def videos_as_images(
    func: Callable[..., Tensor], num_video_args: int = 1
) -> Callable[..., Tensor]:
    """
    Wrapper that enables a function that operates on a batch of images to operate on a batch of videos.
    Can also be used as a decorator.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Tensor:
        # check if the first argument is a tensor or not
        # if not, assume it is "self"
        new_args = list(args)
        videos_idx = 1 if not isinstance(new_args[0], Tensor) else 0
        b = new_args[videos_idx].shape[0]
        for idx in range(videos_idx, videos_idx + num_video_args):
            new_args[idx] = rearrange(new_args[idx], "b t c h w -> (b t) c h w")
        return rearrange(
            func(*new_args, **kwargs),
            "(b t) ... -> b t ...",
            b=b,
        )

    return wrapper
