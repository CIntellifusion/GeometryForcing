from typing import List
import hashlib
import random
import torch
from einops import rearrange
from PIL import Image
import numpy as np


class VideoTransform:
    """
    Adapted from pixelSplat
    https://github.dev/dcharatan/pixelsplat/blob/main/src/dataset/dataset_re10k.py
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape

    def __call__(
        self, images: torch.Tensor  # (*batch, c, h, w)
    ) -> torch.Tensor:  # (*batch, c, *shape)
        return self._rescale_and_crop(images, self.shape)

    @classmethod
    def _rescale(
        cls,
        image: torch.Tensor,  # (c, h, w),
        shape: tuple[int, int],
    ) -> torch.Tensor:  # (c, *shape)
        h, w = shape
        image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
        image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
        image_new = Image.fromarray(image_new)
        image_new = image_new.resize((w, h), Image.Resampling.LANCZOS)
        image_new = np.array(image_new) / 255
        image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
        return rearrange(image_new, "h w c -> c h w")

    @classmethod
    def _center_crop(
        cls,
        images: torch.Tensor,  # (*batch, c, h, w),
        shape: tuple[int, int],
    ) -> torch.Tensor:
        *_, h_in, w_in = images.shape
        h_out, w_out = shape

        # Note that odd input dimensions induce half-pixel misalignments.
        row = (h_in - h_out) // 2
        col = (w_in - w_out) // 2

        # Center-crop the image.
        images = images[..., :, row : row + h_out, col : col + w_out]

        return images

    @classmethod
    def _rescale_and_crop(
        cls,
        images: torch.Tensor,  # (*batch, c, h, w),
        shape: tuple[int, int],
    ):
        """
        Rescale and crop the images to the specified shape.
        Args:
            images (torch.Tensor): images tensor of shape (*batch, c, h, w). Range [0, 1].
            shape (tuple[int, int]): target shape.
        Returns:
            torch.Tensor: rescaled and cropped images tensor of shape (*batch, c, *shape). Range [0, 1].
        """
        *_, h_in, w_in = images.shape
        h_out, w_out = shape
        # assert h_out <= h_in and w_out <= w_in

        scale_factor = max(h_out / h_in, w_out / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)
        assert h_scaled == h_out or w_scaled == w_out

        *batch, c, h, w = images.shape
        images = images.reshape(-1, c, h, w)
        images = torch.stack(
            [cls._rescale(image, (h_scaled, w_scaled)) for image in images]
        )
        images = images.reshape(*batch, c, h_scaled, w_scaled)

        return cls._center_crop(images, shape)


def rescale_and_crop(
    video: torch.Tensor,
    resolution: int,
) -> np.ndarray:
    """
    Rescale and crop the video to the specified resolution. Used for preprocessing.
    Args:
        video (torch.Tensor): video tensor of shape (t, h, w, c). uint8.
        resolution (int): target resolution.
    Returns:
        np.ndarray: rescaled and cropped video tensor of shape (t, resolution, resolution, c). uint8.
    """
    *_, h, w, _ = video.shape
    scale_factor = max(resolution / h, resolution / w)
    h_scaled, w_scaled = round(h * scale_factor), round(w * scale_factor)
    assert h_scaled == resolution or w_scaled == resolution
    row = (h_scaled - resolution) // 2
    col = (w_scaled - resolution) // 2

    def _rescale_and_crop(image: torch.Tensor) -> torch.Tensor:
        image = Image.fromarray(image.numpy())
        image = image.resize((w_scaled, h_scaled), Image.Resampling.LANCZOS)
        return np.array(image)[row : row + resolution, col : col + resolution]

    video = np.stack([_rescale_and_crop(frame) for frame in video])
    return video
