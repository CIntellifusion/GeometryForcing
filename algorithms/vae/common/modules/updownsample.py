from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .conv import PaddedConv3D
from .ops import video_to_image, cast_tuple


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    @video_to_image
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=0
            )

    @video_to_image
    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # pylint: disable-next=not-callable
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class SpatialUpsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (1, 1),
        unup=False,
        is_causal=True,
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.unup = unup
        self.conv = PaddedConv3D(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=1,
            is_causal=is_causal,
        )

    def forward(self, x):
        if not self.unup:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> b (c t) h w")
            x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
            x = rearrange(x, "b (c t) h w -> b c t h w", t=t)
        x = self.conv(x)
        return x


class SpatialDownsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (2, 2),
        is_causal=True,
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 2)
        stride = cast_tuple(stride, 2)
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.conv = PaddedConv3D(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=0,
            is_causal=is_causal,
        )

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 0)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Spatial2xTime2x3DUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, is_causal=True, is_first=False):
        super().__init__()
        self.conv = PaddedConv3D(
            in_channels, out_channels, kernel_size=3, padding=1, is_causal=is_causal
        )
        self.is_causal = is_causal

        if not is_causal and is_first:
            self.temporal_up_conv = nn.ConvTranspose3d(
                in_channels,
                in_channels,
                kernel_size=(2, 1, 1),
                stride=1,
                padding=0,
            )

    def forward(self, x):
        if self.is_causal:
            if x.size(2) > 1:
                x, x_ = x[:, :, :1], x[:, :, 1:]
                x_ = F.interpolate(x_, scale_factor=(2, 2, 2), mode="trilinear")
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
                x = torch.concat([x, x_], dim=2)
            else:
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
        else:
            if x.size(2) > 1:
                x = F.interpolate(x, scale_factor=(2, 2, 2), mode="trilinear")
            else:
                # if temporal length is 1,
                # we upsample temporally using up conv instead of interpolation because interpolation leads to duplicate frames
                x = self.temporal_up_conv(x)
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")

        return self.conv(x)


class Spatial2xTime2x3DDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, is_causal=True):
        super().__init__()
        self.conv = PaddedConv3D(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=0,
            stride=2,
            is_causal=is_causal,
        )

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 0)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x
