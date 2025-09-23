import torch
import torch.nn.functional as F
from algorithms.common.metrics.video.utils import videos_as_images


def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat(
        [
            flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        1,
    )
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(
        input=img,
        grid=grid_,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output


class InputPadder:
    """Pads images such that dimensions are divisible by divisor"""

    def __init__(self, dims, divisor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    @videos_as_images
    def pad(self, *inputs):
        if len(inputs) == 1:
            return F.pad(inputs[0], self._pad, mode="replicate")
        else:
            return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, *inputs):
        if len(inputs) == 1:
            return self._unpad(inputs[0])
        else:
            return [self._unpad(x) for x in inputs]

    def _unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]
