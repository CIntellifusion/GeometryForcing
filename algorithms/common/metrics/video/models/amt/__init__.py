from functools import partial
import torch
from torch import nn, Tensor
from einops import repeat
from algorithms.common.metrics.video.utils import videos_as_images
from .amt_s import AMT_S as _AMT_S
from .utils import InputPadder

AMT_S_PATH = "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth"


class AMT_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _AMT_S(corr_radius=3, corr_lvls=4, num_flows=3)
        state_dict = torch.hub.load_state_dict_from_url(
            AMT_S_PATH, map_location="cpu", progress=False
        )["state_dict"]
        self.model.load_state_dict(state_dict)

        embt = torch.ones(1, 1, 1, 1) * 0.5
        self.register_buffer("embt", embt, persistent=False)

    def train(self, mode: bool) -> "AMT_S":
        super().train(False)

    @partial(videos_as_images, num_video_args=2)
    def forward(self, images1: Tensor, images2: Tensor) -> Tensor:
        images1, images2 = map(lambda x: x.float() / 255.0, (images1, images2))
        preds = self.model(
            images1, images2, repeat(self.embt, "1 1 1 1 -> b 1 1 1", b=images1.size(0))
        )["imgt_pred"]
        return (preds * 255.0).clamp(0, 255).to(torch.uint8)
