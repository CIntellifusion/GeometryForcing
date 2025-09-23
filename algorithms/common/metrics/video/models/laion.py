import torch
from torch import nn, Tensor

LAION_PATH = "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"


class LAION(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(768, 1)
        state_dict = torch.hub.load_state_dict_from_url(
            LAION_PATH, map_location="cpu", progress=False
        )
        self.model.load_state_dict(state_dict)

    def train(self, mode: bool) -> "LAION":
        super().train(False)

    def forward(self, clip_features: Tensor) -> Tensor:
        return self.model(clip_features)
