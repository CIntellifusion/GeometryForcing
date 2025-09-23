import torch
from torch import nn
from utils.torch_utils import freeze_model
from utils.huggingface_utils import download_from_hf


def load_pretrained_i3d() -> torch.jit.ScriptModule:
    model_path = download_from_hf("metrics_models/i3d_torchscript.pt")
    # comes from https://github.com/JunyaoHu/common_metrics_on_video_quality/raw/main/fvd/styleganv/i3d_torchscript.pt
    detector = torch.jit.load(model_path)
    detector.eval()
    freeze_model(detector)

    def fixed_eval_train(self, mode: bool):
        # pylint: disable=bad-super-call
        return super(torch.jit.ScriptModule, self).train(False)

    # pylint: disable=no-value-for-parameter
    detector.train = fixed_eval_train.__get__(detector, torch.jit.ScriptModule)
    return detector


class I3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = load_pretrained_i3d()

    def train(self, mode: bool) -> "I3D":
        super().train(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
