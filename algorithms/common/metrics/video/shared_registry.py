from typing import Dict, List
from torch import nn, Tensor
from torchmetrics.image.lpip import NoTrainLpips
from torchmetrics.image.fid import NoTrainInceptionV3
from utils.torch_utils import freeze_model
from utils.print_utils import suppress_warnings
from .models import I3D, MotionExtractor, CLIP, DINO, LAION, MUSIQ, RAFT, AMT_S
from .types import VideoMetricModelType, VideoMetricType


class SharedVideoMetricModelRegistry(nn.ModuleDict):
    """
    A registry of video metric models. All `VideoMetric` instances share this registry to avoid redundant model loading and save GPU memory.
    """

    METRIC_TO_MODELS: Dict[VideoMetricType, List[VideoMetricModelType]] = {
        VideoMetricType.LPIPS: [VideoMetricModelType.LPIPS],
        VideoMetricType.FID: [VideoMetricModelType.INCEPTION_V3],
        VideoMetricType.FVD: [VideoMetricModelType.I3D],
        VideoMetricType.IS: [VideoMetricModelType.I3D],
        VideoMetricType.REAL_IS: [VideoMetricModelType.I3D],
        VideoMetricType.MSE: [],
        VideoMetricType.SSIM: [],
        VideoMetricType.PSNR: [],
        VideoMetricType.FVMD: [VideoMetricModelType.PIPS],
        VideoMetricType.VBENCH: [
            VideoMetricModelType.CLIP_B_32,
            VideoMetricModelType.CLIP_L_14,
            VideoMetricModelType.DINO,
            VideoMetricModelType.LAION,
            VideoMetricModelType.MUSIQ,
            VideoMetricModelType.RAFT,
            VideoMetricModelType.AMT_S,
        ],
        VideoMetricType.REAL_VBENCH: [
            VideoMetricModelType.CLIP_B_32,
            VideoMetricModelType.CLIP_L_14,
            VideoMetricModelType.DINO,
            VideoMetricModelType.LAION,
            VideoMetricModelType.MUSIQ,
            VideoMetricModelType.RAFT,
            VideoMetricModelType.AMT_S,
        ],
    }

    def __init__(self):
        super().__init__()

    def __getitem__(self, model_type: VideoMetricModelType):
        if model_type not in self:
            self._register(model_type)
        return super().__getitem__(model_type)

    def forward(self, model_type: VideoMetricModelType, *args, **kwargs) -> Tensor:
        return self[model_type](*args, **kwargs)

    def _register(self, model_type: VideoMetricModelType):
        match model_type:
            case VideoMetricModelType.LPIPS:
                with suppress_warnings():
                    self[model_type] = NoTrainLpips(net="vgg", verbose=False)
            case VideoMetricModelType.INCEPTION_V3:
                self[model_type] = NoTrainInceptionV3(
                    name="inception-v3-compat", features_list=["2048"]
                )
            case VideoMetricModelType.I3D:
                self[model_type] = I3D()
            case VideoMetricModelType.PIPS:
                self[model_type] = MotionExtractor()
            case VideoMetricModelType.CLIP_B_32:
                self[model_type] = CLIP("ViT-B/32")
            case VideoMetricModelType.CLIP_L_14:
                self[model_type] = CLIP("ViT-L/14")
            case VideoMetricModelType.DINO:
                self[model_type] = DINO()
            case VideoMetricModelType.LAION:
                self[model_type] = LAION()
            case VideoMetricModelType.MUSIQ:
                self[model_type] = MUSIQ()
            case VideoMetricModelType.RAFT:
                self[model_type] = RAFT()
            case VideoMetricModelType.AMT_S:
                self[model_type] = AMT_S()
            case _:
                raise ValueError(f"Unknown video metric model type: {model_type}")
        freeze_model(self[model_type])
        self[model_type].eval()

    def _register_if_not_exists(self, model_type: VideoMetricModelType):
        if model_type not in self:
            self._register(model_type)

    def register_for_metric(self, metric_type: VideoMetricType):
        for model_type in self.METRIC_TO_MODELS[metric_type]:
            self._register_if_not_exists(model_type)
