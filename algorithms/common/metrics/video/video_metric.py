from typing import List, Optional, Set, Tuple, Iterable
import torch
from torch import Tensor
from torch import nn
from einops import rearrange, repeat
from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
from .fid import FrechetInceptionDistance
from .fvd import FrechetVideoDistance
from .fvmd import FrechetVideoMotionDistance
from .inception_score import InceptionScore
from .lpips import LearnedPerceptualImagePatchSimilarity
from .vbench import VBench
from .types import VideoMetricType, VideoMetricModelType
from .shared_registry import SharedVideoMetricModelRegistry


class VideoMetric(nn.Module):
    """
    A class that wraps all video metrics.

    Args:
        registry: A registry of models used for computing video metrics. When multiple `VideoMetric` instances are created to evaluate multiple models or tasks,
            the same registry shall be passed to all instances to avoid redundant model loading and save GPU memory.
        metric_types: List of video metric types. For supported video metric types, see `VideoMetricType`.
    """

    # Evaluated for the entire video
    VIDEO_WISE_METRICS = {
        VideoMetricType.FVD,
        VideoMetricType.IS,
        VideoMetricType.REAL_IS,
        VideoMetricType.FVMD,
        VideoMetricType.VBENCH,
        VideoMetricType.REAL_VBENCH,
    }

    # Evaluated for the entire video, evaluated using "shared" I3D features
    I3D_DEPENDENT_METRICS = {
        VideoMetricType.FVD,
        VideoMetricType.IS,
        VideoMetricType.REAL_IS,
    }

    # VBench metrics, compute() returns a dictionary of dimension scores and a final VBench score
    VBENCH_METRICS = {VideoMetricType.VBENCH, VideoMetricType.REAL_VBENCH}

    # Evaluated for each "non-context" frame of the video
    FRAME_WISE_METRICS = {
        VideoMetricType.LPIPS,
        VideoMetricType.FID,
        VideoMetricType.MSE,
        VideoMetricType.SSIM,
        VideoMetricType.PSNR,
    }

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        metric_types: List[str] | List[VideoMetricType],
        split_batch_size: int = 16,
    ):
        super().__init__()
        modules = {}
        metric_types = [VideoMetricType(metric_type) for metric_type in metric_types]
        for metric_type in metric_types:
            match metric_type:
                case VideoMetricType.LPIPS:
                    module = LearnedPerceptualImagePatchSimilarity(
                        registry=registry, normalize=True
                    )
                case VideoMetricType.FID:
                    module = FrechetInceptionDistance(registry=registry, normalize=True)
                case VideoMetricType.FVMD:
                    module = FrechetVideoMotionDistance(registry=registry)
                case VideoMetricType.VBENCH | VideoMetricType.REAL_VBENCH:
                    module = VBench(registry=registry)
                case VideoMetricType.FVD:
                    module = FrechetVideoDistance()
                case VideoMetricType.IS | VideoMetricType.REAL_IS:
                    module = InceptionScore()
                case VideoMetricType.MSE:
                    module = MeanSquaredError()
                case VideoMetricType.SSIM:
                    module = StructuralSimilarityIndexMeasure(data_range=1.0)
                case VideoMetricType.PSNR:
                    module = PeakSignalNoiseRatio(data_range=1.0)
                case _:
                    raise ValueError(f"Unknown video metric type: {metric_type}")
            registry.register_for_metric(metric_type)
            modules[metric_type] = module

        self.metrics = nn.ModuleDict(modules)

        self.registry = registry
        self.split_batch_size = split_batch_size

    def keys(self) -> Iterable[VideoMetricType]:
        return self.metrics.keys()

    def items(self) -> Iterable[Tuple[VideoMetricType, nn.Module]]:
        return self.metrics.items()

    def values(self) -> Iterable[nn.Module]:
        return self.metrics.values()

    def _filtered_items(
        self, metric_types: Set[VideoMetricType], not_in: bool = False
    ) -> Iterable[Tuple[VideoMetricType, nn.Module]]:
        if not_in:
            return filter(lambda x: x[0] not in metric_types, self.items())
        return filter(lambda x: x[0] in metric_types, self.items())

    def _extract_i3d_features(self, x: Tensor) -> Tensor:
        """
        Extract I3D features. Requires a batch of videos of shape (B, T, C, H, W) and range [0, 1].
        """
        # temporally pad both ends to be at least 9 frames
        if x.shape[1] < 9:
            pad = (10 - x.shape[1]) // 2
            x = torch.cat(
                [
                    repeat(x[:, 0:1], "b 1 c h w -> b t c h w", t=pad).clone(),
                    x,
                    repeat(x[:, -1:], "b 1 c h w -> b t c h w", t=pad),
                ],
                dim=1,
            )
        x = 2.0 * x - 1.0
        x = rearrange(torch.clamp(x, -1.0, 1.0), "b t c h w -> b c t h w").contiguous()
        return self.registry(
            VideoMetricModelType.I3D,
            x,
            rescale=False,
            resize=True,
            return_features=True,
        )

    def forward(
        self,
        preds: Tensor,
        target: Tensor,
        context_mask: Optional[Tensor] = None,
    ):
        """
        Update video metrics with the given predictions and targets.
        Args:
            preds: Predictions of shape (B, T, C, H, W), [0, 1]
            target: Targets of shape (B, T, C, H, W), [0, 1]
            context_mask: A boolean tensor of shape (T,) indicating whether each frame is a context frame.
                1) Frame-wise metrics are computed only for non-context frames.
                2) Context frames of generated videos will be overwritten with the corresponding frames of the real videos.
        """
        # NOTE: Due to the large memory consumption of video metrics (especially VBench and LPIPS),
        # we split the batch into smaller chunks and update metrics for each chunk.
        preds_split = preds.chunk(self.split_batch_size, dim=0)
        target_split = target.chunk(self.split_batch_size, dim=0)
        assert len(preds_split) == len(
            target_split
        ), "Batch size of preds and target must be the same."
        for preds_chunk, target_chunk in zip(preds_split, target_split):
            self._update(preds_chunk, target_chunk, context_mask)

    def _update(
        self, preds: Tensor, target: Tensor, context_mask: Optional[Tensor] = None
    ):
        """
        Note:
            FVD, IS, REAL_IS: (B, C) (I3D features)
            FVMD: (B, T, C, H, W), [0, 1]
            FID, LPIPS, MSE, SSIM, PSNR: (B, C, H, W), [0, 1]
        """
        if context_mask is None:
            context_mask = torch.zeros(
                preds.shape[1], device=preds.device, dtype=torch.bool
            )

        # replace all NaNs with 0 / clamp to [0, 1] / convert to float32
        preds, target = map(
            lambda x: torch.clamp(torch.nan_to_num(x, nan=0.0), 0.0, 1.0).to(
                torch.float32
            ),
            (preds, target),
        )
        # overwrite context frames of generated videos with the corresponding frames of the real videos
        preds = torch.where(
            rearrange(context_mask, "t -> 1 t 1 1 1"),
            target,
            preds,
        )
        # update I3D-dependent video-wise metrics
        i3d_dependent_metrics = self.I3D_DEPENDENT_METRICS.intersection(self.keys())
        if i3d_dependent_metrics:
            fake_features, real_features = None, None
            if {VideoMetricType.FVD, VideoMetricType.IS}.intersection(self.keys()):
                fake_features = self._extract_i3d_features(preds)
            if {VideoMetricType.FVD, VideoMetricType.REAL_IS}.intersection(self.keys()):
                real_features = self._extract_i3d_features(target)
            for metric_type, module in self._filtered_items(self.I3D_DEPENDENT_METRICS):
                if metric_type == VideoMetricType.FVD:
                    module.update(fake_features, real_features)
                else:
                    module.update(
                        fake_features
                        if metric_type == VideoMetricType.IS
                        else real_features
                    )

        # update VBench metrics
        for metric_type, module in self._filtered_items(self.VBENCH_METRICS):
            module.update(preds if metric_type == VideoMetricType.VBENCH else target)

        # update other video-wise metrics
        for metric_type, module in self._filtered_items(
            self.VIDEO_WISE_METRICS - self.I3D_DEPENDENT_METRICS - self.VBENCH_METRICS
        ):
            module.update(preds, target)

        # reshape a batch of videos to a batch of image frames
        preds, target = map(
            lambda x: rearrange(x[:, ~context_mask], "b t c h w -> (b t) c h w"),
            (preds, target),
        )

        # update frame-wise metrics
        for metric_type, module in self._filtered_items(self.FRAME_WISE_METRICS):
            module.update(preds, target)

    def log(self, prefix: str):
        dict_metrics = {}

        # call compute() for VBench metrics and reorganize the results
        for metric_type, module in self._filtered_items(self.VBENCH_METRICS):
            output = module.compute()
            dict_metrics.update(
                {
                    f"{prefix}/{metric_type}/{key}": value
                    for key, value in output.items()
                }
            )
            # NOTE: manually calling compute() requires manual call to reset() afterwards
            # if we need a functionality to log VBench several times within a single validation epoch,
            # we should move reset() to on_validation_epoch_end() in the Lightning module that uses this VideoMetric
            module.reset()

        # other metrics (no need to call compute())
        dict_metrics.update(
            {
                f"{prefix}/{metric_type}": module
                for metric_type, module in self._filtered_items(
                    self.VBENCH_METRICS, not_in=True
                )
                if not (hasattr(module, "is_empty") and module.is_empty)
            }
        )

        return dict_metrics

    def reset(self):
        for module in self.values():
            module.reset()
