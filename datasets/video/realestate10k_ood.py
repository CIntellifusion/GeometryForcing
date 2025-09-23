from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
from einops import einsum
import torch
import numpy as np
from utils.geometry_utils import CameraPose
from .base_video import SPLIT
from .realestate10k import RealEstate10KAdvancedVideoDataset


class RealEstate10KOODAdvancedVideoDataset(RealEstate10KAdvancedVideoDataset):
    """
    RealEstate10K video dataset, modified for "out-of-distribution history" experiments.
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        assert (
            split != "training"
        ), "RealEstate10KOODAdvancedVideoDataset is only for evaluation"
        self.min_angle = cfg.rotation_angle.min
        self.max_angle = cfg.rotation_angle.max
        super().__init__(cfg, split, current_epoch)

    def _get_angle(self, video_metadata: Dict[str, Any]) -> float:
        """
        Given a video metadata for a RealEstate10K scene,
        compute the maximum angle (degrees) of rotation of the camera within the video.
        """
        poses = self.load_cond(
            video_metadata=video_metadata,
            start_frame=0,
            end_frame=self.video_length(video_metadata),
        )
        poses = CameraPose.from_vectors(
            self._process_external_cond(poses, 1).unsqueeze(0)
        )
        R = poses._R[0]
        R_rel = einsum(R, R, "t2 i j, t1 k j -> t1 t2 i k")
        angles = torch.acos((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)
        angles.fill_diagonal_(0)
        return angles.max().item() * 180 / np.pi

    def load_metadata(self) -> List[Dict[str, Any]]:
        """Only filter videos with camera rotation within the specified range."""
        metadata = super().load_metadata()
        filtered_metadata = []

        for video_metadata in metadata:
            angle = self._get_angle(video_metadata)
            if self.min_angle <= angle <= self.max_angle:
                filtered_metadata.append(video_metadata)
        return filtered_metadata

    def prepare_clips(self):
        """Only a single clip per video."""
        num_clips = torch.as_tensor([1 for _ in range(len(self.metadata))])
        self.cumulative_sizes = num_clips.cumsum(0).tolist()
        self.idx_remap = self._build_idx_remap()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, start_frame = self.get_clip_location(idx)
        assert start_frame == 0, "start_frame should be 0"
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        video, cond = self.load_video_and_cond(video_metadata, 0, video_length)
        assert len(video) == len(cond), "Video and cond have different lengths"

        # NOTE: Context frames are evenly spaced across the full video.
        # and prediction frames evenly interpolate the context frames.
        context_indices = torch.linspace(
            0, video_length - 1, self.cfg.context_length, dtype=torch.long
        )
        pred_indices = torch.linspace(
            context_indices[-2:].float().mean().long().item(),
            context_indices[:2].float().mean().long().item(),
            self.cfg.max_frames - self.cfg.context_length,
            dtype=torch.long,
        )
        indices = torch.cat([context_indices, pred_indices])

        video, cond = video[indices], self._process_external_cond(cond[indices], 1)

        return {
            "videos": self.transform(video),
            "conds": cond,
            "nonterminal": torch.ones(self.cfg.max_frames, dtype=torch.bool),
        }
