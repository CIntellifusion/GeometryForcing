"""
Adapted from https://github.com/ljh0v0/FVMD-frechet-video-motion-distance
"""

from typing import Tuple, List
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .pips2 import Pips


class NoTrainPips(Pips):
    def train(self, mode: bool) -> "NoTrainPips":
        return super().train(False)


class MotionExtractor(nn.Module):
    def __init__(
        self,
        resolution: int = 256,
        segment_length: int = 16,
        num_points: int = 400,
        num_iters: int = 16,
    ):
        super().__init__()
        self.pips = NoTrainPips.from_pretrained()
        self.resolution = resolution
        self.segment_length = segment_length
        self.num_points = num_points
        self.num_iters = num_iters

        self.register_buffer("coords", self._get_coords(), persistent=False)

    def _get_coords(self) -> Tensor:
        sqrt_num_points = int(np.sqrt(self.num_points))
        grid_y, grid_x = torch.meshgrid(
            torch.arange(sqrt_num_points), torch.arange(sqrt_num_points), indexing="ij"
        )
        grid_y = 8 + grid_y.reshape(-1).float() / (sqrt_num_points - 1) * (
            self.resolution - 16
        )
        grid_x = 8 + grid_x.reshape(-1).float() / (sqrt_num_points - 1) * (
            self.resolution - 16
        )
        coords = torch.stack([grid_x, grid_y], dim=-1)  # N, 2
        return repeat(
            coords, "n d -> b s n d", b=1, s=self.segment_length
        )  # 1, S, N, 2

    def _track_keypoints(self, video_segment: Tensor) -> Tensor:
        B, S, _, H, W = video_segment.shape
        assert B == 1, f"PIPS2 only supports batch size 1, but got {B}"
        preds = self.pips(
            self.coords,
            video_segment,
            iters=self.num_iters,
            feat_init=None,
            beautify=True,
        )
        return preds[-1]

    @staticmethod
    def _calc_velocity(trajectories: Tensor) -> Tensor:
        trajs_e0 = trajectories[:, :-1]  # B,S-1,N,2
        trajs_e1 = trajectories[:, 1:]  # B,S-1,N,2
        velocity = trajs_e1 - trajs_e0  # B,S-1,N,2
        velocity = torch.cat(
            [torch.zeros_like(velocity[:, 0:1]), velocity], dim=1
        )  # B,S,N,2
        return velocity

    @staticmethod
    def _calc_acceleration(velocity: Tensor) -> Tensor:
        velocity0 = velocity[:, 1:-1]  # B,S-2,N,2
        velocity1 = velocity[:, 2:]  # B,S-2,N,2
        acceleration = velocity1 - velocity0  # B,S-2,N,2
        acceleration = torch.cat(
            [
                torch.zeros_like(acceleration[:, 0:2]),
                acceleration,
            ],
            dim=1,
        )  # B,S,N,2
        return acceleration

    def _get_velocities_and_accelerations(
        self, video: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        velocities, accelerations = [], []
        # for start_frame in range(0, video.shape[1], self.segment_length - 1):
        for start_frame in range(
            0, video.shape[1] - self.segment_length + 1, self.segment_length - 1
        ):
            video_segment = video[:, start_frame : start_frame + self.segment_length]
            assert (
                video_segment.shape[1] == self.segment_length
            ), f"Video segment length does not match {self.segment_length}"
            trajectories = self._track_keypoints(video_segment)
            velocities.append(self._calc_velocity(trajectories))
            accelerations.append(self._calc_acceleration(velocities[-1]))
        # each segment is considered as a separate video
        return velocities, accelerations

    def _compute_histogram(self, vectors: Tensor) -> Tensor:
        """
        Args:
            vectors (torch.Tensor): (B, S, N, 2) - velocities or accelerations
        """
        B, S, N, _ = vectors.shape
        vectors = vectors.reshape(B * S * N, -1)

    def forward(self, videos: Tensor) -> Tensor:
        """
        Args:
            videos (torch.Tensor): (B, S, C, H, W), range [-1, 1]
        """
        device, batch_size = videos.device, videos.shape[0]
        # resize video self.resolution x self.resolution
        videos = rearrange(
            F.interpolate(
                rearrange(videos, "b s c h w -> (b s) c h w"),
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            ),
            "(b s) c h w -> b s c h w",
            b=batch_size,
        )
        all_velocities, all_accelerations = [], []
        for idx in range(videos.shape[0]):  # Pips require batch size 1
            velocities, accelerations = self._get_velocities_and_accelerations(
                videos[idx : idx + 1]
            )
            all_velocities.extend(velocities)
            all_accelerations.extend(accelerations)
        all_velocities = torch.cat(all_velocities, dim=0)
        all_accelerations = torch.cat(all_accelerations, dim=0)

        hist_velocities, hist_accelerations = map(
            lambda x: rearrange(
                torch.from_numpy(calc_hist(x.cpu().numpy())).to(device),
                "b ... -> b (...)",
            ),
            (all_velocities, all_accelerations),
        )
        return torch.cat([hist_velocities, hist_accelerations], dim=-1)


def cut_subcube(vectors: np.ndarray, cell_size: int = 5, cube_frames: int = 4):
    """
    Cut the whole video sequence into subcubes
    Args:
        vectors (np.ndarray): (B, S, H, W, 2)
        cell_size (int): the height and width of the subcube
        cube_frames (int): the number of frames in a subcube

    Returns:
        vectors (np.ndarray): (B*MS*MH*MW, cube_frames, cell_size, cell_size, 2)
        MS (int): the number of subcubes in the time dimension
        MH (int): the number of subcubes in the height dimension
        MW (int): the number of subcubes in the width dimension
    """
    B, S, H, W, _ = vectors.shape
    MH = H // cell_size
    MW = W // cell_size
    MS = S // cube_frames
    vectors = vectors[:, : MS * cube_frames, : MH * cell_size, : MW * cell_size, :]
    vectors = vectors.reshape(B, MS, cube_frames, MH, cell_size, MW, cell_size, 2)
    vectors = vectors.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    vectors = vectors.reshape(-1, cube_frames, cell_size, cell_size, 2)
    return vectors, MS, MH, MW


def count_subcube_hist(
    vector_cell: np.ndarray, angle_bins: int = 8, magnitude_bins: int = 256
) -> np.ndarray:
    """
    Count the histogram for the subcube
    Args:
        vector_cell (np.ndarray): (S, H, W, 2)
        angle_bins (int): the number of angle bins
        magnitude_bins (int): the number of magnitude bins

    Returns:
        HOG_hist (np.ndarray): (angle_bins,)
    """
    HOG_hist = np.zeros(angle_bins)
    S, H, W, _ = vector_cell.shape

    angle_list = np.arctan2(vector_cell[:, :, :, 0], vector_cell[:, :, :, 1])
    angle_bin_list = (angle_list + np.pi) // (2 * np.pi / angle_bins)
    angle_bin_list = np.clip(angle_bin_list, 0, angle_bins - 1)

    magnitude_list = np.linalg.norm(vector_cell, axis=3)
    magnitude_list = np.clip(magnitude_list, 0, magnitude_bins - 1)
    magnitude_list = magnitude_list + 1
    magnitude_list = np.log2(magnitude_list)
    magnitude_list = np.clip(magnitude_list, 0, int(np.log2(magnitude_bins)))
    magnitude_list = np.ceil(magnitude_list)
    magnitude_list = magnitude_list / np.log2(magnitude_bins)

    for s in range(S):
        for i in range(H):
            for j in range(W):
                HOG_hist[int(angle_bin_list[s, i, j])] += magnitude_list[s, i, j]

    return HOG_hist


def calc_hist(
    vectors: np.ndarray, cell_size: int = 5, angle_bins: int = 8, cube_frames: int = 4
) -> np.ndarray:
    """
    Calculate the histogram for the whole video sequence
    Args:
        vectors (np.ndarray): (B, S, H, W, 2)
        cell_size (int): the height and width of the subcube
        angle_bins (int): the number of angle bins
        cube_frames (int): the number of frames in a subcube

    Returns:
        histos (np.ndarray): (B, MS, MH, MW, angle_bins)
    """
    B, S, N, _ = vectors.shape
    H = np.sqrt(N).round().astype(np.int32)
    W = H
    vectors = vectors.reshape(B, S, H, W, 2)

    vectors, MS, MH, MW = cut_subcube(vectors, cell_size, cube_frames)
    histos = []
    for i in range(vectors.shape[0]):
        histos.append(count_subcube_hist(vectors[i], angle_bins))
    histos = np.stack(histos, axis=0)
    histos = histos.reshape(B, MS, MH, MW, angle_bins)
    return histos
