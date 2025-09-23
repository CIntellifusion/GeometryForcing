from typing import Optional,Dict
import torch 
from omegaconf import DictConfig
from utils.print_utils import cyan
from torchvision.datasets.utils import (
    download_and_extract_archive,
)
from .base_video import SPLIT
from .realestate10k import RealEstate10KAdvancedVideoDataset
import torch.nn.functional as F
import math 

class RealEstate10KRotateAdvancedVideoDataset(RealEstate10KAdvancedVideoDataset):
    _DATASET_URL = "Not Implemented"  # Replace with actual URL if needed

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        assert (
            split != "training"
        ), "RealEstate10KMiniAdvancedVideoDataset is only for evaluation"
        super().__init__(cfg, split, current_epoch)

    def _should_download(self) -> bool:
        return not self.save_dir.exists()
    # TODO: modify this 
    def download_dataset(self):
        print(cyan("Downloading RealEstate10k Mini dataset..."))
        download_and_extract_archive(
            self._DATASET_URL,
            self.save_dir.parent,
            remove_finished=True,
        )
        print(cyan("Finished downloading RealEstate10k Mini dataset!"))

    def rotate_pose(self,pose:torch.Tensor, R_theta) -> torch.Tensor:
        """
        Rotate the camera pose around its own y-axis.
        Pose: [16] row major order 4x4 matrix 
        R_theta: [3, 3] rotation matrix
        """
        assert R_theta.shape == (3, 3), f"Rotation matrix shape {R_theta.shape} is not valid, should be [3, 3]"
        
        # Extract the rotation part of the pose
        prefix = pose[:4]  # fx, fy, cx, cy
        R = torch.stack([
            pose[4:7],    # R row 0
            pose[8:11],   # R row 1
            pose[12:15],  # R row 2
        ], dim=0)  # [3, 3]

        t = torch.stack([
            pose[7],      # t_x
            pose[11],     # t_y
            pose[15],     # t_z
        ], dim=0).reshape(3, 1)  # [3, 1]
        # Rotate the rotation part
        R_new = R @ R_theta
        # Keep the translation part unchanged
        t_new = t
        # Reconstruct the pose 4x4 
        new_pose = torch.cat(
            [prefix,
             R_new[0], t_new[0], 
             R_new[1], t_new[1], 
             R_new[2], t_new[2]], dim=0
        )
        return new_pose
    
    def rotate_pose_seq(self, poses: torch.Tensor, R_ys: torch.Tensor) -> torch.Tensor:
        """
        Rotate a batch of camera poses around their own y-axis.
        poses: [16] tensor where each row is a camera pose in row-major order
        R_ys: [seq_length, 3, 3] tensor of rotation matrices
        Returns: [seq_length, 16] tensor of rotated poses
        """
        device = poses.device
            
        seq_length = R_ys.shape[0]
        
        rotated_poses = []
        for i in range(seq_length):
            rotated_pose = self.rotate_pose(poses, R_ys[i])
            rotated_poses.append(rotated_pose)
        
        return torch.stack(rotated_poses, dim=0)
    
    def theta_to_rotation_matrix(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Convert an angle in radians to a rotation matrix around the y-axis.
        theta: [1] or [N] angle in radians
        Returns: [3, 3] rotation matrix
        """
        # 预先生成所有 R_y 矩阵，避免循环内重复创建tensor
        cos_t = torch.cos(thetas)
        sin_t = torch.sin(thetas)
        zeros = torch.zeros_like(thetas)
        ones = torch.ones_like(thetas)

        R_ys = torch.stack([
            torch.stack([cos_t, zeros, sin_t], dim=1),
            torch.stack([zeros, ones, zeros], dim=1),
            torch.stack([-sin_t, zeros, cos_t], dim=1),
        ], dim=1)  # [seq_length, 3, 3]
        return R_ys
    
    def get_rotated_poses(self, poses: torch.Tensor, seq_length: int) -> torch.Tensor:
        """
        Rotate a batch of camera poses around their own y-axis.
        poses: [16] tensor where each row is a camera pose in row-major order
        seq_length: int, number of rotation steps
        Returns: [seq_length, 16] tensor of rotated poses
        """
        device = poses.device
        
        thetas = torch.linspace(0, 2 * math.pi, seq_length, device=device)
        # import pdb; pdb.set_trace()
        R_ys = self.theta_to_rotation_matrix(thetas)
        poses_seq =  self.rotate_pose_seq(poses, R_ys)
        return poses_seq
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # only load the first frames and get all the conds
        video_idx, start_frame = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        # for rotation data 
        frame_skip = self.frame_skip
        start_frame = 0 
        end_frame = start_frame + (self.cfg.max_frames - 1) * frame_skip + 1
        assert frame_skip > 0, f"Frame skip {frame_skip} should be greater than 0"
        # end_frame = start_frame + (self.cfg.max_frames - 1) * frame_skip + 1
        video, cond = self.load_video_and_cond(video_metadata, start_frame, end_frame)
        # only the first frame is kept here. 
        test_cond = self._process_external_cond(cond, frame_skip) # 256,16
        # value checked , the online format is the same with 'utils/camera_rotation_utils.py'
        cond = self.get_rotated_poses(test_cond[0], self.n_frames).contiguous()  # [seq_length, 16]
        # import pdb; pdb.set_trace()
        pad_len = self.n_frames - len(video)
        nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
        if pad_len > 0:
            if video is not None:
                video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            nonterminal[-pad_len:] = 0
        else:
            video = video.contiguous()
        # load cond 
        assert len(cond) == self.n_frames == len(video), f"Video length {len(video)} and cond length {len(cond)} mismatch for idx {idx} in video {video_idx}."
        
        nonterminal = nonterminal[:: frame_skip]
        video = video[::frame_skip]
        cond = cond[::frame_skip]
        output = {
            "videos": self.transform(video) if video is not None else None,
            "conds": cond,
            "nonterminal": nonterminal,
        }
        return {key: value for key, value in output.items() if value is not None}
