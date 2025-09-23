"""
camera rotation utils:
follow the https://github.com/xizaoqu/WorldMem/issues/8 
given the first camera pose, got a rotation sequence 
"""

import torch
import math
import torch
import math

def rotate_single_pose_y(pose: torch.Tensor, seq_length: int) -> torch.Tensor:
    import math
    device = pose.device
    # [0:4]
    print(pose.shape)
    fx = pose[0:1]
    fy = pose[1:2]
    cx = pose[2:3]
    cy = pose[3:4]
    pad_1 = pose[4:5]
    pad_2 = pose[5:6]
    print(pad_1, pad_2, fx, fy, cx, cy)
    R = torch.stack([
        pose[6:9],    # R row 0
        pose[10:13],   # R row 1
        pose[14:17],  # R row 2
    ], dim=0)  # [3, 3]

    t = torch.stack([
        pose[9],      # t_x
        pose[13],     # t_y
        pose[17],     # t_z
    ], dim=0).reshape(3, 1)  # [3, 1]

    thetas = torch.linspace(0, 2 * math.pi, seq_length, device=device)
    
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
    print(fx.shape)          # torch.Size([1])
    rotated_poses = []
    for R_y in R_ys:
        # 绕着 自己的y轴旋转式
        R_new = R @ R_y  # [3, 3]
        # 绕着 世界坐标系的y轴旋转
        # R_new = R_y @ R  # [3, 3]
        # 如果想旋转t，用 t_new = R_y @ t，否则保持不变
        t_new = t  # or t_new = R_y @ t
        # print(R_new[0].shape,t_new[0:1].shape)  # torch.Size([3, 1])
        cam_pose = torch.cat(
            [fx, fy, cx, cy, pad_1, pad_2,
             R_new[0], t_new[0], 
             R_new[1], t_new[1], 
             R_new[2], t_new[2]], dim=0
        )
        print(cam_pose)  # torch.Size([18])
        rotated_poses.append(cam_pose)

    return torch.stack(rotated_poses, dim=0)  # [seq_length, 18]

if __name__ == "__main__":
    from pathlib import Path
    camera_poses_dir="data/real-estate-10k-rotate-100/test_poses"
    files = sorted(Path(camera_poses_dir).glob("*.pt"))
    print(f"Found {len(files)} camera pose files in {camera_poses_dir}")
    for file in files:
        camera_poses = torch.load(file)
        print(f"Processing {file} with shape {camera_poses.shape}")
        
        # Rotate camera poses
        rotated_poses = rotate_single_pose_y(camera_poses[0], seq_length=256).squeeze(0)  # [B, N, 16]()
        
        # Save rotated poses
        output_file = Path("data/real-estate-10k-rotate-100/test_poses") / file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rotated_poses, output_file)
        print(f"Saved rotated poses to {output_file} with shape {rotated_poses.shape}")