"""
3D Geometry Utils (Camera Pose, Ray, Plücker coordinates)
All functions in this file follows the following convention:
- We assume a "batch" of multiple scenes, i.e. B T ...
- Camera Pose is composed of Rotation R (Tensor, B T 3 3), Translation T (Tensor, B T 3) - world to camera - and the intrinsics (Tensor, B T 4) - camera to image.
- Intrinsic matrix assumes that the pixel coordinates are normalized, i.e. left-top corner is (0, 0) and right-bottom corner is (1, 1). The intrinsics matrix is represented as (fx, fy, px, py).
- Rays are represented as either:
    - Original rays (Tensor, B T H W 6) - origin (Tensor, B T H W 3) and unnormalized direction (Tensor, B T H W 3) concatenated.
    - Plücker coordinates (Tensor, B T H W 6), normalized direction (Tensor, B T H W 3) and moment (Tensor, B T H W 3) concatenated.
"""

from typing import Tuple, Optional
import math
import torch
import roma
from einops import rearrange, einsum, repeat


class Ray:
    """
    A class to represent the batched rays.
    """

    def __init__(self, origin: torch.Tensor, direction: torch.Tensor):
        """
        Args:
            origin (torch.Tensor): The origin of the rays. Shape (B, T, H, W, 3).
            direction (torch.Tensor): The direction of the rays. Shape (B, T, H, W, 3).
        """
        self._origin = origin
        self._direction = direction

    def to_tensor(self, use_plucker: bool = False) -> torch.Tensor:
        """
        Returns the rays represented as a tensor.
        Args:
            use_plucker (bool): Whether to use Plücker coordinates or not.
        Returns:
            torch.Tensor: The rays tensor. Shape (B, T, H, W, 6).
        """
        if not use_plucker:
            return torch.cat([self._origin, self._direction], dim=-1)

        # Plücker coordinates
        direction = self._direction / self._direction.norm(dim=-1, keepdim=True)
        moment = torch.cross(self._origin, direction, dim=-1)
        return torch.cat([direction, moment], dim=-1)

    @staticmethod
    def _nerf_pos_encoding(x: torch.Tensor, freq: int) -> torch.Tensor:
        scale = (
            2 ** torch.linspace(0, freq - 1, freq, device=x.device, dtype=x.dtype)
            * math.pi
        )
        encoding = rearrange(x[..., None] * scale, "b t h w i s -> b t h w (i s)")
        return torch.sin(torch.cat([encoding, encoding + 0.5 * math.pi], dim=-1))

    def to_pos_encoding(
        self,
        freq_origin: int = 15,
        freq_direction: int = 15,
        return_rays: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns the rays represented as positional encoding. Follows NeRF to map the rays into a higher-dimensional space.
        Args:
            freq_origin (int): The frequency for the origin.
            freq_direction (int): The frequency for the direction.
            return_rays (bool): Whether to return the rays tensor or not.
        Returns:
            torch.Tensor: The rays tensor. Shape (B, T, H, W, 6 * (freq_origin + freq_direction)).
        """
        encoding = torch.cat(
            [
                self._nerf_pos_encoding(self._origin, freq_origin),
                self._nerf_pos_encoding(self._direction, freq_direction),
            ],
            dim=-1,
        )
        rays_tensor = self.to_tensor(use_plucker=False) if return_rays else None
        return encoding, rays_tensor


class CameraPose:
    """
    A class to represent the batched camera poses.
    """

    # pylint: disable=invalid-name

    def __init__(self, R: torch.Tensor, T: torch.Tensor, K: torch.Tensor):
        """
        Args:
            R (torch.Tensor): The rotation matrix. Shape (B, T, 3, 3).
            T (torch.Tensor): The translation vector. Shape (B, T, 3).
            K (torch.Tensor): The intrinsics vector. Shape (B, T, 4).
        """
        self._R = R
        self._T = T
        self._K = K

    @classmethod
    def from_vectors(cls, raw_camera_poses: torch.Tensor):
        """
        Creates a CameraPose object from the raw camera poses.
        Args:
            raw_camera_poses (torch.Tensor): The raw camera poses. Shape (B, T, 4 + 12). The first 4 elements are the intrinsics, the next 12 elements are the flattened extrinsics.
        Returns:
            CameraPose: The CameraPose object.
        """
        K, RT = raw_camera_poses.split([4, 12], dim=-1)
        RT = rearrange(RT, "b t (i j) -> b t i j", i=3, j=4)
        R = RT[..., :3, :3]
        T = RT[..., :3, 3]
        return cls(R, T, K)

    def _normalize_by(self, R_ref: torch.Tensor, T_ref: torch.Tensor) -> None:
        """
        Normalizes so that the camera given by R_ref and T_ref becomes the world coordinates.
        Args:
            R_ref (torch.Tensor): The rotation matrix. Shape (B, 3, 3).
            T_ref (torch.Tensor): The translation vector. Shape (B, 3).
        """
        R_inv = rearrange(R_ref, "b i j -> b j i")
        self._R = einsum(self._R, R_inv, "b t i j1, b j1 j2 -> b t i j2")
        self._T = self._T - einsum(self._R, T_ref, "b t i j, b j -> b t i")

    def normalize_by_first(self) -> None:
        """
        Normalizes the camera poses by the first camera, i.e. computes the relative poses w.r.t. the first camera.
        After normalization, the first camera will have identity rotation and zero translation, i.e. the first camera = world coordinates.
        """
        self._normalize_by(self._R[:, 0], self._T[:, 0])

    def normalize_by_mean(self) -> None:
        """
        Normalizes the camera poses by the mean of all cameras, i.e. computes the relative poses w.r.t. the mean frame.
        The mean camera becomes the world coordinates.
        """
        # convert to quaternions, average them, and convert back to rotation matrices
        q = roma.rotmat_to_unitquat(self._R)
        q_mean = q.mean(dim=1)
        R_mean = roma.unitquat_to_rotmat(q_mean)
        # average translation on world coordinates
        # R_mean^T @ T_mean = mean(sum(R_i^T @ T_i))
        T_world_mean = einsum(
            rearrange(self._R, "b t i j -> b t j i"), self._T, "b t i j, b t j -> b t i"
        ).mean(dim=1)
        T_mean = einsum(R_mean, T_world_mean, "b i j, b j -> b i")
        self._normalize_by(R_mean, T_mean)

    def scale_within_bounds(self, bounds: float = 1.0) -> None:
        """
        Scales the camera locations, so that they are within the boundary box [-bounds, bounds]^3.
        Each scene is scaled independently, while each frame within a scene is scaled by the same factor.
        Args:
            bounds (float): The boundary box. Requires bounds > 0.
        """
        # simply scale the translation vectors by the same factor
        max_vals = self._T.abs().max(dim=1, keepdim=True).values
        scale = bounds / max_vals.clamp(min=1e-6)
        self._T = self._T * scale

    def replace_with_interpolation(self, mask: torch.Tensor) -> None:
        """
        For each sequence in the batch,
        replaces the invalid camera poses (mask == True) by interpolating
        between the nearest valid camera poses (mask == False).
        Args:
            mask (torch.Tensor): The mask for the camera poses to replace. Shape (B, T).
        """
        q = roma.rotmat_to_unitquat(self._R)  # (B, T, 4)
        T = self._T.clone()

        for b in range(mask.shape[0]):
            curr_mask = mask[b]
            if not curr_mask.any() or curr_mask.all():
                continue
            valid_ts = torch.where(~curr_mask)[0]

            if valid_ts[0] != 0:
                q[b, : valid_ts[0]] = q[b, valid_ts[0]]
                T[b, : valid_ts[0]] = T[b, valid_ts[0]]
            if valid_ts[-1] != mask.shape[1] - 1:
                q[b, valid_ts[-1] + 1 :] = q[b, valid_ts[-1]]
                T[b, valid_ts[-1] + 1 :] = T[b, valid_ts[-1]]

            for left_t, right_t in zip(valid_ts[:-1], valid_ts[1:]):
                if right_t - left_t == 1:
                    continue
                left_q, right_q = q[b, [left_t, right_t]]
                q[b, left_t : right_t + 1] = roma.unitquat_slerp(
                    left_q,
                    right_q,
                    torch.linspace(0, 1, right_t - left_t + 1, device=q.device),
                )
                T[b, left_t : right_t + 1] = torch.lerp(
                    T[b, left_t],
                    T[b, right_t],
                    torch.linspace(
                        0, 1, right_t - left_t + 1, device=T.device
                    ).unsqueeze(-1),
                )

        self._R = roma.unitquat_to_rotmat(q)
        self._T = T

    def extrinsics(self, flatten: bool = False) -> torch.Tensor:
        """
        Returns the extrinsics matrix [R | T] for the camera poses.
        Args:
            flatten (bool): Whether to flatten the extrinsics matrix.
        Returns:
            torch.Tensor: The extrinsics matrix. Shape (B, T, 12) if flatten is True, else (B, T, 3, 4).
        """
        extrinsics = torch.cat(
            [self._R, rearrange(self._T, "b t i -> b t i 1")], dim=-1
        )
        return rearrange(extrinsics, "b t i j -> b t (i j)") if flatten else extrinsics

    def intrinsics(self, flatten: bool = False) -> torch.Tensor:
        """
        Returns the intrinsics matrix for the camera poses.
        Args:
            flatten (bool): Whether to flatten the intrinsics matrix.
        Returns:
            torch.Tensor: The intrinsics matrix. Shape (B, T, 3, 3) if flatten is False, else (B, T, 4).
        """
        if flatten:
            return self._K
        else:
            K = repeat(
                torch.eye(3, device=self._K.device),
                "i j -> b t i j",
                b=self._K.shape[0],
                t=self._K.shape[1],
            )
            K[:, :, 0, 0] = self._K[:, :, 0]
            K[:, :, 1, 1] = self._K[:, :, 1]
            K[:, :, 0, 2] = self._K[:, :, 2]
            K[:, :, 1, 2] = self._K[:, :, 3]
            return K

    def rays(self, resolution: int) -> Ray:
        """
        Returns the rays for the camera poses.
        Args:
            resolution (int): The resolution of the image.
        Returns:
            Ray: The rays object.
        """

        # Direction
        # compute ray direction in camera coordinates
        coord_w, coord_h = torch.meshgrid(
            torch.linspace(
                0,
                resolution - 1,
                resolution,
                device=self._K.device,
                dtype=self._K.dtype,
            ),
            torch.linspace(
                0,
                resolution - 1,
                resolution,
                device=self._K.device,
                dtype=self._K.dtype,
            ),
            indexing="xy",
        )
        coord_w = rearrange(coord_w, "h w -> 1 1 h w") + 0.5  # (1, 1, H, W)
        coord_h = rearrange(coord_h, "h w -> 1 1 h w") + 0.5  # (1, 1, H, W)

        fx, fy, px, py = rearrange(self._K * resolution, "b t i -> b t i 1").chunk(
            4, dim=-2
        )
        x = (coord_w - px) / fx
        y = (coord_h - py) / fy
        z = torch.ones_like(x)
        direction = torch.stack([x, y, z], dim=-1)
        # convert to world coordinates
        R_inv = rearrange(self._R, "b t i j -> b t j i")
        direction = einsum(
            R_inv,
            direction,
            "b t i j, b t h w j -> b t h w i",
        )

        # Origin
        origin = -einsum(R_inv, self._T, "b t i j, b t j -> b t i")
        origin = repeat(
            origin, "b t i -> b t h w i", h=resolution, w=resolution
        ).clone()
        return Ray(origin, direction)
