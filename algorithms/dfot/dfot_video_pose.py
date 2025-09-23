import torch
from omegaconf import OmegaConf
from typing import Optional
from torch import Tensor
from omegaconf import DictConfig
from einops import rearrange
from utils.geometry_utils import CameraPose
from .dfot_video import DFoTVideo
from transformers import get_scheduler
from external import VGGTAlignmentLoss
from torch.nn import functional  as F 
from typing import Optional, Any, Dict, Literal, Callable, Tuple
from .history_guidance import HistoryGuidance
from tqdm import tqdm
from einops import rearrange, repeat, reduce
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio

class DFoTVideoPose(DFoTVideo):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for pose-conditioned video generation.
    """

    def __init__(self, cfg: DictConfig):
        self.camera_pose_conditioning = cfg.camera_pose_conditioning
        self.conditioning_type = cfg.camera_pose_conditioning.type
        self._check_cfg(cfg)
        self._update_backbone_cfg(cfg)
        super().__init__(cfg)
        # import pdb; pdb.set_trace()
        
    def _check_cfg(self, cfg: DictConfig):
        """
        Check if the config is valid
        """
        if cfg.backbone.name not in {"dit3d_pose", "u_vit3d_pose","u_vit3d_pose_controlnet"}:
            raise ValueError(
                f"DiffusionForcingVideo3D only supports backbone 'dit3d_pose' or 'u_vit3d_pose_controlnet' 'dit3d_pose_controlnet', got {cfg.backbone.name}"
            )

        if (
            cfg.backbone.name == "dit3d_pose"
            and self.conditioning_type == "global"
            and cfg.backbone.conditioning.modeling != "film"
        ):
            raise ValueError(
                f"When using global camera pose conditioning, `algorithm.backbone.conditioning.modeling` should be 'film', got {cfg.backbone.conditioning.modeling}"
            )
        if (cfg.backbone.name in ["u_vit3d_pose","u_vit3d_pose_controlnet"]) and self.conditioning_type == "global":
            raise ValueError(
                "Global camera pose conditioning is not supported for U-ViT3DPose"
            )
        ## adding additional control to diffusion backbone 

    def _update_backbone_cfg(self, cfg: DictConfig):
        """
        Update backbone config with camera pose conditioning
        """
        conditioning_dim = None
        match self.conditioning_type:
            case "global":
                conditioning_dim = 12
            case "ray" | "plucker":
                conditioning_dim = 6
            case "ray_encoding":
                conditioning_dim = 180
            case _:
                raise ValueError(
                    f"Unknown camera pose conditioning type: {self.conditioning_type}"
                )
        cfg.backbone.conditioning.dim = conditioning_dim

    @torch.no_grad()
    @torch.autocast(
        device_type="cuda", enabled=False
    )  # force 32-bit precision for camera pose processing
    def _process_conditions(
        self, conditions: Tensor, noise_levels: Optional[Tensor] = None
    ) -> Tensor:
        """
        Process conditions (raw camera poses) to desired format for the model
        Args:
            conditions (Tensor): raw camera poses (B, T, 12)
        """
        # import pdb; pdb.set_trace()
        camera_poses = CameraPose.from_vectors(conditions)
        if self.cfg.tasks.prediction.history_guidance.name == "temporal":
            # NOTE: when using temporal history guidance,
            # some frames are fully masked out and thus their camera poses are not needed
            # so we replace them with interpolated camera poses from the nearest non-masked frames
            # this is important b/c we normalize camera poses by the first frame
            camera_poses.replace_with_interpolation(
                mask=noise_levels == self.timesteps - 1
            )

        match self.camera_pose_conditioning.normalize_by:
            case "first":
                camera_poses.normalize_by_first()
            case "mean":
                camera_poses.normalize_by_mean()
            case _:
                raise ValueError(
                    f"Unknown camera pose normalization method: {self.camera_pose_conditioning.normalize_by}"
                )

        if self.camera_pose_conditioning.bound is not None:
            camera_poses.scale_within_bounds(self.camera_pose_conditioning.bound)

        match self.conditioning_type:
            case "global":
                return camera_poses.extrinsics(flatten=True)
            case "ray" | "ray_encoding" | "plucker":
                rays = camera_poses.rays(resolution=self.x_shape[1])
                if self.conditioning_type == "ray_encoding":
                    rays = rays.to_pos_encoding()[0]
                else:
                    rays = rays.to_tensor(
                        use_plucker=self.conditioning_type == "plucker"
                    )
                return rearrange(rays, "b t h w c -> b t c h w")

        
class DFoTGeometryForcing(DFoTVideoPose):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for pose-conditioned video generation.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # init loss class 
        self.build_alignment_module(cfg)
        
    def build_alignment_module(self,cfg):
        # load a naive vggt model 
        alignment_config = cfg.alignment 
        # alignment_config 是 DictConfig
        alignment_dict = OmegaConf.to_container(alignment_config, resolve=True)  # 转成 dict，resolve=True 会把 interpolation 展开
        encoder_type = alignment_dict.pop("encoder_type", "vggt")
        self.alignment_coeff = alignment_dict.pop('alignment_coeff',0.5) # alignment default 0.5 
        if self.alignment_coeff > 0:
            self.encoder_type = encoder_type  # 保存类型，用于之后 loss 判断
            self.vggt_layer = None 
            self.dino_layer = None 
            if encoder_type == "vggt":
                self.vggt_alignment_loss = VGGTAlignmentLoss(**alignment_dict)
                self.vggt_layer = alignment_dict["latents_info"]
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")
            
    def alignment_loss(self, latents_list, context_images):
        """
        latents_list: list of latents, each is [b, 16, c, h, w]
        context_images: [b, 1, c, h, w]
        """
        if self.vggt_layer is not None: 
            vggt_latents_list = [latents_list[self.vggt_layer]]
        else:
            print(f"[DFoTGeometryForcing] using default vggt layer for alignment loss: {self.vggt_layer}")
            vggt_latents_list = latents_list[-3:]
            
        if self.encoder_type == "vggt":
            return self.vggt_alignment_loss(vggt_latents_list, context_images)
        else:
            raise ValueError(f"[alignment_loss] Unknown encoder_type: {self.encoder_type}")
        
    def training_step(self, batch, batch_idx, namespace="training"):
        """Training step"""
        xs, conditions, masks, *_ = batch
        # import pdb; pdb.set_trace()
        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        # xs_pred, loss = self.diffusion_model(xs,self._process_conditions(conditions),k=noise_levels)
        xs_pred, loss,latents_list = self.diffusion_model.forward_with_alignment_loss(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
        )
    
        diffusion_loss_mean = self._reweight_loss(loss, masks)
        if self.alignment_coeff > 0:
            context_images = xs # all gt frames are given 
            context_images = self._unnormalize_x(context_images) # [b, 1, c, h, w] 
            alignment_context_images = context_images
            alignment_loss = self.alignment_loss(latents_list,alignment_context_images)
            alignment_loss_mean = alignment_loss.mean()
            loss = diffusion_loss_mean + alignment_loss_mean * self.alignment_coeff
        else: 
            # print(f"[DFoTGeometryForcing][training_step] alignment_coeff is 0, no alignment loss will be applied.")
            loss = diffusion_loss_mean

        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(
                f"{namespace}/loss",
                loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
                prog_bar=True, 
            )
            self.log(
                f"{namespace}/diffusion_loss",
                diffusion_loss_mean,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )
            if self.alignment_coeff > 0:
                self.log(
                    f"{namespace}/alignment_loss",
                    alignment_loss_mean,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                    prog_bar=True, 
                )

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))
        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }
        return output_dict
