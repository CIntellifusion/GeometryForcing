import sys
sys.path.append('../')
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.vggt import VGGT
from einops import rearrange
from torch import Tensor
from external.alignment_projector import ConvProjector

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class VGGTAlignmentLoss(nn.Module):
    def __init__(self,
                 alignment_context_length: int = 16,
                 apply_unnormalize_recon: bool = False,
                 unormalize_lambda: float =  1, # lambda for unormalize recon loss
                 latents_info: list = None, # the shape of generative model latents
                 encoder_info: list = [24, 1374, 2048], # the shape of encoder output
                 mid_channels: int = 128,
                 vggt_layer_index=None, # -1 means last layer, -2 means second last layer, etc.
                ):
        super().__init__()
        self.alignment_context_length = alignment_context_length
        self.unormalize_lambda = unormalize_lambda 
        # === 1. 初始化冻结的 VGGT 模型 ===
        self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
        self.vggt_model.eval()
        for p in self.vggt_model.parameters():
            p.requires_grad = False
        # import pdb; pdb.set_trace()
        # === 2. 构建连接器 ===
        # for realestate 10k uvit setting: totally 7 blocks 
        if latents_info is None:
            latents_info = [
                [576, 32, 32],
                [256, 64, 64],
                [128, 128, 128]
            ]
        elif latents_info ==-1 or latents_info ==0 :
            latents_info = [
                [128, 128, 128]
            ]
        elif latents_info == -2 or latents_info == 1: 
            # 倒数第二
            latents_info = [
                [256, 64, 64]
            ]
        elif latents_info == -3 or latents_info == 2:
            latents_info = [
               [576, 32, 32]
            ]
        elif latents_info == -4 or latents_info==3:
            latents_info = [
                [1152, 16, 16],
            ]
        self.latents_info = latents_info 
        self.encoder_info = encoder_info
        self.mid_channels = mid_channels
        self.projectors = nn.ModuleList()
        for c, h, w in self.latents_info:
            projector = ConvProjector(
                in_channels=c,
                in_h=h,
                in_w=w,
                out_h=self.encoder_info[1],
                out_w=self.encoder_info[2],
                mid_channels=mid_channels
            )
            self.projectors.append(projector)
        ## this module aims to recon normalized latent feature to unnormalized feature 
        self.apply_unnormalize_recon = apply_unnormalize_recon
        if self.apply_unnormalize_recon:
            self.feature_unormalizer = nn.Sequential(
                nn.Conv2d(self.encoder_info[0], mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(mid_channels, self.encoder_info[0], kernel_size=1)
            )
    def load_projector_from_ckpt(self,ckpt_path):
        ckpt = torch.load(ckpt_path,map_location='cpu',weights_only=False)
        new_state_dict = {k.replace("alignment_loss.",""):v for k,v in ckpt['state_dict'].items()}
        missing, unexpected = self.load_state_dict(new_state_dict,strict=False)
        # filter missing and unexpected keys 
        # 1. vggt is loaded from pretrained model
        missing = [k for k in missing if not k.startswith("vggt_model.")]
        # 2. model is saved in ckpt but not in here 
        unexpected = [k for k in unexpected if not k.startswith("diffusion_model.")]
        print(f"Load projector from {ckpt_path} successfully. with  missing keys: {missing}, unexpected keys: {unexpected}")
        
    def latent_to_aggregated_token(self,latent, layer_index=-1):
        """
        Args:
            latent: list[Tensor], shape [BxT,C,H,W]
        Returns:
            aggregated_tokens_lists: list[Tensor], shape [B,T,1374,2048]
        """
        # assert self.apply_unnormalize_recon is True, "Please set apply_unnormalize_recon to True to enable unnormalize operation" 
        aggregated_tokens_lists = []
        projector = self.projectors[layer_index] 
        if latent.dim() == 5:
            b, t, c, h, w = latent.shape
            latent = rearrange(latent, 'b t c h w -> (b t) c h w')
        else:
            b, c, h, w = latent.shape
            t = 16
            latent = latent
        projected_latent = projector(latent)  # [BT, c, h , w , ]
        ph =  projected_latent.shape[2]
        # projected_latent = rearrange(projected_latent, '(b t) c h w -> b t c h w', t = self.temporal_length)
        projected_latent_flat = rearrange(projected_latent, 'b c h w -> b c (h w)')  # [B, C, H*W]
        projected_latent_flat_norm = F.normalize(projected_latent_flat, p=2, dim=-1)  # [B, C, H*W]
        projected_latent_norm = rearrange(projected_latent_flat_norm, 'b c (h w) -> b c h w', h=ph)  # [B, C, H, W]
        # unnormalize 
        if self.apply_unnormalize_recon:
            print(f"[external.vggt_alignment_loss][VGGTAlignmentLoss] Applying unnormalize recon operation.")
            pred_gt_latent = self.feature_unormalizer(projected_latent_norm)  # [B,T,24,512,512] 
        else:
            print(f"[external.vggt_alignment_loss][VGGTAlignmentLoss] without unnormalize recon operation.")
            pred_gt_latent = projected_latent_norm
        # uninterpolate from 512 512 -> 1374 2048
        pred_gt_latent = F.interpolate(pred_gt_latent, size=(1374, 2048), mode='bilinear', align_corners=False)
        pred_gt_latent = rearrange(pred_gt_latent, '(b t) c h w -> b t c h w', h=1374,t=t)  # [B, C, H, W]
        aggregated_tokens_list = [pred_gt_latent[:,:,i,:,:] for i in range(pred_gt_latent.shape[2])] # each item is [B,T,1374,2048] 
        aggregated_tokens_lists.append(aggregated_tokens_list)
        return aggregated_tokens_lists 
    
    def forward_vggt_prediction(self, images,latent_list,layer_index=-1,return_original_predictions=False): 
        """
        Args:
            images: original images, shape [B, T, C, H, W]
            latent_list: list[Tensor], shape [B,T,24,1374,2048]  #
        intermidiate output:
            aggregated_tokens_list: List[B,N,1374,2048],len(latents)=24
        Returns:
            point_cloud 
        """
        images = self.vggt_processor(images)  # [B, T, C, H, W]
        latent = latent_list[-1]
        # import pdb; pdb.set_trace()
        aggregated_tokens_lists = self.latent_to_aggregated_token(latent,layer_index)  # list[Tensor], shape [B,T,1374,2048]
        # print(f"[external.vggt_alignment_loss][VGGTAlignmentLoss][forward_vggt_prediction] Default Using Last Layer Output")
        aggregated_tokens_list = aggregated_tokens_lists[0]
        predictions,original_predictions = self.vggt_model.forward_with_external_feature(
            images,
            aggregated_tokens_list=aggregated_tokens_list,
            return_original_predictions=return_original_predictions
        )
        return predictions ,original_predictions
        
    def vggt_processor(self, images: Tensor): 
        # images: b t c h w 
        b = images.shape[0] 
        images = rearrange(images, 'b f c h w -> (b f) c h w')
        # print(f"[external.vggt_alignment_loss][VGGTAlignmentLoss][vggt_processor] using imagesize 224x224")
        # using a  224x224 will cause drop from 317->327(fvd)
        images_resized = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        images_resized = rearrange(images_resized, '(b f) c h w -> b f c h w', b=b)
        images_resized = torch.clamp(images_resized, 0.0, 1.0)
        return images_resized
    
    def forward(self, latents, images):
        """
        Args:
            latents from UViT :  (B, T * H * W, C)  T*H*W = patches 
            latents: list[Tensor], shape [B,T,C,H,W]
            images: Tensor, [B, C, H, W]，原始图像
        Returns:
            alignment_loss: 对齐损失（平均 cosine 距离）
        """
        images = images[:, :self.alignment_context_length, :, :, :]  # [B, T, C, H, W]
        images = self.vggt_processor(images) # [B, 3, T, H, W]
        with torch.no_grad():
            # target_feats = self.vggt_model.encode(images)  # list[Tensor], shape [B,T,1374, 2048]
            aggregated_tokens_list, patch_start_idx = self.vggt_model.shortcut_forward(images) #  24x []
            # interpolate to few dimension 
            target_h,target_w =  self.encoder_info[1], self.encoder_info[2]  # 1374, 2048
            aggregated_tokens_list = [F.interpolate(tokens, size=(target_h, target_w), mode='bilinear', align_corners=False) for tokens in aggregated_tokens_list]
            aggregated_tokens = torch.stack(aggregated_tokens_list)  # [B*T, 512, 512]
            target_feats = rearrange(aggregated_tokens,'c b t h w -> b t c h w')
            
        alignment_loss = 0.
        unormalize_loss = 0. 
        assert len(latents) == len(self.projectors), f"latents length {len(latents)} should match projectors length {len(self.projectors)}"
        
        for latent, projector in zip(latents, self.projectors):
            # print(f"[externl.vggt_alignment_loss][VGGTAlignmentLoss][forward] latent shape: {latent.shape}, projector: {projector}")
            latent = latent[:,:self.alignment_context_length,...]
            latent = rearrange(latent, 'b t c h w -> (b t) c h w')  # [B*T, C, H, W]
            
            target_feat = rearrange(target_feats, 'b t c h w -> (b t) c h w')  # [B, C, H, W]
            # target feature : torch.Size([16, 24, 512, 512])
            latent_proj = projector(latent)  # [BT,  C ,H, W]
            H,W = latent_proj.shape[2], latent_proj.shape[3]
            # flatten空间维度
            latent_proj_flat = rearrange(latent_proj, 'b c h w -> b c (h w)')
            target_feat_flat = rearrange(target_feat, 'b c h w -> b c (h w)')
            # 假设 T是通道，S是空间flatten：
            latent_proj_norm = F.normalize(latent_proj_flat, p=2, dim=-1)   # 归一化通道维
            target_feat_norm = F.normalize(target_feat_flat, p=2, dim=-1) # B C H*W 

            # 计算每个空间点的向量相似度（点积），
            # sum over通道T维度得到空间向量相似度，shape: [B, S]
            alignment_loss += mean_flat(-(latent_proj_norm * target_feat_norm)).sum(dim=-1)  # [B, S]
            if self.apply_unnormalize_recon:
                # 计算 unnormalize loss
                latent_proj_norm = rearrange(latent_proj_norm, 'b c (h w) -> b c h w ',h = H)  # [B, C, S, 1] 
                # import pdb; pdb.set_trace() 
                unormalized_latent_proj = self.feature_unormalizer(latent_proj_norm)
                # mse loss 
                unormalize_loss += F.mse_loss(unormalized_latent_proj, target_feat)
                
        alignment_loss /= len(latents)
        if self.apply_unnormalize_recon:
            unormalize_loss /= len(latents)
            unormalize_lambda = self.unormalize_lambda
            unormalize_loss = unormalize_lambda * unormalize_loss 
            loss = alignment_loss + unormalize_loss
        else:
            loss = alignment_loss
        return loss
