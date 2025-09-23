from enum import Enum


class VideoMetricType(str, Enum):
    LPIPS = "lpips"
    FID = "fid"
    FVD = "fvd"
    MSE = "mse"
    SSIM = "ssim"
    PSNR = "psnr"
    IS = "is"
    REAL_IS = "real_is"
    FVMD = "fvmd"
    VBENCH = "vbench"
    REAL_VBENCH = "real_vbench"


class VideoMetricModelType(str, Enum):
    LPIPS = "Lpips"
    INCEPTION_V3 = "InceptionV3"
    I3D = "I3D"
    PIPS = "PIPS"
    CLIP_B_32 = "CLIP_B_32"
    CLIP_L_14 = "CLIP_L_14"
    DINO = "DINO"
    LAION = "LAION"
    MUSIQ = "MUSIQ"
    RAFT = "RAFT"
    AMT_S = "AMT_S"
