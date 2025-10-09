import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'revisit_error','third_party','common_metrics_on_video_quality'))

from third_party.common_metrics_on_video_quality.calculate_lpips import loss_fn as img_lpips_loss_fn
from third_party.common_metrics_on_video_quality.calculate_psnr import img_psnr
from third_party.common_metrics_on_video_quality.calculate_ssim import calculate_ssim_function