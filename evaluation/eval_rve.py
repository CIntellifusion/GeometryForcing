#!/usr/bin/env python3
"""
Video Evaluation Script for DFOT-VGGT using Reprojection Error Metrics
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch

from revisit_error import img_psnr, calculate_ssim_function, img_lpips_loss_fn
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image

def split_gif_to_videos(gif_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a side-by-side GIF into generated and ground truth video tensors."""
    gif_frames = iio.imread(gif_path)
    
    # Ensure proper format
    if len(gif_frames.shape) == 3:
        gif_frames = gif_frames[None, ...]
        
    if gif_frames.shape[-1] == 1:  # Grayscale to RGB
        gif_frames = np.repeat(gif_frames, 3, axis=-1)
    elif gif_frames.shape[-1] == 4:  # Remove alpha
        gif_frames = gif_frames[..., :3]
    
    # Split horizontally and convert to tensors
    mid_width = gif_frames.shape[2] // 2
    gen_frames = gif_frames[:, :, :mid_width, :]
    
    # Convert to torch tensors (T, C, H, W) in range [0, 1]
    gen_tensor = torch.from_numpy(gen_frames).float().permute(0, 3, 1, 2) / 255.0
    
    return gen_tensor

def save_tensor_as_video(tensor: torch.Tensor, video_path: str, fps: int = 8):
    """Save a video tensor as an MP4 file."""
    video_np = tensor.permute(0, 2, 3, 1).numpy()
    video_np = (video_np * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    iio.imwrite(video_path, video_np, fps=fps)

def save_video_frames(video_tensor: torch.Tensor, output_dir: str) -> List[str]:
    """Save video frames as images and return list of paths."""
    os.makedirs(output_dir, exist_ok=True)
    video_np = video_tensor.permute(0, 2, 3, 1).numpy()
    video_np = (video_np * 255).astype(np.uint8)
    
    frame_paths = []
    for i, frame in enumerate(video_np):
        frame_path = os.path.join(output_dir, f"frame_{i:06d}.png")
        iio.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    return frame_paths


def calculate_revisisting_error(gen_video: torch.Tensor) -> Dict[str, float]:
    """
    Calculate revisiting error metrics by comparing first and last frames.
    
    Args:
        gen_video: torch.Tensor of shape [T, C, H, W] with values in [0, 1]
        
    Returns:
        Dictionary containing revisiting error metrics
    """
    if len(gen_video.shape) != 4:
        raise ValueError(f"Expected 4D tensor [T, C, H, W], got shape {gen_video.shape}")
    
    if gen_video.shape[0] < 2:
        print(f"Warning: Video has less than 2 frames ({gen_video.shape[0]}), cannot calculate revisiting error")
        return {
            'revisiting_psnr': float('nan'),
            'revisiting_ssim': float('nan'), 
            'revisiting_lpips': float('nan'),
            'revisiting_fid': float('nan')
        }
    
    # Get first and last frames
    first_frame = gen_video[0:1]  # [1, C, H, W]
    last_frame = gen_video[-1:]   # [1, C, H, W]
    
    metrics = {}
    
    try:
        # Calculate PSNR
        if img_psnr is not None:
            first_np = first_frame[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
            last_np = last_frame[0].cpu().numpy().transpose(1, 2, 0)    # [H, W, C]
            psnr_val = img_psnr(first_np, last_np)
            metrics['revisiting_psnr'] = float(psnr_val)
        else:
            metrics['revisiting_psnr'] = float('nan')
        
        # Calculate SSIM
        if calculate_ssim_function is not None:
            first_ssim = first_frame[0].cpu().numpy()  # [C, H, W]
            last_ssim = last_frame[0].cpu().numpy()    # [C, H, W]
            ssim_val = calculate_ssim_function(first_ssim, last_ssim)
            metrics['revisiting_ssim'] = float(ssim_val)
        else:
            metrics['revisiting_ssim'] = float('nan')
        
        # Calculate LPIPS
        if img_lpips_loss_fn is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            img_lpips_loss_fn.to(device)
            img_lpips_loss_fn.eval()
            
            # Normalize to [-1, 1] for LPIPS
            first_lpips = (first_frame * 2 - 1).to(device)
            last_lpips = (last_frame * 2 - 1).to(device)
            
            with torch.no_grad():
                lpips_val = img_lpips_loss_fn(first_lpips, last_lpips).mean().item()
            metrics['revisiting_lpips'] = float(lpips_val)
        else:
            metrics['revisiting_lpips'] = float('nan')
        
        # Calculate FID
        if calculate_metrics is not None and save_image is not None:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    first_dir = os.path.join(temp_dir, "first")
                    last_dir = os.path.join(temp_dir, "last")
                    os.makedirs(first_dir, exist_ok=True)
                    os.makedirs(last_dir, exist_ok=True)
                    
                    # For FID with single images, we need multiple copies
                    for i in range(50):  # Create multiple copies for stable FID calculation
                        save_image(first_frame[0], os.path.join(first_dir, f"frame_{i:03d}.png"))
                        save_image(last_frame[0], os.path.join(last_dir, f"frame_{i:03d}.png"))
                    
                    fid_metrics = calculate_metrics(
                        input1=first_dir, 
                        input2=last_dir,
                        cuda=torch.cuda.is_available(),
                        isc=False,
                        fid=True,
                        kid=False
                    )
                    metrics['revisiting_fid'] = float(fid_metrics['frechet_inception_distance'])
            except Exception as e:
                print(f"Warning: Could not calculate FID: {e}")
                metrics['revisiting_fid'] = float('nan')
        else:
            metrics['revisiting_fid'] = float('nan')
            
    except Exception as e:
        print(f"Error calculating revisiting metrics: {e}")
        metrics.update({
            'revisiting_psnr': float('nan'),
            'revisiting_ssim': float('nan'),
            'revisiting_lpips': float('nan'),
            'revisiting_fid': float('nan')
        })
    
    return metrics
    

def process_single_gif(gif_path: str, output_dir: str, temp_dir: Optional[str]) -> Dict[str, Any]:
    """Process a single GIF file and calculate metrics."""
    gif_name = Path(gif_path).stem
    print(f"Processing {gif_name}...")
    
    # Split GIF into video tensors
    gen_video = split_gif_to_videos(gif_path)
    
    # Optionally save videos for visualization
    gen_video_path = None
    if temp_dir:
        gen_video_path = os.path.join(temp_dir, "generated", f"{gif_name}_generated.mp4")
        save_tensor_as_video(gen_video, gen_video_path)
    
    # Calculate metrics
    metrics = calculate_revisisting_error(gen_video)
    
    # Print metrics
    for key, value in metrics.items():
        if isinstance(value, float) and not np.isnan(value):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: N/A")
    print()
    
    return {
        'gif_file': gif_name,
        'gif_path': gif_path,
        'generated_video': gen_video_path,
        'video_shape': list(gen_video.shape),
        'metrics': metrics
    }
        

def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate aggregate metrics across all processed videos."""
    valid_results = [r for r in results if 'metrics' in r and r['metrics']]
    if not valid_results:
        return {}
    
    # Get all metric keys from first valid result
    metric_keys = valid_results[0]['metrics'].keys()
    aggregated = {}
    
    for key in metric_keys:
        values = [r['metrics'][key] for r in valid_results if key in r['metrics'] and not np.isnan(r['metrics'][key])]
        if values:
            aggregated.update({
                f'{key}_mean': float(np.mean(values)),
                f'{key}_std': float(np.std(values)),
                f'{key}_median': float(np.median(values))
            })
    
    return aggregated


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def parse_latest_checkpoint_val_video(gif_files: List[Path]) -> List[Path]:
    """
    Filter GIF files to only include those from the latest checkpoint.
    
    Video naming pattern: video_54_419_fc115c5889bd8293f1ae.gif
    - 54: video number
    - 419: log number (checkpoint)
    - For each video number, select the one with the highest log number
    
    Args:
        gif_files: List of Path objects for GIF files
        
    Returns:
        List of Path objects for latest checkpoint validation videos
    """
    if not gif_files:
        return []
    
    # Dictionary to store video_number -> (max_log_number, file_path)
    latest_videos = {}
    
    # Pattern to match video naming: video_{video_num}_{log_num}_{hash}.gif
    pattern = r'video_(\d+)_(\d+)_([a-f0-9]+)\.gif'
    
    for gif_file in gif_files:
        filename = gif_file.name
        match = re.match(pattern, filename)
        
        if match:
            video_num = int(match.group(1))
            log_num = int(match.group(2))
            hash_part = match.group(3)
            
            # If this video number hasn't been seen, or this log number is higher
            if video_num not in latest_videos or log_num > latest_videos[video_num][0]:
                latest_videos[video_num] = (log_num, gif_file)
    
    # Extract the file paths for latest checkpoints
    latest_checkpoint_files = [file_path for _, file_path in latest_videos.values()]
    
    # Sort by video number for consistent ordering
    latest_checkpoint_files.sort(key=lambda x: int(re.match(pattern, x.name).group(1)))
    
    print(f"Found {len(latest_checkpoint_files)} latest checkpoint validation videos:")
    for file_path in latest_checkpoint_files:
        match = re.match(pattern, file_path.name)
        if match:
            video_num, log_num = match.group(1), match.group(2)
            print(f"  Video {video_num}: Log_index {log_num} -> {file_path.name}")
    
    return latest_checkpoint_files

def main():
    parser = argparse.ArgumentParser(description='Evaluate video quality from GIF predictions')
    parser.add_argument('--gif_dir', type=str, default=None, help='Directory containing GIF files')
    parser.add_argument('--output_dir', type=str, default='./video_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--temp_dir', type=str, default=None,
                       help='Temporary directory for video files (optional)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of GIF files to process')
    parser.add_argument('--pattern', type=str, default='*.gif',
                       help='File pattern to match GIF files')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)
    
    # Find GIF files
    gif_files = list(Path(args.gif_dir).glob(args.pattern))
    gif_files = parse_latest_checkpoint_val_video(gif_files)
    
    
    if args.max_files:
        gif_files = gif_files[:args.max_files]
    
    print(f"Found {len(gif_files)} GIF files to process")
    
    # Process files
    results = []
    for gif_path in gif_files:
        result = process_single_gif(str(gif_path), args.output_dir, args.temp_dir)
        results.append(result)
    
    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(results)
    
    # Prepare and save results
    detailed_results = {
        'individual_results': results,
        'aggregate_metrics': aggregate_metrics,
        'summary': {
            'total_files': len(gif_files),
            'successful_files': len([r for r in results if 'metrics' in r and r['metrics']]),
            'failed_files': len([r for r in results if 'error' in r or not r.get('metrics')])
        },
        'config': vars(args)
    }
    
    # Save results
    detailed_results = convert_numpy_types(detailed_results)
    output_file = os.path.join(args.output_dir, 'video_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(gif_files)}")
    print(f"Successful: {detailed_results['summary']['successful_files']}")
    print(f"Failed: {detailed_results['summary']['failed_files']}")
    
    if aggregate_metrics:
        print("\nAGGREGATE METRICS:")
        print("-" * 40)
        for metric, value in aggregate_metrics.items():
            if '_mean' in metric:
                print(f"{metric}: {value:.4f}")
    
    print(f"\nResults saved to: {output_file}")
    if args.temp_dir:
        print(f"Video files saved to: {args.temp_dir}")


if __name__ == "__main__":
    main()