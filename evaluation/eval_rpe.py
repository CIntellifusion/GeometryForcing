#!/usr/bin/env python3
"""
Video Evaluation Script for DFOT-VGGT using Reprojection Error Metrics
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch

from evaluation.reprojection_error import ReprojectionErrorMetric


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
    gt_frames = gif_frames[:, :, mid_width:, :]
    
    # Convert to torch tensors (T, C, H, W) in range [0, 1]
    gen_tensor = torch.from_numpy(gen_frames).float().permute(0, 3, 1, 2) / 255.0
    gt_tensor = torch.from_numpy(gt_frames).float().permute(0, 3, 1, 2) / 255.0
    
    return gen_tensor, gt_tensor

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


def calculate_reprojection_error(gen_video: torch.Tensor, gt_video: torch.Tensor) -> Dict[str, float]:
    """Calculate reprojection error metrics for generated and ground truth videos."""    
    metrics = {}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames and calculate metrics for both videos
            gen_frames_dir = os.path.join(temp_dir, "gen_frames")
            gt_frames_dir = os.path.join(temp_dir, "gt_frames")
            
            gen_image_paths = save_video_frames(gen_video, gen_frames_dir)
            gt_image_paths = save_video_frames(gt_video, gt_frames_dir)
            
            # Calculate reprojection errors
            metric = ReprojectionErrorMetric()
            gen_error = metric.compute_scores(gen_image_paths)
            gt_error = metric.compute_scores(gt_image_paths)
            
            metrics.update({
                'reprojection_error_gen': gen_error,
                'reprojection_error_gt': gt_error,
                'reprojection_error_relative': gen_error / gt_error if gt_error > 0 else (float('inf') if gen_error > 0 else 1.0)
            })
            
    except Exception as e:
        print(f"Error calculating reprojection error: {e}")
        metrics.update({
            'reprojection_error_gen': float('nan'),
            'reprojection_error_gt': float('nan'),
            'reprojection_error_relative': float('nan')
        })
    
    return metrics
    

def process_single_gif(gif_path: str, output_dir: str, temp_dir: Optional[str]) -> Dict[str, Any]:
    """Process a single GIF file and calculate metrics."""
    gif_name = Path(gif_path).stem
    print(f"Processing {gif_name}...")
    
    # Split GIF into video tensors
    gen_video, gt_video = split_gif_to_videos(gif_path)
    
    # Optionally save videos for visualization
    gen_video_path = gt_video_path = None
    if temp_dir:
        gen_video_path = os.path.join(temp_dir, "generated", f"{gif_name}_generated.mp4")
        gt_video_path = os.path.join(temp_dir, "ground_truth", f"{gif_name}_gt.mp4")
        save_tensor_as_video(gen_video, gen_video_path)
        save_tensor_as_video(gt_video, gt_video_path)
    
    # Calculate metrics
    metrics = calculate_reprojection_error(gen_video, gt_video)
    
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
        'gt_video': gt_video_path,
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