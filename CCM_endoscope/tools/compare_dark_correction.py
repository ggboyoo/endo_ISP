import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Robust image read supporting non-ASCII Windows paths"""
    if not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def read_raw_image(file_path: str, width: int, height: int, data_type: str) -> np.ndarray:
    """Read RAW image data"""
    try:
        if data_type == "uint16":
            raw_data = np.fromfile(file_path, dtype=np.uint16)
        elif data_type == "uint8":
            raw_data = np.fromfile(file_path, dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Reshape to image dimensions
        expected_size = width * height
        if len(raw_data) != expected_size:
            print(f"Warning: Expected {expected_size} pixels, got {len(raw_data)}")
            # Truncate or pad as needed
            if len(raw_data) > expected_size:
                raw_data = raw_data[:expected_size]
            else:
                raw_data = np.pad(raw_data, (0, expected_size - len(raw_data)), 'constant')
        
        return raw_data.reshape((height, width))
    except Exception as e:
        print(f"Error reading RAW file: {e}")
        return None


    """Apply dark current correction"""
def apply_dark_correction(raw_data: np.ndarray, dark_data: np.ndarray) -> np.ndarray:
    print(f"  Applying dark current correction...")
    print(f"    Original range: {np.min(raw_data)} - {np.max(raw_data)}")
    print(f"    Dark range: {np.min(dark_data)} - {np.max(dark_data)}")
    
    # Ensure same data type for subtraction
    if raw_data.dtype != dark_data.dtype:
        raw_data = raw_data.astype(dark_data.dtype)
    
    # Subtract dark current
    corrected_data = raw_data.astype(np.float64) - dark_data.astype(np.float64)
    
    # Clip negative values to 0
    corrected_data = np.clip(corrected_data, 0, None)
    
    print(f"    Dark-corrected range: {np.min(corrected_data)} - {np.max(corrected_data)}")
    
    return corrected_data


def analyze_image_statistics(image: np.ndarray, name: str) -> dict:
    """Analyze image statistics"""
    stats = {
        'name': name,
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'median': float(np.median(image)),
        'percentile_25': float(np.percentile(image, 25)),
        'percentile_75': float(np.percentile(image, 75)),
    }
    
    # For color images, also analyze per channel
    if len(image.shape) == 3:
        for i, channel in enumerate(['B', 'G', 'R']):
            channel_data = image[:, :, i]
            stats[f'{channel}_mean'] = float(np.mean(channel_data))
            stats[f'{channel}_std'] = float(np.std(channel_data))
    
    return stats


def print_statistics_comparison(stats_before: dict, stats_after: dict):
    """Print comparison of statistics"""
    print(f"\n{'='*60}")
    print(f"IMAGE STATISTICS COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n{stats_before['name']:>20} | {stats_after['name']:>20} | {'Difference':>12}")
    print(f"{'-'*60}")
    
    # Basic statistics
    metrics = ['mean', 'std', 'median', 'min', 'max']
    for metric in metrics:
        val_before = stats_before[metric]
        val_after = stats_after[metric]
        diff = val_after - val_before
        print(f"{metric:>20} | {val_before:>20.2f} | {val_after:>20.2f} | {diff:>+12.2f}")
    
    # Per-channel statistics for color images
    if 'R_mean' in stats_before:
        print(f"\nPer-channel comparison:")
        for channel in ['R', 'G', 'B']:
            val_before = stats_before[f'{channel}_mean']
            val_after = stats_after[f'{channel}_mean']
            diff = val_after - val_before
            channel_name = f"{channel}_mean"
            print(f"{channel_name:>20} | {val_before:>20.2f} | {val_after:>20.2f} | {diff:>+12.2f}")


def main():
    parser = argparse.ArgumentParser(description="Compare images before and after dark correction")
    parser.add_argument("--original", required=True, help="Path to original RAW image")
    parser.add_argument("--dark", required=True, help="Path to dark reference image")
    parser.add_argument("--output_dir", default=None, help="Output directory for comparison images")
    
    args = parser.parse_args()
    
    # 使用与lens_shading.py相同的配置参数
    IMAGE_WIDTH = 3840      # 图像宽度
    IMAGE_HEIGHT = 2160     # 图像高度
    DATA_TYPE = 'uint16'    # 数据类型
    
    # Read original image
    print(f"Reading original image: {args.original}")
    original_data = read_raw_image(args.original, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
    if original_data is None:
        print("Failed to read original image")
        return
    
    # Read dark reference
    print(f"Reading dark reference: {args.dark}")
    dark_data = read_raw_image(args.dark, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
    if dark_data is None:
        print("Failed to read dark reference")
        return
    
    # Apply dark correction
    corrected_data = apply_dark_correction(original_data, dark_data)
    
    # Analyze statistics
    stats_original = analyze_image_statistics(original_data, "Original")
    stats_corrected = analyze_image_statistics(corrected_data, "Dark Corrected")
    
    # Print comparison
    print_statistics_comparison(stats_original, stats_corrected)
    
    # Save comparison images if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Normalize to 8-bit for visualization
        def normalize_to_8bit(data):
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            return (data_norm * 255).astype(np.uint8)
        
        # Save original
        original_8bit = normalize_to_8bit(original_data)
        cv2.imwrite(str(output_dir / "original.png"), original_8bit)
        
        # Save corrected
        corrected_8bit = normalize_to_8bit(corrected_data)
        cv2.imwrite(str(output_dir / "dark_corrected.png"), corrected_8bit)
        
        # Save difference
        diff_data = corrected_data.astype(np.float64) - original_data.astype(np.float64)
        diff_8bit = normalize_to_8bit(diff_data)
        cv2.imwrite(str(output_dir / "difference.png"), diff_8bit)
        
        print(f"\nComparison images saved to: {output_dir}")
    
    # Check for potential issues
    print(f"\n{'='*60}")
    print(f"ANALYSIS")
    print(f"{'='*60}")
    
    mean_diff = stats_corrected['mean'] - stats_original['mean']
    if mean_diff > 0:
        print(f"⚠️  WARNING: Image became BRIGHTER after dark correction!")
        print(f"   Mean increased by {mean_diff:.2f}")
        print(f"   This is unusual - dark correction should make image darker")
        
        # Check if dark reference is valid
        dark_mean = np.mean(dark_data)
        original_mean = np.mean(original_data)
        print(f"\n   Dark reference mean: {dark_mean:.2f}")
        print(f"   Original image mean: {original_mean:.2f}")
        
        if dark_mean > original_mean:
            print(f"   ❌ PROBLEM: Dark reference is brighter than original image!")
            print(f"   This suggests the dark reference is incorrect")
        elif dark_mean < 0:
            print(f"   ❌ PROBLEM: Dark reference has negative values!")
        else:
            print(f"   ✅ Dark reference appears valid")
    else:
        print(f"✅ Image became darker as expected")
        print(f"   Mean decreased by {abs(mean_diff):.2f}")


if __name__ == "__main__":
    main()
