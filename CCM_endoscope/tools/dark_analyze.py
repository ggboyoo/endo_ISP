#!/usr/bin/env python3
"""
Dark Image Analysis Script
Reads dark RAW images and performs FFT analysis to visualize noise characteristics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
import json
from datetime import datetime

# Import RAW reader
try:
    from raw_reader import read_raw_image
except ImportError:
    print("Error: raw_reader.py not found in the same directory!")
    print("Please ensure raw_reader.py is in the same directory as this script.")
    exit(1)


def load_dark_image(dark_path: str, width: int, height: int, data_type: str = 'uint16') -> Optional[np.ndarray]:
    """Load dark reference image"""
    try:
        if not os.path.exists(dark_path):
            print(f"Dark reference file not found: {dark_path}")
            return None
        
        dark_data = read_raw_image(dark_path, width, height, data_type)
        if dark_data is not None:
            print(f"Dark reference loaded: {dark_data.shape}, range: {np.min(dark_data)}-{np.max(dark_data)}")
        return dark_data
    except Exception as e:
        print(f"Error loading dark reference: {e}")
        return None


def compute_fft_2d(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D FFT of the image
    
    Args:
        image: Input image (2D array)
        
    Returns:
        fft_magnitude: Magnitude spectrum
        fft_phase: Phase spectrum  
        fft_shifted: Shifted FFT (DC component at center)
    """
    # Convert to float for FFT
    img_float = image.astype(np.float64)
    
    # Compute 2D FFT
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute magnitude and phase
    fft_magnitude = np.abs(fft_shifted)
    fft_phase = np.angle(fft_shifted)
    
    return fft_magnitude, fft_phase, fft_shifted


def compute_radial_profile(fft_magnitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial profile of FFT magnitude
    
    Args:
        fft_magnitude: 2D FFT magnitude spectrum
        
    Returns:
        radii: Radial distances from center
        profile: Average magnitude at each radius
    """
    h, w = fft_magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Bin the radii
    r_max = min(center_x, center_y)
    r_bins = np.arange(0, r_max + 1)
    
    # Compute radial profile
    profile = []
    radii = []
    
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.any(mask):
            profile.append(np.mean(fft_magnitude[mask]))
            radii.append((r_bins[i] + r_bins[i + 1]) / 2)
    
    return np.array(radii), np.array(profile)


def analyze_dark_image(dark_data: np.ndarray, output_dir: Path) -> None:
    """
    Perform comprehensive FFT analysis on dark image
    
    Args:
        dark_data: Dark reference image
        output_dir: Output directory for plots
    """
    print("Performing FFT analysis on dark image...")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # 1. Basic statistics
    print(f"Dark image statistics:")
    print(f"  Shape: {dark_data.shape}")
    print(f"  Data type: {dark_data.dtype}")
    print(f"  Min: {np.min(dark_data)}")
    print(f"  Max: {np.max(dark_data)}")
    print(f"  Mean: {np.mean(dark_data):.2f}")
    print(f"  Std: {np.std(dark_data):.2f}")
    
    # 2. Compute FFT
    fft_magnitude, fft_phase, fft_shifted = compute_fft_2d(dark_data)
    
    # 3. Compute radial profile
    radii, radial_profile = compute_radial_profile(fft_magnitude)
    
    # 4. Create plots - dark image and FFT spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original dark image (left)
    im1 = axes[0].imshow(dark_data, cmap='gray')
    axes[0].set_title('Dark Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Pixel Value', rotation=270, labelpad=15)
    
    # FFT magnitude spectrum (right) - using viridis colormap like in the reference
    log_magnitude = np.log10(fft_magnitude + 1e-10)  # Add small value to avoid log(0)
    im2 = axes[1].imshow(log_magnitude, cmap='viridis', aspect='equal')
    axes[1].set_title('FFT Magnitude Spectrum', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Log Magnitude', rotation=270, labelpad=15)
    
    # Set equal aspect ratio for both plots to make them square
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"dark_analysis_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved: {plot_path}")
    
    # Save analysis data
    analysis_data = {
        'timestamp': timestamp,
        'image_shape': dark_data.shape,
        'image_dtype': str(dark_data.dtype),
        'statistics': {
            'min': float(np.min(dark_data)),
            'max': float(np.max(dark_data)),
            'mean': float(np.mean(dark_data)),
            'std': float(np.std(dark_data)),
            'var': float(np.var(dark_data))
        },
        'fft_statistics': {
            'max_magnitude': float(np.max(fft_magnitude)),
            'mean_magnitude': float(np.mean(fft_magnitude)),
            'dc_component': float(fft_magnitude[dark_data.shape[0]//2, dark_data.shape[1]//2])
        },
        'radial_profile': {
            'radii': radii.tolist(),
            'profile': radial_profile.tolist()
        }
    }
    
    json_path = output_dir / f"dark_analysis_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"Analysis data saved: {json_path}")
    
    plt.show()


# ============================================================================
# 配置参数 - 直接在这里修改，无需命令行输入
# ============================================================================

# 输入文件配置
DARK_RAW_PATH = r"F:\ZJU\Picture\dark\g9\25-09-01 155313.raw"  # 暗电流图像路径

# 图像参数配置
IMAGE_WIDTH = 3840      # 图像宽度
IMAGE_HEIGHT = 2160     # 图像高度
DATA_TYPE = 'uint16'    # 数据类型

# 输出配置
OUTPUT_DIRECTORY = "dark_analysis_output"  # 输出目录


def main():
    """Main function"""
    print("=" * 60)
    print("Dark Image FFT Analysis")
    print("=" * 60)
    print(f"Input file: {DARK_RAW_PATH}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Data type: {DATA_TYPE}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print("=" * 60)
    
    # Load dark image
    dark_data = load_dark_image(DARK_RAW_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
    if dark_data is None:
        print("Failed to load dark image")
        return
    
    # Perform analysis
    output_dir = Path(OUTPUT_DIRECTORY)
    analyze_dark_image(dark_data, output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
