#!/usr/bin/env python3
"""
Lens Shading Correction Script
Analyzes RAW images to calculate lens shading correction matrix
Uses grid-based method to analyze RGGB channels separately
"""

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.patches as patches
from scipy import ndimage
from typing import Optional, Tuple, List, Dict, Union
import json
from datetime import datetime
from PIL import Image

# Import functions from raw_reader.py
try:
    from raw_reader import read_raw_image
except ImportError:
    print("Error: raw_reader.py not found in the same directory!")
    print("Please ensure raw_reader.py is in the same directory as this script.")
    exit(1)

# ============================================================================
# 配置文件路径 - 直接在这里修改，无需交互输入
# ============================================================================

# 输入路径配置
INPUT_PATH = r"F:\ZJU\Picture\lens shading\700.raw"  # 待分析的RAW图像文件夹路径
# 或者单个文件路径，例如: r"F:\ZJU\Picture\lens_shading\test_image.raw"

# 暗点平矫正配置
DARK_RAW_PATH = r"F:\ZJU\Picture\dark\g8\average_dark.raw"  # 暗电流图像路径
ENABLE_DARK_CORRECTION = True  # 是否启用暗点平矫正

# 图像参数配置
IMAGE_WIDTH = 3840      # 图像宽度
IMAGE_HEIGHT = 2160     # 图像高度
DATA_TYPE = 'uint16'    # 数据类型

# Lens Shading分析配置
GRID_SIZE =   32     # 网格大小（建议32x32或64x64）
CHANNEL_NAMES = ['R', 'G1', 'G2', 'B']  # RGGB四个通道名称
MIN_VALID_VALUE = 50  # 最小有效像素值（避免过暗区域）
MAX_VALID_VALUE = 4095  # 最大有效像素值（避免过亮区域）
ENABLE_MEDIAN_FILTER = True  # 是否对网格均值进行中值滤波
MEDIAN_FILTER_SIZE = 3  # 中值滤波核大小（3x3或5x5）
BLACK_PIXEL_THRESHOLD = 50  # 32x32格内若有像素低于该值，则该格不矫正（系数=1）

# 输出配置 True/False
OUTPUT_DIRECTORY = r"F:\ZJU\Picture\lens shading\new"
GENERATE_PLOTS = True   # 是否生成分析图表
SAVE_PLOTS = True       # 是否保存图表文件
SAVE_CORRECTION_MATRIX = True  # 是否保存矫正矩阵
SAVE_CORRECTED_IMAGES = True   # 是否保存矫正后的图像
SHOW_INTERACTIVE_PLOTS = True  # 是否显示交互式图像对比
SHOW_HISTOGRAMS = False        # 是否显示/生成直方图（关闭则不再显示/保存）

# ============================================================================
# 脚本功能代码（无需修改）
# ============================================================================

def load_dark_reference(dark_path: str, width: int, height: int, data_type: str) -> Optional[np.ndarray]:
    """
    Load dark current reference image
    
    Args:
        dark_path: Path to dark RAW file
        width: Image width
        height: Image height
        data_type: Data type
        
    Returns:
        Dark reference image or None if failed
    """
    print(f"Loading dark reference image: {dark_path}")
    
    try:
        dark_data = read_raw_image(dark_path, width, height, data_type)
        print(f"  Dark image loaded: {dark_data.shape}, dtype: {dark_data.dtype}")
        print(f"  Dark image range: {np.min(dark_data)} - {np.max(dark_data)}")
        return dark_data
    except Exception as e:
        print(f"  Error loading dark image: {e}")
        return None


def apply_dark_correction(raw_data: np.ndarray, dark_data: np.ndarray) -> np.ndarray:
    """
    Apply dark current correction to RAW data
    
    Args:
        raw_data: Original RAW image data
        dark_data: Dark current reference data
        
    Returns:
        Dark-corrected RAW data
    """
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


def apply_lens_shading_correction(channel_data: np.ndarray, correction_matrix: np.ndarray) -> np.ndarray:
    """
    Apply lens shading correction to a channel
    
    Args:
        channel_data: Single channel data
        correction_matrix: Correction matrix for the channel
        
    Returns:
        Corrected channel data
    """
    print(f"    Applying lens shading correction...")
    print(f"      Original range: {np.min(channel_data)} - {np.max(channel_data)}")
    
    # Apply correction
    corrected_data = channel_data.astype(np.float64) * correction_matrix
    
    print(f"      Corrected range: {np.min(corrected_data)} - {np.max(corrected_data)}")
    
    return corrected_data


def normalize_to_8bit(data: np.ndarray) -> np.ndarray:
    """Normalize data to 8-bit for display"""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:
        normalized = (data - data_min) / (data_max - data_min)
        return (normalized * 255).astype(np.uint8)
    else:
        return np.zeros_like(data, dtype=np.uint8)


def show_interactive_comparison(original_data: np.ndarray, 
                              dark_corrected_data: np.ndarray, 
                              lens_shading_corrected_data: np.ndarray,
                              title: str = "Image Comparison"):
    """Show interactive comparison of three images with pixel value display"""
    
    # Find global min/max across all images for consistent normalization
    global_min = min(np.min(original_data), np.min(dark_corrected_data), np.min(lens_shading_corrected_data))
    global_max = max(np.max(original_data), np.max(dark_corrected_data), np.max(lens_shading_corrected_data))
    
    print(f"  Global value range: {global_min:.1f} - {global_max:.1f}")
    print(f"  Original mean: {np.mean(original_data):.1f}")
    print(f"  Dark corrected mean: {np.mean(dark_corrected_data):.1f}")
    print(f"  Lens shading mean: {np.mean(lens_shading_corrected_data):.1f}")
    
    # Normalize all images using the same range
    def normalize_with_global_range(data):
        if global_max > global_min:
            normalized = (data - global_min) / (global_max - global_min)
            return (normalized * 255).astype(np.uint8)
        else:
            return np.zeros_like(data, dtype=np.uint8)
    
    original_8bit = normalize_with_global_range(original_data)
    dark_corrected_8bit = normalize_with_global_range(dark_corrected_data)
    lens_shading_8bit = normalize_with_global_range(lens_shading_corrected_data)
    
    # Create figure with 3 subplots and more space
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # Plot images with consistent vmin/vmax
    im1 = axes[0].imshow(original_8bit, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original RAW', fontsize=14, pad=20)
    axes[0].axis('off')
    
    im2 = axes[1].imshow(dark_corrected_8bit, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Dark Corrected', fontsize=14, pad=20)
    axes[1].axis('off')
    
    im3 = axes[2].imshow(lens_shading_8bit, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Lens Shading Corrected', fontsize=14, pad=20)
    axes[2].axis('off')
    
    # Add single colorbar for all images (showing global range) with better positioning
    cbar = fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.08, shrink=0.8)
    cbar.set_label(f'Pixel Value ({global_min:.0f} - {global_max:.0f})', rotation=270, labelpad=25, fontsize=12)
    
    # Add text box for pixel value display
    text_box = fig.text(0.02, 0.02, '', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def on_mouse_move(event):
        if event.inaxes is None:
            return
        
        # Get mouse position
        x, y = int(event.xdata), int(event.ydata) if event.xdata and event.ydata else (0, 0)
        
        # Check bounds
        if (0 <= x < original_data.shape[1] and 0 <= y < original_data.shape[0]):
            # Get pixel values from original data (not normalized)
            orig_val = original_data[y, x]
            dark_val = dark_corrected_data[y, x]
            lens_val = lens_shading_corrected_data[y, x]
            
            # Calculate differences
            dark_diff = dark_val - orig_val
            lens_diff = lens_val - orig_val
            
            # Update text box with real values and differences
            text = f'Position: ({x}, {y})\n'
            text += f'Original: {orig_val:.1f}\n'
            text += f'Dark Corrected: {dark_val:.1f} ({dark_diff:+.1f})\n'
            text += f'Lens Shading: {lens_val:.1f} ({lens_diff:+.1f})'
            text_box.set_text(text)
        
        fig.canvas.draw_idle()
    
    # Connect mouse move event
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    # Add instructions
    fig.text(0.5, 0.95, 'Move mouse over images to see pixel values', 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.show()


def reconstruct_corrected_image(channels: Dict[str, np.ndarray], 
                              corrections: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Reconstruct corrected image from corrected channels
    
    Args:
        channels: Dictionary of corrected channel data
        corrections: Dictionary of correction matrices
        
    Returns:
        Reconstructed corrected image
    """
    height, width = channels['R'].shape
    
    # Create full-size image
    corrected_image = np.zeros((height * 2, width * 2), dtype=np.float64)
    
    # Reconstruct RGGB pattern
    # R  G  R  G  R  G ...
    # G  B  G  B  G  B ...
    # R  G  R  G  R  G ...
    # G  B  G  B  G  B ...
    
    # Place corrected channels back to original positions
    corrected_image[0::2, 0::2] = channels['R']      # Even rows, even columns
    corrected_image[0::2, 1::2] = channels['G1']     # Even rows, odd columns
    corrected_image[1::2, 0::2] = channels['G2']     # Odd rows, even columns
    corrected_image[1::2, 1::2] = channels['B']      # Odd rows, odd columns
    
    print(f"    Reconstructed corrected image: {corrected_image.shape}")
    print(f"    Corrected image range: {np.min(corrected_image)} - {np.max(corrected_image)}")
    
    return corrected_image


def separate_rggb_channels(raw_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Separate RGGB channels from RAW image
    
    Args:
        raw_data: RAW image data (H, W)
        
    Returns:
        Dictionary containing separated channels
    """
    height, width = raw_data.shape
    
    # RGGB pattern layout:
    # R  G  R  G  R  G ...
    # G  B  G  B  G  B ...
    # R  G  R  G  R  G ...
    # G  B  G  B  G  B ...
    
    # Extract channels
    R_channel = raw_data[0::2, 0::2]      # Even rows, even columns
    G1_channel = raw_data[0::2, 1::2]     # Even rows, odd columns  
    G2_channel = raw_data[1::2, 0::2]     # Odd rows, even columns
    B_channel = raw_data[1::2, 1::2]      # Odd rows, odd columns
    
    channels = {
        'R': R_channel,
        'G1': G1_channel,
        'G2': G2_channel,
        'B': B_channel
    }
    
    print(f"  Channel separation complete:")
    print(f"    R channel: {R_channel.shape}")
    print(f"    G1 channel: {G1_channel.shape}")
    print(f"    G2 channel: {G2_channel.shape}")
    print(f"    B channel: {B_channel.shape}")
    
    return channels


def plot_channel_center_profiles(channels: Dict[str, np.ndarray], title: str = "Channel Center Profiles") -> None:
    """
    Plot center horizontal and vertical profiles for R, G1, G2, B channels.

    For each channel array of shape (Hc, Wc):
    - horizontal profile: values at row Hc//2 across all columns
    - vertical profile: values at column Wc//2 across all rows
    """
    try:
        import matplotlib.pyplot as plt
        color_map = {
            'R': 'red',
            'G1': 'green',
            'G2': 'lime',
            'B': 'blue',
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Define band half-width using grid size to average, fallback to 1
        try:
            band = max(1, int(GRID_SIZE // 2))
        except Exception:
            band = 1

        # Horizontal profiles (mean of a band around center row)
        ax_h = axes[0]
        for name in ['R', 'G1', 'G2', 'B']:
            if name not in channels:
                continue
            ch = channels[name]
            hh, ww = ch.shape
            row_c = hh // 2
            r0 = max(0, row_c - band)
            r1 = min(hh, row_c + band + 1)
            prof = np.mean(ch[r0:r1, :].astype(np.float64), axis=0)
            ax_h.plot(np.arange(ww), prof, label=name, color=color_map.get(name, None), linewidth=1.2)
        ax_h.set_title(f'Mean Center Row Band (±{band})')
        ax_h.set_xlabel('Column (channel domain)')
        ax_h.set_ylabel('Pixel Value')
        ax_h.grid(True, alpha=0.3)
        ax_h.legend()

        # Vertical profiles (mean of a band around center column)
        ax_v = axes[1]
        for name in ['R', 'G1', 'G2', 'B']:
            if name not in channels:
                continue
            ch = channels[name]
            hh, ww = ch.shape
            col_c = ww // 2
            c0 = max(0, col_c - band)
            c1 = min(ww, col_c + band + 1)
            prof = np.mean(ch[:, c0:c1].astype(np.float64), axis=1)
            ax_v.plot(np.arange(hh), prof, label=name, color=color_map.get(name, None), linewidth=1.2)
        ax_v.set_title(f'Mean Center Column Band (±{band})')
        ax_v.set_xlabel('Row (channel domain)')
        ax_v.set_ylabel('Pixel Value')
        ax_v.grid(True, alpha=0.3)
        ax_v.legend()

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"  Error plotting channel center profiles: {e}")


def create_grid_analysis(channel_data: np.ndarray, grid_size: int, 
                        min_valid: int, max_valid: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create grid-based analysis for lens shading
    
    Args:
        channel_data: Single channel data
        grid_size: Size of grid cells
        min_valid: Minimum valid pixel value
        max_valid: Maximum valid pixel value
        
    Returns:
        Tuple of (grid_means, grid_counts)
    """
    height, width = channel_data.shape
    
    # Calculate grid dimensions
    grid_h = height // grid_size
    grid_w = width // grid_size
    
    # Initialize grid arrays
    grid_means = np.zeros((grid_h, grid_w), dtype=np.float64)
    grid_counts = np.zeros((grid_h, grid_w), dtype=np.int32)
    
    print(f"    Grid analysis: {grid_h}x{grid_w} grids of size {grid_size}x{grid_size}")
    
    # Analyze each grid cell
    for i in range(grid_h):
        for j in range(grid_w):
            # Extract grid cell
            start_h = i * grid_size
            end_h = start_h + grid_size
            start_w = j * grid_size
            end_w = start_w + grid_size
            
            grid_cell = channel_data[start_h:end_h, start_w:end_w]
            
            # Filter valid pixels: ignore black outliers within the block
            # Use a stricter lower bound: max(min_valid, BLACK_PIXEL_THRESHOLD)
            lower_bound = max(min_valid, BLACK_PIXEL_THRESHOLD)
            valid_mask = (grid_cell >= lower_bound) & (grid_cell <= max_valid)
            valid_pixels = grid_cell[valid_mask]
            
            if len(valid_pixels) > 0:
                grid_means[i, j] = np.mean(valid_pixels)
                grid_counts[i, j] = len(valid_pixels)
            else:
                grid_means[i, j] = 0
                grid_counts[i, j] = 0
    
    # Apply median filter to remove outliers if enabled
    if ENABLE_MEDIAN_FILTER:
        print(f"    Applying median filter (size={MEDIAN_FILTER_SIZE}x{MEDIAN_FILTER_SIZE}) to grid means...")
        
        # Create mask for valid grids (non-zero means)
        valid_mask = grid_means > 0
        
        if np.any(valid_mask):
            # Apply median filter only to valid regions
            filtered_means = ndimage.median_filter(grid_means, size=MEDIAN_FILTER_SIZE)
            
            # Only update valid regions, keep invalid regions as 0
            grid_means = np.where(valid_mask, filtered_means, grid_means)
            
            # Calculate statistics before and after filtering
            original_mean = np.mean(grid_means[valid_mask])
            filtered_mean = np.mean(filtered_means[valid_mask])
            print(f"    Grid means - Original: {original_mean:.2f}, Filtered: {filtered_mean:.2f}")
        else:
            print(f"    No valid grids found, skipping median filter")
    
    return grid_means, grid_counts


def calculate_lens_shading_correction(grid_means: np.ndarray, 
                                    reference_value: Optional[float] = None) -> np.ndarray:
    """
    Calculate lens shading correction matrix
    
    Args:
        grid_means: Grid means array
        reference_value: Reference value for normalization (None = use center)
        
    Returns:
        Correction matrix
    """
    # Find reference value (center of image or specified)
    if reference_value is None:
        center_h, center_w = grid_means.shape[0] // 2, grid_means.shape[1] // 2
        reference_value = grid_means[center_h, center_w]
        print(f"    Using center reference value: {reference_value:.2f}")
    else:
        print(f"    Using specified reference value: {reference_value:.2f}")
    
    # Calculate correction factors
    correction_matrix = np.zeros_like(grid_means, dtype=np.float64)
    
    for i in range(grid_means.shape[0]):
        for j in range(grid_means.shape[1]):
            if grid_means[i, j] > 0:
                correction_matrix[i, j] = reference_value / grid_means[i, j]
            else:
                correction_matrix[i, j] = 1.0  # No correction for invalid areas
    
    # Clip correction factors to reasonable range
    correction_matrix = np.clip(correction_matrix, 0.5, 2.0)
    
    print(f"    Correction matrix range: {np.min(correction_matrix):.3f} - {np.max(correction_matrix):.3f}")
    
    return correction_matrix


def interpolate_correction_matrix(correction_matrix: np.ndarray, 
                                target_height: int, target_width: int) -> np.ndarray:
    """
    Interpolate correction matrix to full image size
    
    Lens Shading矫正矩阵插值算法原理：
    
    1. 网格矫正矩阵计算：
       - 将图像分成多个网格（如16x16像素的网格）
       - 计算每个网格的平均亮度
       - 以中心网格为参考，计算矫正系数：correction = center_mean / grid_mean
       - 中心区域矫正系数 = 1.0，边缘区域矫正系数 > 1.0（因为边缘更暗）
    
    2. 插值扩展原理：
       - 网格矫正矩阵只有少数几个点（如135x240个网格点）
       - 需要插值到全图像尺寸（如2160x3840个像素点）
       - 使用三次样条插值（RectBivariateSpline）进行平滑插值
       - 插值后每个像素都有对应的矫正系数
    
    3. 为什么会出现小于1的值：
       - 虽然中心是最亮的，但插值过程中会产生中间值
       - 三次样条插值会创建平滑的过渡，可能产生略小于1的值
       - 这是正常的，因为插值函数需要平滑连接网格点
       - 实际应用中，这些值通常接近1.0，不会显著影响图像质量
    
    4. 矫正公式：
       corrected_pixel = original_pixel * correction_coefficient
       - correction_coefficient > 1.0：使像素变亮（边缘区域）
       - correction_coefficient = 1.0：不改变（中心区域）
       - correction_coefficient < 1.0：使像素变暗（插值过渡区域）
    
    Args:
        correction_matrix: Grid-based correction matrix (网格矫正矩阵)
        target_height: Target image height (目标图像高度)
        target_width: Target image width (目标图像宽度)
        
    Returns:
        Full-size correction matrix (全尺寸矫正矩阵)
    """
    grid_h, grid_w = correction_matrix.shape
    
    print(f"    Grid matrix shape: {grid_h}x{grid_w}")
    print(f"    Target size: {target_height}x{target_width}")
    print(f"    Grid correction range: {np.min(correction_matrix):.3f} - {np.max(correction_matrix):.3f}")
    
    # Create coordinate arrays for interpolation
    # 为插值创建坐标数组
    grid_y = np.linspace(0, target_height - 1, grid_h)
    grid_x = np.linspace(0, target_width - 1, grid_w)
    
    # Create target coordinate arrays
    # 创建目标坐标数组
    target_y = np.arange(target_height)
    target_x = np.arange(target_width)
    
    # Interpolate using scipy
    from scipy.interpolate import RectBivariateSpline
    
    # Create interpolation function with cubic spline
    # 使用三次样条插值创建插值函数
    interp_func = RectBivariateSpline(grid_y, grid_x, correction_matrix, kx=3, ky=3)
    
    # Interpolate to full size
    # 插值到全尺寸
    full_correction = interp_func(target_y, target_x)
    
    print(f"    Full matrix shape: {full_correction.shape}")
    print(f"    Full matrix range: {np.min(full_correction):.3f} - {np.max(full_correction):.3f}")
    print(f"    Values < 1.0: {np.sum(full_correction < 1.0)} pixels ({np.sum(full_correction < 1.0)/full_correction.size*100:.1f}%)")
    
    return full_correction


def analyze_single_image(raw_file: str, width: int, height: int, 
                        data_type: str, grid_size: int, dark_data: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze a single RAW image for lens shading
    
    Args:
        raw_file: Path to RAW file
        width: Image width
        height: Image height
        data_type: Data type
        grid_size: Grid size for analysis
        dark_data: Dark current reference data
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"\nAnalyzing: {Path(raw_file).name}")
    
    try:
        # Read RAW data
        raw_data = read_raw_image(raw_file, width, height, data_type)
        print(f"  Image loaded: {raw_data.shape}, dtype: {raw_data.dtype}")
        print(f"  Data range: {np.min(raw_data)} - {np.max(raw_data)}")
        
        # Apply dark current correction if enabled
        corrected_raw_data = raw_data.copy()
        if ENABLE_DARK_CORRECTION and dark_data is not None:
            # Calculate and compare means before dark correction
            original_mean = np.mean(raw_data)
            print(f"  Original image mean: {original_mean:.2f}")
            
            corrected_raw_data = apply_dark_correction(raw_data, dark_data)
            
            # Calculate and compare means after dark correction
            corrected_mean = np.mean(corrected_raw_data)
            print(f"  Dark-corrected image mean: {corrected_mean:.2f}")
            
            # Compare the difference
            mean_diff = corrected_mean - original_mean
            print(f"  Mean difference: {mean_diff:+.2f}")
            
            if mean_diff > 0:
                print(f"  ⚠️  WARNING: Image became BRIGHTER after dark correction!")
                print(f"     This is unusual - dark correction should make image darker")
            else:
                print(f"  ✅ Image became darker as expected")
        else:
            print(f"  Dark correction skipped")
        
        # Separate RGGB channels
        channels = separate_rggb_channels(corrected_raw_data)
        
        # Analyze each channel
        channel_results = {}
        corrected_channels = {}
        
        for channel_name in CHANNEL_NAMES:
            print(f"  Analyzing {channel_name} channel...")
            
            channel_data = channels[channel_name]
            
            # Create grid analysis
            grid_means, grid_counts = create_grid_analysis(
                channel_data, grid_size, MIN_VALID_VALUE, MAX_VALID_VALUE
            )
            
            # Calculate lens shading correction
            correction_matrix = calculate_lens_shading_correction(grid_means)
            
            # Interpolate to full channel size
            full_correction = interpolate_correction_matrix(
                correction_matrix, channel_data.shape[0], channel_data.shape[1]
            )
            
            # Apply lens shading correction
            corrected_channel_data = apply_lens_shading_correction(channel_data, full_correction)
            corrected_channels[channel_name] = corrected_channel_data
            
            channel_results[channel_name] = {
                'grid_means': grid_means,
                'grid_counts': grid_counts,
                'correction_matrix': correction_matrix,
                'full_correction': full_correction,
                'channel_data': channel_data,
                'corrected_channel_data': corrected_channel_data
            }
        
        # Reconstruct corrected image
        corrected_image = reconstruct_corrected_image(corrected_channels, 
                                                   {name: result['full_correction'] 
                                                    for name, result in channel_results.items()})

        # Plot center profiles for channels (before correction) for diagnostics
        try:
            plot_channel_center_profiles(channels, title=f"Center Profiles - {Path(raw_file).name}")
        except Exception as e:
            print(f"  Warning: failed to plot channel profiles: {e}")
        
        # Show interactive comparison if enabled
        if SHOW_INTERACTIVE_PLOTS:
            print(f"  Showing interactive image comparison...")
            show_interactive_comparison(
                raw_data, 
                corrected_raw_data, 
                corrected_image,
                title=f"Lens Shading Analysis - {Path(raw_file).name}"
            )
        
        original_rggb_histogram_stats = None
        
        corrected_rggb_histogram_stats = None
        
        result = {
            'filename': raw_file,
            'raw_data': raw_data,
            'corrected_raw_data': corrected_raw_data,
            'channels': channel_results,
            'corrected_image': corrected_image,
            'original_rggb_histogram_stats': original_rggb_histogram_stats,
            'corrected_rggb_histogram_stats': corrected_rggb_histogram_stats,
            'analysis_success': True
        }
        
        return result
        
    except Exception as e:
        print(f"  Error analyzing file: {e}")
        return {
            'filename': raw_file,
            'analysis_success': False,
            'error': str(e)
        }


def create_lens_shading_plots(results: List[Dict], output_dir: Path):
    """
    Create visualization plots for lens shading analysis
    
    Args:
        results: List of analysis results
        output_dir: Output directory path
    """
    if not results or not GENERATE_PLOTS:
        return
    
    print(f"\nGenerating lens shading analysis plots...")
    
    for i, result in enumerate(results[:2]):  # Show first 2 images
        if not result['analysis_success']:
            continue
        
        # Create comprehensive subplot with 5 columns to include corrected image
        fig, axes = plt.subplots(4, 5, figsize=(25, 16))
        fig.suptitle(f'Lens Shading Analysis: {Path(result["filename"]).name}', fontsize=16)
        
        for j, channel_name in enumerate(CHANNEL_NAMES):
            channel_data = result['channels'][channel_name]
            
            # Column 1: Original channel data
            axes[j, 0].imshow(channel_data['channel_data'], cmap='gray')
            axes[j, 0].set_title(f'{channel_name} - Original')
            axes[j, 0].axis('off')
            
            # Column 2: Grid means
            im1 = axes[j, 1].imshow(channel_data['grid_means'], cmap='viridis')
            axes[j, 1].set_title(f'{channel_name} - Grid Means')
            axes[j, 1].axis('off')
            plt.colorbar(im1, ax=axes[j, 1], fraction=0.046, pad=0.04)
            
            # Column 3: Correction matrix
            im2 = axes[j, 2].imshow(channel_data['correction_matrix'], cmap='plasma')
            axes[j, 2].set_title(f'{channel_name} - Correction Matrix')
            axes[j, 2].axis('off')
            plt.colorbar(im2, ax=axes[j, 2], fraction=0.046, pad=0.04)
            
            # Column 4: Full correction
            im3 = axes[j, 3].imshow(channel_data['full_correction'], cmap='plasma')
            axes[j, 3].set_title(f'{channel_name} - Full Correction')
            axes[j, 3].axis('off')
            plt.colorbar(im3, ax=axes[j, 3], fraction=0.046, pad=0.04)
            
            # Column 5: Corrected channel data
            if 'corrected_channel_data' in channel_data:
                im4 = axes[j, 4].imshow(channel_data['corrected_channel_data'], cmap='gray')
                axes[j, 4].set_title(f'{channel_name} - Corrected')
                axes[j, 4].axis('off')
                plt.colorbar(im4, ax=axes[j, 4], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save plot
        if SAVE_PLOTS:
            plot_filename = f"lens_shading_analysis_{Path(result['filename']).stem}_{i+1}.png"
            plot_path = output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Plot saved: {plot_path}")
        
        plt.show()
        
        # Create additional comparison plot for full image
        create_image_comparison_plot(result, output_dir, i)
        
        # Skip histogram plots per user request


def analyze_rggb_channels_histogram(channels: Dict[str, np.ndarray]) -> Dict:
    """
    Analyze RGGB channels histogram statistics
    
    Args:
        channels: Dictionary containing RGGB channel data
        
    Returns:
        Dictionary containing channel histogram statistics
    """
    try:
        channel_stats = {}
        
        for channel_name, channel_data in channels.items():
            # Convert to float for analysis
            channel_float = channel_data.astype(np.float32)
            
            # Calculate basic statistics
            mean_val = np.mean(channel_float)
            std_val = np.std(channel_float)
            var_val = np.var(channel_float)
            min_val = np.min(channel_float)
            max_val = np.max(channel_float)
            median_val = np.median(channel_float)
            
            # Calculate noise metrics
            snr = mean_val / std_val if std_val > 0 else 0
            cv = std_val / mean_val if mean_val > 0 else 0
            
            # Calculate histogram
            hist, bins = np.histogram(channel_float, bins=256, range=(min_val, max_val))
            
            channel_stats[channel_name] = {
                'data': channel_data,
                'mean': float(mean_val),
                'std': float(std_val),
                'variance': float(var_val),
                'min': float(min_val),
                'max': float(max_val),
                'median': float(median_val),
                'snr': float(snr),
                'cv': float(cv),
                'histogram': hist.tolist(),
                'bins': bins.tolist(),
                'channel_size': channel_data.shape
            }
        
        return channel_stats
        
    except Exception as e:
        print(f"  Error analyzing RGGB channels histogram: {e}")
        return {'error': str(e)}

def create_rggb_histogram_plots(original_stats: Dict, corrected_stats: Dict, output_dir: Path, base_filename: str):
    """
    Create RGGB channel histogram comparison plots (before vs after correction)
    
    Args:
        original_stats: Original channel statistics dictionary
        corrected_stats: Corrected channel statistics dictionary
        output_dir: Output directory
        base_filename: Base filename
    """
    print("Creating RGGB channel histogram comparison plots...")
    
    try:
        if 'error' in original_stats or 'error' in corrected_stats:
            print(f"  Error in channel stats")
            return
        
        # Create before/after comparison plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('RGGB Channel Histogram Comparison (Before vs After Lens Shading Correction)', fontsize=16)
        
        channel_names = ['R', 'G1', 'G2', 'B']
        colors = ['red', 'green', 'lime', 'blue']
        
        for i, (channel_name, color) in enumerate(zip(channel_names, colors)):
            if channel_name in original_stats and channel_name in corrected_stats:
                orig_stats = original_stats[channel_name]
                corr_stats = corrected_stats[channel_name]
                
                # Original histogram (left column)
                orig_hist = np.array(orig_stats['histogram'])
                orig_bins = np.array(orig_stats['bins'])
                orig_bin_centers = (orig_bins[:-1] + orig_bins[1:]) / 2
                
                axes[0, i].bar(orig_bin_centers, orig_hist, width=(orig_bins[1] - orig_bins[0]), 
                              alpha=0.7, color=color, edgecolor='black')
                axes[0, i].set_title(f'{channel_name} - Before Correction')
                axes[0, i].set_xlabel('Pixel Value')
                axes[0, i].set_ylabel('Frequency')
                axes[0, i].grid(True, alpha=0.3)
                
                # Set adaptive axis limits
                value_range = np.max(orig_bin_centers) - np.min(orig_bin_centers)
                axes[0, i].set_xlim(np.min(orig_bin_centers) - 0.02 * value_range, 
                                   np.max(orig_bin_centers) + 0.02 * value_range)
                
                # Add statistics text for original
                orig_stats_text = f"Mean: {orig_stats['mean']:.1f}\n"
                orig_stats_text += f"Std: {orig_stats['std']:.1f}\n"
                orig_stats_text += f"SNR: {orig_stats['snr']:.1f}"
                
                axes[0, i].text(0.02, 0.98, orig_stats_text, transform=axes[0, i].transAxes, 
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Corrected histogram (right column)
                corr_hist = np.array(corr_stats['histogram'])
                corr_bins = np.array(corr_stats['bins'])
                corr_bin_centers = (corr_bins[:-1] + corr_bins[1:]) / 2
                
                axes[1, i].bar(corr_bin_centers, corr_hist, width=(corr_bins[1] - corr_bins[0]), 
                              alpha=0.7, color=color, edgecolor='black')
                axes[1, i].set_title(f'{channel_name} - After Correction')
                axes[1, i].set_xlabel('Pixel Value')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].grid(True, alpha=0.3)
                
                # Set adaptive axis limits
                value_range = np.max(corr_bin_centers) - np.min(corr_bin_centers)
                axes[1, i].set_xlim(np.min(corr_bin_centers) - 0.02 * value_range, 
                                   np.max(corr_bin_centers) + 0.02 * value_range)
                
                # Add statistics text for corrected
                corr_stats_text = f"Mean: {corr_stats['mean']:.1f}\n"
                corr_stats_text += f"Std: {corr_stats['std']:.1f}\n"
                corr_stats_text += f"SNR: {corr_stats['snr']:.1f}"
                
                axes[1, i].text(0.02, 0.98, corr_stats_text, transform=axes[1, i].transAxes, 
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plot_path = output_dir / f"{base_filename}_rggb_before_after_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  RGGB before/after comparison saved: {plot_path}")
        
        if GENERATE_PLOTS:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"  Error creating RGGB histogram comparison plots: {e}")
        import traceback
        traceback.print_exc()

def save_rggb_histogram_data(channel_stats: Dict, output_dir: Path, base_filename: str):
    """
    Save RGGB channel histogram analysis data to JSON file
    
    Args:
        channel_stats: Channel statistics dictionary
        output_dir: Output directory
        base_filename: Base filename
    """
    print("Saving RGGB channel histogram data...")
    
    try:
        # Create summary statistics
        summary_stats = {}
        for channel_name, stats in channel_stats.items():
            if channel_name != 'error':
                summary_stats[channel_name] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'variance': stats['variance'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': stats['median'],
                    'snr': stats['snr'],
                    'cv': stats['cv'],
                    'channel_size': stats['channel_size']
                }
        
        # Create complete data structure
        rggb_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_type': 'rggb_channel_histogram_analysis',
            'image_dimensions': [IMAGE_WIDTH, IMAGE_HEIGHT],
            'data_type': DATA_TYPE,
            'processing_settings': {
                'dark_correction_enabled': ENABLE_DARK_CORRECTION,
                'grid_size': GRID_SIZE,
                'min_valid_value': MIN_VALID_VALUE,
                'max_valid_value': MAX_VALID_VALUE
            },
            'channel_statistics': summary_stats
        }
        
        # Save to JSON file
        data_path = output_dir / f"{base_filename}_rggb_histogram_analysis.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(rggb_data, f, indent=2, ensure_ascii=False)
        
        print(f"  RGGB histogram data saved: {data_path}")
        
    except Exception as e:
        print(f"  Error saving RGGB histogram data: {e}")
        import traceback
        traceback.print_exc()

def create_image_comparison_plot(result: Dict, output_dir: Path, plot_index: int):
    """
    Create comparison plot showing original vs corrected full images
    
    Args:
        result: Analysis result dictionary
        output_dir: Output directory path
        plot_index: Plot index for saving
    """
    if not result['analysis_success']:
        return
    
    # Create comparison subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Image Comparison: {Path(result["filename"]).name}', fontsize=16)
    
    # Row 1: Original and corrected RAW images
    axes[0, 0].imshow(result['raw_data'], cmap='gray')
    axes[0, 0].set_title('Original RAW')
    axes[0, 0].axis('off')
    
    if 'corrected_raw_data' in result:
        axes[0, 1].imshow(result['corrected_raw_data'], cmap='gray')
        axes[0, 1].set_title('Dark-Corrected RAW')
        axes[0, 1].axis('off')
    
    # Row 2: Corrected image and statistics
    if 'corrected_image' in result:
        axes[1, 0].imshow(result['corrected_image'], cmap='gray')
        axes[1, 0].set_title('Fully Corrected Image')
        axes[1, 0].axis('off')
        
        # Show correction statistics
        correction_stats = []
        for channel_name in CHANNEL_NAMES:
            if channel_name in result['channels']:
                corr_matrix = result['channels'][channel_name]['correction_matrix']
                correction_stats.append(f"{channel_name}: {np.min(corr_matrix):.3f}-{np.max(corr_matrix):.3f}")
        
        stats_text = "Correction Matrix Ranges:\n" + "\n".join(correction_stats)
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('Correction Statistics')
        axes[1, 1].axis('off')
    
    # Row 3: Histogram comparison
    if 'corrected_image' in result:
        axes[1, 2].hist(result['raw_data'].flatten(), bins=100, alpha=0.7, label='Original', density=True)
        axes[1, 2].hist(result['corrected_image'].flatten(), bins=100, alpha=0.7, label='Corrected', density=True)
        axes[1, 2].set_title('Histogram Comparison')
        axes[1, 2].set_xlabel('Pixel Value')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save comparison plot
    if SAVE_PLOTS:
        plot_filename = f"image_comparison_{Path(result['filename']).stem}_{plot_index+1}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Comparison plot saved: {plot_path}")
    
    plt.show()


def save_correction_matrices(results: List[Dict], output_dir: Path):
    """
    Save lens shading correction matrices
    
    Args:
        results: List of analysis results
        output_dir: Output directory path
    """
    if not SAVE_CORRECTION_MATRIX:
        return
    
    print(f"\nSaving correction matrices...")
    
    for i, result in enumerate(results):
        if not result['analysis_success']:
            continue
        
        base_filename = Path(result['filename']).stem
        
        for channel_name in CHANNEL_NAMES:
            channel_data = result['channels'][channel_name]
            
            # Save grid correction matrix
            grid_corr_path = output_dir / f"{base_filename}_{channel_name}_grid_correction.npy"
            np.save(grid_corr_path, channel_data['correction_matrix'])
            
            # Save full correction matrix
            full_corr_path = output_dir / f"{base_filename}_{channel_name}_full_correction.npy"
            np.save(full_corr_path, channel_data['full_correction'])
            
            print(f"  {channel_name} correction matrices saved for {base_filename}")
        
        # Skip saving histogram data per user request
    
    # Save combined correction data
    if len(results) > 0 and results[0]['analysis_success']:
        combined_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'image_dimensions': [IMAGE_HEIGHT, IMAGE_WIDTH],
            'grid_size': GRID_SIZE,
            'channel_names': CHANNEL_NAMES,
            'correction_matrices': {}
        }
        
        # Combine all channels from all images
        for channel_name in CHANNEL_NAMES:
            channel_corrections = []
            for result in results:
                if result['analysis_success']:
                    channel_corrections.append(result['channels'][channel_name]['correction_matrix'])
            
            if channel_corrections:
                # Calculate average correction matrix
                avg_correction = np.mean(channel_corrections, axis=0)
                combined_data['correction_matrices'][channel_name] = {
                    'average_correction': avg_correction.tolist(),
                    'shape': avg_correction.shape
                }
        
        # Save combined data
        combined_path = output_dir / "combined_lens_shading_correction.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Combined correction data saved: {combined_path}")


def save_corrected_images(results: List[Dict], output_dir: Path):
    """
    Save corrected images
    
    Args:
        results: List of analysis results
        output_dir: Output directory path
    """
    if not SAVE_CORRECTED_IMAGES:
        return
    
    print(f"\nSaving corrected images...")
    
    for i, result in enumerate(results):
        if not result['analysis_success']:
            continue
        
        base_filename = Path(result['filename']).stem
        
        try:
            # Save dark-corrected RAW data
            if 'corrected_raw_data' in result:
                dark_corr_path = output_dir / f"{base_filename}_dark_corrected.raw"
                corrected_data = result['corrected_raw_data']
                
                if corrected_data.dtype == np.float64:
                    # Convert float64 to uint16 for saving
                    corrected_data = np.clip(corrected_data, 0, 4095).astype(np.uint16)
                
                with open(dark_corr_path, 'wb') as f:
                    corrected_data.tofile(f)
                print(f"  Dark-corrected RAW saved: {dark_corr_path}")
            
            # Save fully corrected image
            if 'corrected_image' in result:
                # Save as RAW
                corrected_raw_path = output_dir / f"{base_filename}_fully_corrected.raw"
                corrected_data = result['corrected_image']
                
                if corrected_data.dtype == np.float64:
                    # Convert float64 to uint16 for saving
                    corrected_data = np.clip(corrected_data, 0, 4095).astype(np.uint16)
                
                with open(corrected_raw_path, 'wb') as f:
                    corrected_data.tofile(f)
                print(f"  Fully corrected RAW saved: {corrected_raw_path}")
                
                # Save as PNG (normalized to 8-bit)
                corrected_png_path = output_dir / f"{base_filename}_fully_corrected.png"
                normalized_data = np.clip(corrected_data, 0, 4095)
                normalized_data = (normalized_data / 4095 * 255).astype(np.uint8)
                
                # Use PIL for PNG to ensure 8-bit output
                pil_image = Image.fromarray(normalized_data, mode='L')
                pil_image.save(str(corrected_png_path), 'PNG')
                print(f"  Fully corrected PNG saved: {corrected_png_path}")
                
        except Exception as e:
            print(f"  Error saving corrected images for {base_filename}: {e}")


def load_correction_parameters(correction_dir: str) -> Dict[str, np.ndarray]:
    """
    Load lens shading correction parameters from saved files
    
    Args:
        correction_dir: Directory containing correction parameter files
        
    Returns:
        Dictionary containing correction parameters for each channel
    """
    correction_dir = Path(correction_dir)
    if not correction_dir.exists():
        raise FileNotFoundError(f"Correction directory not found: {correction_dir}")
    
    # Look for combined correction data first
    combined_file = correction_dir / "combined_lens_shading_correction.json"
    if combined_file.exists():
        print(f"Loading combined correction parameters from: {combined_file}")
        with open(combined_file, 'r', encoding='utf-8') as f:
            combined_data = json.load(f)
        
        correction_params = {}
        for channel_name, channel_data in combined_data['correction_matrices'].items():
            # Convert list back to numpy array
            correction_params[channel_name] = np.array(channel_data['average_correction'])
            print(f"  Loaded {channel_name} channel: {correction_params[channel_name].shape}")
        
        return correction_params
    
    # If no combined file, look for individual channel files
    print(f"Loading individual channel correction parameters from: {correction_dir}")
    correction_params = {}
    
    for channel_name in CHANNEL_NAMES:
        # Look for full correction matrix files
        pattern = f"*_{channel_name}_full_correction.npy"
        correction_files = list(correction_dir.glob(pattern))
        
        if correction_files:
            # Use the first file found (you might want to implement more sophisticated selection)
            correction_file = correction_files[0]
            correction_params[channel_name] = np.load(str(correction_file))
            print(f"  Loaded {channel_name} channel from: {correction_file.name}")
        else:
            print(f"  Warning: No correction file found for {channel_name} channel")
    
    if not correction_params:
        raise FileNotFoundError(f"No correction parameters found in: {correction_dir}")
    
    return correction_params

def shading_correct(input_image_path: str, correction_params: Dict[str, np.ndarray], 
                   dark_image_path: Optional[str] = None, output_dir: Optional[str] = None,
                   save_formats: List[str] = None) -> Dict[str, str]:
    """
    Apply lens shading correction to an input image using pre-saved correction parameters
    
    Args:
        input_image_path: Path to input RAW image
        correction_params: Dictionary of correction parameters for each channel
        dark_image_path: Optional path to dark current reference image
        output_dir: Output directory (None = same as input image)
        save_formats: List of formats to save ('raw', 'png', 'jpg'). Default: ['raw', 'png']
        
    Returns:
        Dictionary containing paths to saved corrected images
    """
    if save_formats is None:
        save_formats = ['raw', 'png']
    
    print(f"=== Lens Shading Correction ===")
    print(f"Input image: {input_image_path}")
    print(f"Dark correction: {dark_image_path is not None}")
    print(f"Output formats: {save_formats}")
    
    # Load input image
    input_path = Path(input_image_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    # Read RAW data
    raw_data = read_raw_image(str(input_path), IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
    print(f"Input image loaded: {raw_data.shape}, dtype: {raw_data.dtype}")
    print(f"Data range: {np.min(raw_data)} - {np.max(raw_data)}")
    
    # Apply dark current correction if provided
    corrected_raw_data = raw_data.copy()
    if dark_image_path is not None:
        dark_path = Path(dark_image_path)
        if not dark_path.exists():
            raise FileNotFoundError(f"Dark image not found: {dark_image_path}")
        
        dark_data = read_raw_image(str(dark_path), IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
        corrected_raw_data = apply_dark_correction(raw_data, dark_data)
        print(f"Dark correction applied")
    
    # Separate RGGB channels
    channels = separate_rggb_channels(corrected_raw_data)
    
    # Apply lens shading correction to each channel
    corrected_channels = {}
    for channel_name in CHANNEL_NAMES:
        if channel_name in correction_params:
            print(f"Applying correction to {channel_name} channel...")
            
            # Get correction matrix for this channel
            correction_matrix = correction_params[channel_name]
            
            # Resize correction matrix to match channel size if necessary
            channel_data = channels[channel_name]
            if correction_matrix.shape != channel_data.shape:
                print(f"  Resizing correction matrix from {correction_matrix.shape} to {channel_data.shape}")
                from scipy.interpolate import RectBivariateSpline
                
                # Create interpolation function
                grid_h, grid_w = correction_matrix.shape
                grid_y = np.linspace(0, channel_data.shape[0] - 1, grid_h)
                grid_x = np.linspace(0, channel_data.shape[1] - 1, grid_w)
                target_y = np.arange(channel_data.shape[0])
                target_x = np.arange(channel_data.shape[1])
                
                interp_func = RectBivariateSpline(grid_y, grid_x, correction_matrix, kx=3, ky=3)
                correction_matrix = interp_func(target_y, target_x)
            
            # Apply correction
            corrected_channel = apply_lens_shading_correction(channel_data, correction_matrix)
            corrected_channels[channel_name] = corrected_channel
        else:
            print(f"Warning: No correction parameters for {channel_name} channel, using original")
            corrected_channels[channel_name] = channels[channel_name]
    
    # Reconstruct corrected image
    corrected_image = reconstruct_corrected_image(corrected_channels, correction_params)
    
    # Prepare output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    base_filename = input_path.stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save corrected images in specified formats
    saved_files = {}
    
    try:
        if 'raw' in save_formats:
            # Save as RAW
            corrected_raw_path = output_dir / f"{base_filename}_shading_corrected_{timestamp}.raw"
            corrected_data = corrected_image
            
            if corrected_data.dtype == np.float64:
                corrected_data = np.clip(corrected_data, 0, 4095).astype(np.uint16)
             
            with open(corrected_raw_path, 'wb') as f:
                corrected_data.tofile(f)
            saved_files['raw'] = str(corrected_raw_path)
            print(f"Corrected RAW saved: {corrected_raw_path}")
        
        if 'png' in save_formats:
            # Save as PNG
            corrected_png_path = output_dir / f"{base_filename}_shading_corrected_{timestamp}.png"
            normalized_data = np.clip(corrected_image, 0, 4095)
            normalized_data = (normalized_data / 4095 * 255).astype(np.uint8)
            
            # Use PIL for PNG to ensure 8-bit output
            pil_image = Image.fromarray(normalized_data, mode='L')
            pil_image.save(str(corrected_png_path), 'PNG')
            saved_files['png'] = str(corrected_png_path)
            print(f"Corrected PNG saved: {corrected_png_path}")
        
        if 'jpg' in save_formats:
            # Save as JPG
            corrected_jpg_path = output_dir / f"{base_filename}_shading_corrected_{timestamp}.jpg"
            normalized_data = np.clip(corrected_image, 0, 4095)
            normalized_data = (normalized_data / 4095 * 255).astype(np.uint8)
            
            # Use PIL for JPG with quality control
            pil_image = Image.fromarray(normalized_data, mode='L')
            pil_image.save(str(corrected_jpg_path), 'JPEG', quality=95)
            saved_files['jpg'] = str(corrected_jpg_path)
            print(f"Corrected JPG saved: {corrected_jpg_path}")
        
        # Save correction summary
        # summary = {
        #     'correction_timestamp': timestamp,
        #     'input_image': str(input_path),
        #     'dark_correction_applied': dark_image_path is not None,
        #     'dark_image_path': dark_image_path,
        #     'correction_parameters_source': str(list(correction_params.keys())),
        #     'output_files': saved_files,
        #     'image_dimensions': [IMAGE_HEIGHT, IMAGE_WIDTH],
        #     'data_type': DATA_TYPE
        # }
        
        # summary_path = output_dir / f"{base_filename}_correction_summary_{timestamp}.json"
        # with open(summary_path, 'w', encoding='utf-8') as f:
        #     json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # print(f"Correction summary saved: {summary_path}")
        
    except Exception as e:
        print(f"Error saving corrected images: {e}")
        raise
    
    print(f"=== Correction Complete ===")
    print(f"Results saved to: {output_dir}")
    
    # Return both saved files and image data for histogram analysis
    result_data = {
        'saved_files': saved_files,
        'original_image': raw_data,
        'corrected_image': corrected_image,
        'dark_corrected_image': corrected_raw_data if dark_image_path is not None else None
    }
    
    return result_data


def batch_shading_correct(input_dir: str, correction_params: Dict[str, np.ndarray],
                         dark_image_path: Optional[str] = None, output_dir: Optional[str] = None,
                         save_formats: List[str] = None) -> List[Dict[str, str]]:
    """
    Apply lens shading correction to multiple images in batch
    
    Args:
        input_dir: Directory containing input RAW images
        correction_params: Dictionary of correction parameters for each channel
        dark_image_path: Optional path to dark current reference image
        output_dir: Output directory (None = same as input directory)
        save_formats: List of formats to save. Default: ['raw', 'png']
        
    Returns:
        List of dictionaries containing paths to saved corrected images for each input
    """
    if save_formats is None:
        save_formats = ['raw', 'png']
    
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all RAW files
    raw_files = list(input_path.glob("*.raw")) + list(input_path.glob("*.RAW"))
    if not raw_files:
        raise FileNotFoundError(f"No RAW files found in: {input_dir}")
    
    print(f"Found {len(raw_files)} RAW files for batch correction")
    
    # Prepare output directory
    if output_dir is None:
        output_dir = input_path / f"shading_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each file
    results = []
    for i, raw_file in enumerate(raw_files, 1):
        print(f"\nProcessing {i}/{len(raw_files)}: {raw_file.name}")
        try:
            result = shading_correct(
                str(raw_file), 
                correction_params, 
                dark_image_path, 
                str(output_dir), 
                save_formats
            )
            results.append({
                'input_file': str(raw_file),
                'output_files': result['saved_files'],
                'success': True
            })
        except Exception as e:
            print(f"Error processing {raw_file.name}: {e}")
            results.append({
                'input_file': str(raw_file),
                'error': str(e),
                'success': False
            })
    
    # Save batch processing summary
    batch_summary = {
        'batch_timestamp': datetime.now().isoformat(),
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'total_files': len(raw_files),
        'successful_corrections': len([r for r in results if r['success']]),
        'failed_corrections': len([r for r in results if not r['success']]),
        'save_formats': save_formats,
        'dark_correction_applied': dark_image_path is not None,
        'dark_image_path': dark_image_path,
        'results': results
    }
    
    summary_path = output_dir / "batch_correction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Batch Correction Complete ===")
    print(f"Total files: {len(raw_files)}")
    print(f"Successful: {batch_summary['successful_corrections']}")
    print(f"Failed: {batch_summary['failed_corrections']}")
    print(f"Results saved to: {output_dir}")
    
    return results


def main():
    """Main function"""
    print("=== Lens Shading Analysis Tool ===")
    print(f"Input path: {INPUT_PATH}")
    print(f"Dark correction: {ENABLE_DARK_CORRECTION}")
    if ENABLE_DARK_CORRECTION:
        print(f"Dark reference: {DARK_RAW_PATH}")
    print(f"Dimensions: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"Data type: {DATA_TYPE}")
    print(f"Grid size: {GRID_SIZE}")
    print(f"Channel names: {CHANNEL_NAMES}")
    print(f"Valid value range: {MIN_VALID_VALUE} - {MAX_VALID_VALUE}")
    print()
    
    # Check input path
    input_path = Path(INPUT_PATH)
    
    if not input_path.exists():
        print(f"Error: Input path not found: {INPUT_PATH}")
        return
    
    # Load dark reference if enabled
    dark_data = None
    dark_correction_enabled = ENABLE_DARK_CORRECTION  # Create local copy
    if dark_correction_enabled:
        dark_path = Path(DARK_RAW_PATH)
        if not dark_path.exists():
            print(f"Error: Dark reference path not found: {DARK_RAW_PATH}")
            return
        
        dark_data = load_dark_reference(str(dark_path), IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
        if dark_data is None:
            print("Warning: Failed to load dark reference, continuing without dark correction")
            dark_correction_enabled = False
    
    # Create output directory
    if OUTPUT_DIRECTORY is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = input_path / f"lens_shading_analysis_{timestamp}"
    else:
        output_dir = Path(OUTPUT_DIRECTORY)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find RAW files
    if input_path.is_file():
        raw_files = [str(input_path)]
    else:
        raw_files = list(input_path.glob("*.raw")) + list(input_path.glob("*.RAW"))
        raw_files = [str(f) for f in raw_files]
    
    if not raw_files:
        print("No RAW files found!")
        return
    
    print(f"\nFound {len(raw_files)} RAW files")
    
    # Analyze images
    results = []
    for raw_file in raw_files:
        result = analyze_single_image(raw_file, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE, GRID_SIZE, dark_data)
        results.append(result)
    
    # Generate plots
    create_lens_shading_plots(results, output_dir)
    
    # Save correction matrices
    save_correction_matrices(results, output_dir)
    
    # Save corrected images
    save_corrected_images(results, output_dir)
    
    # Save analysis summary
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'input_path': INPUT_PATH,
        'dark_correction_enabled': dark_correction_enabled,
        'dark_reference_path': DARK_RAW_PATH if dark_correction_enabled else None,
        'total_files': len(raw_files),
        'successful_analysis': len([r for r in results if r['analysis_success']]),
        'failed_analysis': len([r for r in results if not r['analysis_success']]),
        'settings': {
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'data_type': DATA_TYPE,
            'grid_size': GRID_SIZE,
            'channel_names': CHANNEL_NAMES,
            'min_valid_value': MIN_VALID_VALUE,
            'max_valid_value': MAX_VALID_VALUE
        }
    }
    
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()


# ============================================================================
# 使用示例 - 应用已保存的矫正参数
# ============================================================================

def example_usage():
    """
    示例：如何使用shading_correct函数应用已保存的矫正参数
    """
    print("=== 使用示例 ===")
    print("1. 加载矫正参数")
    print("2. 应用矫正到单张图片")
    print("3. 批量矫正多张图片")
    
    # 示例1: 加载矫正参数
    # correction_params = load_correction_parameters(r"F:\ZJU\Picture\lens shading\lens_shading_analysis_20241201_143022")
    
    # 示例2: 矫正单张图片
    # result = shading_correct(
    #     input_image_path=r"F:\ZJU\Picture\lens shading\new_image.raw",
    #     correction_params=correction_params,
    #     dark_image_path=r"F:\ZJU\Picture\dark\g8\average_dark.raw",  # 可选
    #     output_dir=r"F:\ZJU\Picture\lens shading\corrected",
    #     save_formats=['raw', 'png', 'jpg']
    # )
    
    # 示例3: 批量矫正
    # batch_results = batch_shading_correct(
    #     input_dir=r"F:\ZJU\Picture\lens shading\new_images",
    #     correction_params=correction_params,
    #     dark_image_path=r"F:\ZJU\Picture\dark\g8\average_dark.raw",  # 可选
    #     output_dir=r"F:\ZJU\Picture\lens shading\batch_corrected",
    #     save_formats=['raw', 'png']
    # )
    
    print("请取消注释上述代码并修改路径来使用这些功能")


# 如果要使用示例，取消注释下面这行
# example_usage()
