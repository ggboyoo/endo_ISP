#!/usr/bin/env python3
"""
Noise Calibration Program
读取噪声标定拍摄的灰阶板图RAW数据，输出展示直方图
"""

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # Set backend for plot display
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import json
from datetime import datetime
import argparse
import pandas as pd

# Import functions from raw_reader.py
try:
    from raw_reader import read_raw_image, demosaic_image_corrected_fixed
except ImportError:
    print("Error: raw_reader.py not found in the same directory!")
    print("Please ensure raw_reader.py is in the same directory as this script.")
    exit(1)

# ============================================================================
# 配置文件路径 - 直接在这里修改，无需交互输入
# ============================================================================

# 输入路径配置
INPUT_PATH = r"F:\ZJU\Picture\denoise\g8\1.raw"  # 灰阶板图RAW文件夹路径
DARK_RAW_PATH = r"F:\ZJU\Picture\dark\g8\average_dark.raw"  # 暗电流图像路径（可选）

# 图像参数配置
IMAGE_WIDTH = 3840      # 图像宽度
IMAGE_HEIGHT = 2160     # 图像高度

DATA_TYPE = 'uint16'    # 数据类型

# 输出配置
OUTPUT_DIRECTORY =  r"F:\ZJU\Picture\denoise\g8"  # 输出目录（None为自动生成）
GENERATE_PLOTS = True   # 是否生成直方图
SAVE_PLOTS = True       # 是否保存直方图文件
SAVE_DATA = True        # 是否保存噪声统计数据

# 暗电流校正配置
DARK_SUBTRACTION_ENABLED = True  # 是否启用暗电流校正
CLIP_NEGATIVE_VALUES = True      # 是否将负值裁剪为0

# 灰阶板配置
GRAY_CHART_PATCHES = 24  # 灰阶板色块数量（通常为24个）
PATCH_ROWS = 4           # 色块行数
PATCH_COLS = 6           # 色块列数

# 噪声分析配置
NOISE_ANALYSIS_ENABLED = True    # 是否启用噪声分析
HISTOGRAM_BINS = 256            # 直方图bin数量
SHOW_INDIVIDUAL_PATCHES = True  # 是否显示每个色块的直方图
SHOW_OVERALL_HISTOGRAM = True   # 是否显示整体直方图

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

def apply_dark_subtraction(raw_data: np.ndarray, dark_data: np.ndarray) -> np.ndarray:
    """
    Apply dark current subtraction
    
    Args:
        raw_data: Original RAW data
        dark_data: Dark reference data
        
    Returns:
        Dark-subtracted data
    """
    print("Applying dark current subtraction...")
    
    try:
        # Convert to float for calculation
        raw_float = raw_data.astype(np.float32)
        dark_float = dark_data.astype(np.float32)
        
        # Subtract dark current
        corrected = raw_float - dark_float
        
        # Clip negative values if enabled
        if CLIP_NEGATIVE_VALUES:
            corrected = np.clip(corrected, 0, np.max(raw_float))
        
        print(f"  Dark subtraction applied")
        print(f"  Original range: {np.min(raw_data)} - {np.max(raw_data)}")
        print(f"  Corrected range: {np.min(corrected)} - {np.max(corrected)}")
        
        return corrected.astype(np.uint16)
        
    except Exception as e:
        print(f"  Error applying dark subtraction: {e}")
        return raw_data

def extract_gray_patches(image: np.ndarray, rows: int = 4, cols: int = 6) -> List[np.ndarray]:
    """
    Extract gray patches from the image
    
    Args:
        image: Input image
        rows: Number of patch rows
        cols: Number of patch columns
        
    Returns:
        List of patch arrays
    """
    print(f"Extracting {rows}x{cols} gray patches...")
    
    try:
        h, w = image.shape[:2]
        patch_h = h // rows
        patch_w = w // cols
        
        patches = []
        patch_positions = []
        
        for i in range(rows):
            for j in range(cols):
                # Calculate patch boundaries
                start_y = i * patch_h
                end_y = (i + 1) * patch_h
                start_x = j * patch_w
                end_x = (j + 1) * patch_w
                
                # Extract patch
                patch = image[start_y:end_y, start_x:end_x]
                patches.append(patch)
                patch_positions.append((start_x, start_y, patch_w, patch_h))
                
                print(f"  Patch {i*cols + j + 1}: position=({start_x}, {start_y}), size=({patch_w}, {patch_h})")
        
        print(f"  Extracted {len(patches)} patches")
        return patches, patch_positions
        
    except Exception as e:
        print(f"  Error extracting patches: {e}")
        return [], []

def analyze_patch_noise(patch: np.ndarray, patch_id: int) -> Dict:
    """
    Analyze noise statistics for a single patch
    
    Args:
        patch: Patch image data
        patch_id: Patch identifier
        
    Returns:
        Dictionary containing noise statistics
    """
    try:
        # Convert to float for analysis
        patch_float = patch.astype(np.float32)
        
        # Calculate basic statistics
        mean_val = np.mean(patch_float)
        std_val = np.std(patch_float)
        var_val = np.var(patch_float)
        min_val = np.min(patch_float)
        max_val = np.max(patch_float)
        median_val = np.median(patch_float)
        
        # Calculate noise metrics
        snr = mean_val / std_val if std_val > 0 else 0
        cv = std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
        
        # Calculate histogram
        hist, bins = np.histogram(patch_float, bins=HISTOGRAM_BINS, range=(min_val, max_val))
        
        stats = {
            'patch_id': patch_id,
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
            'patch_size': patch.shape
        }
        
        return stats
        
    except Exception as e:
        print(f"  Error analyzing patch {patch_id}: {e}")
        return {'patch_id': patch_id, 'error': str(e)}

def create_histogram_plots(patches: List[np.ndarray], patch_stats: List[Dict], output_dir: Path):
    """
    Create histogram plots for noise analysis
    
    Args:
        patches: List of patch arrays
        patch_stats: List of patch statistics
        output_dir: Output directory
    """
    print("Creating histogram plots...")
    
    try:
        if SHOW_INDIVIDUAL_PATCHES:
            # Create individual patch histograms
            fig, axes = plt.subplots(PATCH_ROWS, PATCH_COLS, figsize=(20, 16))
            fig.suptitle('Gray Chart Patch Histograms', fontsize=16)
            
            for i, (patch, stats) in enumerate(zip(patches, patch_stats)):
                row = i // PATCH_COLS
                col = i % PATCH_COLS
                ax = axes[row, col]
                
                if 'error' not in stats:
                    # Plot histogram
                    hist = np.array(stats['histogram'])
                    bins = np.array(stats['bins'])
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    
                    ax.bar(bin_centers, hist, width=(bins[1] - bins[0]), alpha=0.7)
                    ax.set_title(f'Patch {i+1}\nMean: {stats["mean"]:.1f}, Std: {stats["std"]:.1f}')
                    ax.set_xlabel('Pixel Value')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Patch {i+1}\nError: {stats["error"]}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Patch {i+1} - Error')
            
            plt.tight_layout()
            
            if SAVE_PLOTS:
                plot_path = output_dir / "gray_chart_patch_histograms.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"  Patch histograms saved: {plot_path}")
            
            if GENERATE_PLOTS:
                plt.show()
            else:
                plt.close()
        
        if SHOW_OVERALL_HISTOGRAM:
            # Create overall histogram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Overall image histogram
            all_pixels = np.concatenate([patch.flatten() for patch in patches])
            ax1.hist(all_pixels, bins=HISTOGRAM_BINS, alpha=0.7, color='blue')
            ax1.set_title('Overall Gray Chart Histogram')
            ax1.set_xlabel('Pixel Value')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Noise statistics by patch
            patch_ids = [stats['patch_id'] for stats in patch_stats if 'error' not in stats]
            means = [stats['mean'] for stats in patch_stats if 'error' not in stats]
            stds = [stats['std'] for stats in patch_stats if 'error' not in stats]
            
            ax2.plot(patch_ids, means, 'o-', label='Mean', linewidth=2, markersize=6)
            ax2.plot(patch_ids, stds, 's-', label='Std Dev', linewidth=2, markersize=6)
            ax2.set_title('Noise Statistics by Patch')
            ax2.set_xlabel('Patch ID')
            ax2.set_ylabel('Pixel Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if SAVE_PLOTS:
                plot_path = output_dir / "gray_chart_overall_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"  Overall analysis saved: {plot_path}")
            
            if GENERATE_PLOTS:
                plt.show()
            else:
                plt.close()
        
        # Create noise summary plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gray Chart Noise Analysis Summary', fontsize=16)
        
        # Plot 1: Mean values
        patch_ids = [stats['patch_id'] for stats in patch_stats if 'error' not in stats]
        means = [stats['mean'] for stats in patch_stats if 'error' not in stats]
        axes[0, 0].plot(patch_ids, means, 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Mean Values by Patch')
        axes[0, 0].set_xlabel('Patch ID')
        axes[0, 0].set_ylabel('Mean Pixel Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation
        stds = [stats['std'] for stats in patch_stats if 'error' not in stats]
        axes[0, 1].plot(patch_ids, stds, 's-', linewidth=2, markersize=6, color='red')
        axes[0, 1].set_title('Standard Deviation by Patch')
        axes[0, 1].set_xlabel('Patch ID')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: SNR
        snrs = [stats['snr'] for stats in patch_stats if 'error' not in stats]
        axes[1, 0].plot(patch_ids, snrs, '^-', linewidth=2, markersize=6, color='green')
        axes[1, 0].set_title('Signal-to-Noise Ratio by Patch')
        axes[1, 0].set_xlabel('Patch ID')
        axes[1, 0].set_ylabel('SNR')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Coefficient of variation
        cvs = [stats['cv'] for stats in patch_stats if 'error' not in stats]
        axes[1, 1].plot(patch_ids, cvs, 'd-', linewidth=2, markersize=6, color='orange')
        axes[1, 1].set_title('Coefficient of Variation by Patch')
        axes[1, 1].set_xlabel('Patch ID')
        axes[1, 1].set_ylabel('CV')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plot_path = output_dir / "gray_chart_noise_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Noise summary saved: {plot_path}")
        
        if GENERATE_PLOTS:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"  Error creating histogram plots: {e}")
        import traceback
        traceback.print_exc()

def save_noise_data(patch_stats: List[Dict], output_dir: Path, base_filename: str):
    """
    Save noise analysis data to JSON file
    
    Args:
        patch_stats: List of patch statistics
        output_dir: Output directory
        base_filename: Base filename
    """
    if not SAVE_DATA:
        return
    
    print("Saving noise analysis data...")
    
    try:
        # Calculate overall statistics
        valid_stats = [stats for stats in patch_stats if 'error' not in stats]
        
        if valid_stats:
            overall_stats = {
                'mean_of_means': float(np.mean([stats['mean'] for stats in valid_stats])),
                'std_of_means': float(np.std([stats['mean'] for stats in valid_stats])),
                'mean_of_stds': float(np.mean([stats['std'] for stats in valid_stats])),
                'std_of_stds': float(np.std([stats['std'] for stats in valid_stats])),
                'mean_snr': float(np.mean([stats['snr'] for stats in valid_stats])),
                'mean_cv': float(np.mean([stats['cv'] for stats in valid_stats])),
                'total_patches': len(valid_stats)
            }
        else:
            overall_stats = {'error': 'No valid patches found'}
        
        # Create complete data structure
        noise_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'input_file': base_filename,
            'image_dimensions': [IMAGE_WIDTH, IMAGE_HEIGHT],
            'data_type': DATA_TYPE,
            'patch_configuration': {
                'rows': PATCH_ROWS,
                'cols': PATCH_COLS,
                'total_patches': GRAY_CHART_PATCHES
            },
            'processing_settings': {
                'dark_subtraction_enabled': DARK_SUBTRACTION_ENABLED,
                'histogram_bins': HISTOGRAM_BINS,
                'clip_negative_values': CLIP_NEGATIVE_VALUES
            },
            'overall_statistics': overall_stats,
            'patch_statistics': patch_stats
        }
        
        # Save to JSON file
        data_path = output_dir / f"{base_filename}_noise_analysis.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(noise_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Noise data saved: {data_path}")
        
    except Exception as e:
        print(f"  Error saving noise data: {e}")
        import traceback
        traceback.print_exc()

def select_roi_rectangle(image: np.ndarray, window_name: str = "Select ROI Rectangle") -> Optional[Tuple[int, int, int, int]]:
    """
    Interactive ROI rectangle selection using OpenCV mouse callbacks
    
    Args:
        image: Input image for ROI selection
        window_name: Window name for display
        
    Returns:
        Tuple of (x, y, width, height) or None if cancelled
    """
    print(f"  Please select ROI rectangle in the window: {window_name}")
    print(f"  Instructions:")
    print(f"    - Click and drag to select a rectangle")
    print(f"    - Press 'Enter' to confirm selection")
    print(f"    - Press 'r' to reset selection")
    print(f"    - Press 'Esc' to cancel")
    
    # Normalize image to 8-bit for display
    if image.dtype != np.uint8:
        img_display = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    else:
        img_display = image.copy()
    
    # Convert to BGR for OpenCV display
    if len(img_display.shape) == 2:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
    
    # Variables for rectangle selection
    drawing = False
    start_point = None
    end_point = None
    current_rect = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, current_rect
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
            current_rect = (x, y, 0, 0)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
                current_rect = (min(start_point[0], end_point[0]), 
                              min(start_point[1], end_point[1]),
                              abs(end_point[0] - start_point[0]),
                              abs(end_point[1] - start_point[1]))
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            current_rect = (min(start_point[0], end_point[0]), 
                          min(start_point[1], end_point[1]),
                          abs(end_point[0] - start_point[0]),
                          abs(end_point[1] - start_point[1]))
    
    # Create window and set mouse callback
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        # Create display image with current rectangle
        display_img = img_display.copy()
        
        if current_rect is not None and current_rect[2] > 0 and current_rect[3] > 0:
            # Draw rectangle
            cv2.rectangle(display_img, 
                         (current_rect[0], current_rect[1]),
                         (current_rect[0] + current_rect[2], current_rect[1] + current_rect[3]),
                         (0, 255, 0), 2)
            
            # Draw corner points
            cv2.circle(display_img, (current_rect[0], current_rect[1]), 5, (0, 0, 255), -1)
            cv2.circle(display_img, (current_rect[0] + current_rect[2], current_rect[1] + current_rect[3]), 5, (0, 0, 255), -1)
            
            # Add text with rectangle info
            text = f"ROI: {current_rect[2]}x{current_rect[3]} pixels"
            cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            if current_rect is not None and current_rect[2] > 0 and current_rect[3] > 0:
                cv2.destroyWindow(window_name)
                return current_rect
            else:
                print("  Please select a valid rectangle first")
        
        elif key == ord('r'):  # Reset
            current_rect = None
            start_point = None
            end_point = None
            print("  Selection reset")
        
        elif key == 27:  # Esc key
            cv2.destroyWindow(window_name)
            return None
    
    cv2.destroyWindow(window_name)
    return None

def separate_rggb_channels(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Separate RGGB channels from RAW image
    
    Args:
        image: Input RAW image (single channel)
        
    Returns:
        Dictionary containing R, G1, G2, B channels
    """
    height, width = image.shape
    
    # Extract RGGB channels (assuming RGGB Bayer pattern)
    # R: (0,0), G1: (0,1), G2: (1,0), B: (1,1)
    r_channel = image[0::2, 0::2]      # Red channel
    g1_channel = image[0::2, 1::2]     # Green channel 1
    g2_channel = image[1::2, 0::2]     # Green channel 2
    b_channel = image[1::2, 1::2]      # Blue channel
    
    return {
        'R': r_channel,
        'G1': g1_channel,
        'G2': g2_channel,
        'B': b_channel
    }

def analyze_rggb_roi_statistics(image: np.ndarray, roi_rect: Tuple[int, int, int, int]) -> Dict:
    """
    Analyze RGGB channel statistics for a specific ROI region
    
    Args:
        image: Input image
        roi_rect: ROI rectangle (x, y, width, height)
        
    Returns:
        Dictionary containing RGGB channel statistics
    """
    try:
        x, y, w, h = roi_rect
        
        # Extract ROI region
        roi_data = image[y:y+h, x:x+w]
        
        # Separate RGGB channels
        channels = separate_rggb_channels(roi_data)
        
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
            
            channel_stats[channel_name] = {
                'mean': float(mean_val),
                'variance': float(var_val)
            }
        
        # Calculate overall statistics
        overall_stats = {
            'roi_rectangle': roi_rect,
            'channels': channel_stats
        }
        
        return overall_stats
        
    except Exception as e:
        print(f"  Error analyzing RGGB ROI statistics: {e}")
        return {'error': str(e)}

def analyze_roi_statistics(image: np.ndarray, roi_rect: Tuple[int, int, int, int]) -> Dict:
    """
    Analyze statistics for a specific ROI region
    
    Args:
        image: Input image
        roi_rect: ROI rectangle (x, y, width, height)
        
    Returns:
        Dictionary containing ROI statistics
    """
    try:
        x, y, w, h = roi_rect
        
        # Extract ROI region
        roi_data = image[y:y+h, x:x+w]
        
        # Convert to float for analysis
        roi_float = roi_data.astype(np.float32)
        
        # Calculate basic statistics
        mean_val = np.mean(roi_float)
        std_val = np.std(roi_float)
        var_val = np.var(roi_float)
        min_val = np.min(roi_float)
        max_val = np.max(roi_float)
        median_val = np.median(roi_float)
        
        # Calculate noise metrics
        snr = mean_val / std_val if std_val > 0 else 0
        cv = std_val / mean_val if mean_val > 0 else 0
        
        # Calculate histogram
        hist, bins = np.histogram(roi_float, bins=256, range=(min_val, max_val))
        
        roi_stats = {
            'roi_rectangle': roi_rect,
            'roi_size': (w, h),
            'roi_pixels': w * h,
            'mean': float(mean_val),
            'std': float(std_val),
            'variance': float(var_val),
            'min': float(min_val),
            'max': float(max_val),
            'median': float(median_val),
            'snr': float(snr),
            'cv': float(cv),
            'histogram': hist.tolist(),
            'bins': bins.tolist()
        }
        
        return roi_stats
        
    except Exception as e:
        print(f"  Error analyzing ROI statistics: {e}")
        return {'error': str(e)}

def save_rggb_roi_statistics(rggb_roi_stats: Dict, output_dir: Path, base_filename: str):
    """
    Save RGGB ROI statistics to JSON file
    
    Args:
        rggb_roi_stats: RGGB ROI statistics dictionary
        output_dir: Output directory
        base_filename: Base filename
    """
    if not SAVE_DATA:
        return
    
    print("Saving RGGB ROI statistics...")
    
    try:
        # Create complete data structure
        rggb_roi_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_type': 'rggb_roi_statistics',
            'image_dimensions': [IMAGE_WIDTH, IMAGE_HEIGHT],
            'data_type': DATA_TYPE,
            'processing_settings': {
                'dark_subtraction_enabled': DARK_SUBTRACTION_ENABLED,
                'clip_negative_values': CLIP_NEGATIVE_VALUES
            },
            'rggb_roi_statistics': rggb_roi_stats
        }
        
        # Save to JSON file
        data_path = output_dir / f"{base_filename}_rggb_roi_statistics.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(rggb_roi_data, f, indent=2, ensure_ascii=False)
        
        print(f"  RGGB ROI statistics saved: {data_path}")
        
    except Exception as e:
        print(f"  Error saving RGGB ROI statistics: {e}")
        import traceback
        traceback.print_exc()

def save_rggb_to_excel(channel_stats: Dict, roi_corners: List, filename: str, output_dir: Path):
    """
    Save RGGB statistics to Excel file, appending to existing data
    
    Args:
        channel_stats: Dictionary containing channel statistics
        roi_corners: ROI corner coordinates
        filename: Source filename
        output_dir: Output directory
    """
    try:
        # Create Excel file path
        excel_path = output_dir / "rggb_average_results.xlsx"
        
        # Prepare data for Excel
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create row data
        row_data = {
            'Timestamp': current_time,
            'Filename': filename,
            'ROI_Corners': str(roi_corners),
            'R_Mean': channel_stats['R']['mean'],
            'R_Variance': channel_stats['R']['variance'],
            'R_Pixels': channel_stats['R']['pixel_count'],
            'G1_Mean': channel_stats['G1']['mean'],
            'G1_Variance': channel_stats['G1']['variance'],
            'G1_Pixels': channel_stats['G1']['pixel_count'],
            'G2_Mean': channel_stats['G2']['mean'],
            'G2_Variance': channel_stats['G2']['variance'],
            'G2_Pixels': channel_stats['G2']['pixel_count'],
            'B_Mean': channel_stats['B']['mean'],
            'B_Variance': channel_stats['B']['variance'],
            'B_Pixels': channel_stats['B']['pixel_count']
        }
        
        # Check if Excel file exists
        if excel_path.exists():
            # Read existing data
            try:
                existing_df = pd.read_excel(excel_path)
                # Append new row
                new_df = pd.DataFrame([row_data])
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                print(f"  Warning: Could not read existing Excel file: {e}")
                # Create new DataFrame if reading fails
                combined_df = pd.DataFrame([row_data])
        else:
            # Create new DataFrame
            combined_df = pd.DataFrame([row_data])
        
        # Save to Excel
        combined_df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"  RGGB data saved to Excel: {excel_path}")
        print(f"  Total records in Excel: {len(combined_df)}")
        
    except Exception as e:
        print(f"  Error saving to Excel: {e}")
        import traceback
        traceback.print_exc()

def save_roi_statistics(roi_stats: Dict, output_dir: Path, base_filename: str):
    """
    Save ROI statistics to JSON file
    
    Args:
        roi_stats: ROI statistics dictionary
        output_dir: Output directory
        base_filename: Base filename
    """
    if not SAVE_DATA:
        return
    
    print("Saving ROI statistics...")
    
    try:
        # Create complete data structure
        roi_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_type': 'roi_statistics',
            'image_dimensions': [IMAGE_WIDTH, IMAGE_HEIGHT],
            'data_type': DATA_TYPE,
            'processing_settings': {
                'dark_subtraction_enabled': DARK_SUBTRACTION_ENABLED,
                'clip_negative_values': CLIP_NEGATIVE_VALUES
            },
            'roi_statistics': roi_stats
        }
        
        # Save to JSON file
        data_path = output_dir / f"{base_filename}_roi_statistics.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(roi_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ROI statistics saved: {data_path}")
        
    except Exception as e:
        print(f"  Error saving ROI statistics: {e}")
        import traceback
        traceback.print_exc()

def analyze_whole_image_noise(image: np.ndarray) -> Dict:
    """
    Analyze noise statistics for the whole image
    
    Args:
        image: Input image data
        
    Returns:
        Dictionary containing noise statistics
    """
    try:
        # Convert to float for analysis
        image_float = image.astype(np.float32)
        
        # Calculate basic statistics
        mean_val = np.mean(image_float)
        std_val = np.std(image_float)
        var_val = np.var(image_float)
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        median_val = np.median(image_float)
        
        # Calculate noise metrics
        snr = mean_val / std_val if std_val > 0 else 0
        cv = std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
        
        # Calculate histogram
        hist, bins = np.histogram(image_float, bins=HISTOGRAM_BINS, range=(min_val, max_val))
        
        stats = {
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
            'image_size': image.shape,
            'total_pixels': int(image.size)
        }
        
        return stats
        
    except Exception as e:
        print(f"  Error analyzing whole image: {e}")
        return {'error': str(e)}

def create_whole_image_histogram(image: np.ndarray, image_stats: Dict, output_dir: Path):
    """
    Create histogram plots for whole image analysis
    
    Args:
        image: Input image
        image_stats: Image statistics
        output_dir: Output directory
    """
    print("Creating whole image histogram plots...")
    
    try:
        if 'error' in image_stats:
            print(f"  Error in image stats: {image_stats['error']}")
            return
        
        # Create main histogram plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Whole Image Noise Analysis', fontsize=16)
        
        # Plot 1: Histogram
        hist = np.array(image_stats['histogram'])
        bins = np.array(image_stats['bins'])
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        axes[0, 0].bar(bin_centers, hist, width=(bins[1] - bins[0]), alpha=0.7, color='blue')
        axes[0, 0].set_title('Pixel Value Histogram')
        axes[0, 0].set_xlabel('Pixel Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Set adaptive axis limits for histogram
        value_range = np.max(bin_centers) - np.min(bin_centers)
        axes[0, 0].set_xlim(np.min(bin_centers) - 0.02 * value_range, 
                           np.max(bin_centers) + 0.02 * value_range)
        
        # Add statistics text
        stats_text = f"Mean: {image_stats['mean']:.1f}\n"
        stats_text += f"Std: {image_stats['std']:.1f}\n"
        stats_text += f"Min: {image_stats['min']:.1f}\n"
        stats_text += f"Max: {image_stats['max']:.1f}\n"
        stats_text += f"Median: {image_stats['median']:.1f}"
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Log-scale histogram
        axes[0, 1].bar(bin_centers, hist, width=(bins[1] - bins[0]), alpha=0.7, color='green')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Pixel Value Histogram (Log Scale)')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Frequency (Log Scale)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Set adaptive axis limits for log-scale histogram
        axes[0, 1].set_xlim(np.min(bin_centers) - 0.02 * value_range, 
                           np.max(bin_centers) + 0.02 * value_range)
        
        # Plot 3: Image preview (normalized to 8-bit for display)
        image_8bit = (image.astype(np.float32) / np.max(image) * 255).astype(np.uint8)
        axes[1, 0].imshow(image_8bit, cmap='gray')
        axes[1, 0].set_title('Image Preview')
        axes[1, 0].axis('off')
        
        # Plot 4: Noise metrics
        metrics_text = f"Signal-to-Noise Ratio: {image_stats['snr']:.2f}\n"
        metrics_text += f"Coefficient of Variation: {image_stats['cv']:.4f}\n"
        metrics_text += f"Variance: {image_stats['variance']:.1f}\n"
        metrics_text += f"Total Pixels: {image_stats['total_pixels']:,}\n"
        metrics_text += f"Image Size: {image_stats['image_size']}"
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Noise Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plot_path = output_dir / "whole_image_histogram.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Whole image histogram saved: {plot_path}")
        
        if GENERATE_PLOTS:
            plt.show()
        else:
            plt.close()
        
        # Create detailed histogram with more bins
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Use more bins for detailed histogram
        detailed_hist, detailed_bins = np.histogram(image.astype(np.float32), bins=1024)
        detailed_bin_centers = (detailed_bins[:-1] + detailed_bins[1:]) / 2
        
        ax.bar(detailed_bin_centers, detailed_hist, width=(detailed_bins[1] - detailed_bins[0]), 
               alpha=0.7, color='purple')
        ax.set_title('Detailed Pixel Value Histogram (1024 bins)', fontsize=14)
        ax.set_xlabel('Pixel Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set adaptive axis limits for detailed histogram
        detailed_value_range = np.max(detailed_bin_centers) - np.min(detailed_bin_centers)
        ax.set_xlim(np.min(detailed_bin_centers) - 0.02 * detailed_value_range, 
                   np.max(detailed_bin_centers) + 0.02 * detailed_value_range)
        
        # Add vertical lines for key statistics
        ax.axvline(image_stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {image_stats["mean"]:.1f}')
        ax.axvline(image_stats['median'], color='orange', linestyle='--', linewidth=2, label=f'Median: {image_stats["median"]:.1f}')
        ax.legend()
        
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plot_path = output_dir / "detailed_histogram.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Detailed histogram saved: {plot_path}")
        
        if GENERATE_PLOTS:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"  Error creating histogram plots: {e}")
        import traceback
        traceback.print_exc()

def save_whole_image_data(image_stats: Dict, output_dir: Path, base_filename: str):
    """
    Save whole image analysis data to JSON file
    
    Args:
        image_stats: Image statistics
        output_dir: Output directory
        base_filename: Base filename
    """
    if not SAVE_DATA:
        return
    
    print("Saving whole image analysis data...")
    
    try:
        # Create complete data structure
        noise_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'input_file': base_filename,
            'image_dimensions': [IMAGE_WIDTH, IMAGE_HEIGHT],
            'data_type': DATA_TYPE,
            'analysis_type': 'whole_image',
            'processing_settings': {
                'dark_subtraction_enabled': DARK_SUBTRACTION_ENABLED,
                'histogram_bins': HISTOGRAM_BINS,
                'clip_negative_values': CLIP_NEGATIVE_VALUES
            },
            'image_statistics': image_stats
        }
        
        # Save to JSON file
        data_path = output_dir / f"{base_filename}_whole_image_analysis.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(noise_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Whole image data saved: {data_path}")
        
    except Exception as e:
        print(f"  Error saving whole image data: {e}")
        import traceback
        traceback.print_exc()

def select_roi_corners(image: np.ndarray, window_name: str = "Select ROI Corners") -> Optional[List[Tuple[int, int]]]:
    """
    Interactive ROI corner selection
    
    Args:
        image: Input image for corner selection
        window_name: Window name for display
        
    Returns:
        List of 4 corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] or None if cancelled
    """
    print(f"Interactive ROI corner selection...")
    print(f"Instructions:")
    print(f"  1. Click 4 points to define ROI corners")
    print(f"  2. Click in order: top-left, top-right, bottom-right, bottom-left")
    print(f"  3. Press 'r' to reset selection")
    print(f"  4. Press 'Enter' to confirm selection")
    print(f"  5. Press 'Esc' to cancel")
    
    corners = []
    temp_image = image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal corners, temp_image
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 4:
                corners.append((x, y))
                print(f"  Corner {len(corners)}: ({x}, {y})")
                
                # Draw corner point
                cv2.circle(temp_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(temp_image, str(len(corners)), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw lines between corners
                if len(corners) > 1:
                    cv2.line(temp_image, corners[-2], corners[-1], (0, 255, 0), 2)
                
                # Draw closing line when 4 corners are selected
                if len(corners) == 4:
                    cv2.line(temp_image, corners[-1], corners[0], (0, 255, 0), 2)
                    print(f"  ROI selection complete! Press Enter to confirm or 'r' to reset.")
        
        cv2.imshow(window_name, temp_image)
    
    # Normalize image for display
    display_image = (image.astype(np.float32) / np.max(image) * 255).astype(np.uint8)
    if len(display_image.shape) == 2:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
    
    temp_image = display_image.copy()
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, temp_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            if len(corners) == 4:
                print(f"  ROI confirmed with corners: {corners}")
                cv2.destroyWindow(window_name)
                return corners
            else:
                print(f"  Please select exactly 4 corners (currently {len(corners)})")
        
        elif key == ord('r'):  # Reset
            corners = []
            temp_image = display_image.copy()
            cv2.imshow(window_name, temp_image)
            print(f"  Selection reset. Please select 4 corners again.")
        
        elif key == 27:  # Esc key
            print(f"  ROI selection cancelled.")
            cv2.destroyWindow(window_name)
            return None
    
    cv2.destroyWindow(window_name)
    return None

def apply_roi_mask(image: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
    """
    Apply ROI mask to image based on selected corners
    
    Args:
        image: Input image
        corners: List of 4 corner coordinates
        
    Returns:
        Masked image (pixels outside ROI set to 0)
    """
    if len(corners) != 4:
        print(f"  Error: Expected 4 corners, got {len(corners)}")
        return image
    
    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Convert corners to numpy array
    pts = np.array(corners, dtype=np.int32)
    
    # Fill the polygon
    cv2.fillPoly(mask, [pts], 255)
    
    # Apply mask
    if len(image.shape) == 3:
        masked_image = image.copy()
        for i in range(image.shape[2]):
            masked_image[:, :, i] = np.where(mask == 255, image[:, :, i], 0)
    else:
        masked_image = np.where(mask == 255, image, 0)
    
    return masked_image

def calculate_pixel_statistics(raw_files: List[str], dark_data: Optional[np.ndarray], 
                              width: int, height: int, data_type: str, roi_corners: Optional[List[Tuple[int, int]]] = None) -> Dict:
    """
    Calculate pixel-wise statistics across multiple RAW files
    
    Args:
        raw_files: List of RAW file paths
        dark_data: Dark reference data
        width: Image width
        height: Image height
        data_type: Data type
        roi_corners: Optional ROI corners for masking
        
    Returns:
        Dictionary containing pixel statistics
    """
    print(f"\n=== Calculating Pixel Statistics Across {len(raw_files)} Images ===")
    if roi_corners:
        print(f"  Using ROI with corners: {roi_corners}")
    else:
        print(f"  Processing entire image")
    
    try:
        # Initialize accumulators
        pixel_sum = np.zeros((height, width), dtype=np.float64)
        pixel_sum_sq = np.zeros((height, width), dtype=np.float64)
        valid_pixels = np.zeros((height, width), dtype=np.int32)
        
        # Process each RAW file
        for i, raw_file in enumerate(raw_files):
            print(f"  Processing file {i+1}/{len(raw_files)}: {Path(raw_file).name}")
            
            try:
                # Read RAW data
                raw_data = read_raw_image(raw_file, width, height, data_type)
                
                # Apply dark current subtraction if enabled
                if DARK_SUBTRACTION_ENABLED and dark_data is not None:
                    corrected_data = apply_dark_subtraction(raw_data, dark_data)
                else:
                    corrected_data = raw_data.copy()
                
                # Apply ROI mask if provided
                if roi_corners:
                    corrected_data = apply_roi_mask(corrected_data, roi_corners)
                
                # Convert to float for accumulation
                corrected_float = corrected_data.astype(np.float64)
                
                # Accumulate statistics
                pixel_sum += corrected_float
                pixel_sum_sq += corrected_float ** 2
                valid_pixels += 1
                
            except Exception as e:
                print(f"    Error processing {raw_file}: {e}")
                continue
        
        # Calculate final statistics
        print(f"  Calculating final statistics...")
        
        # Avoid division by zero
        valid_pixels = np.maximum(valid_pixels, 1)
        
        # Calculate mean and variance for each pixel
        pixel_means = pixel_sum / valid_pixels
        pixel_vars = (pixel_sum_sq / valid_pixels) - (pixel_means ** 2)
        
        # Ensure variance is non-negative (due to floating point precision)
        pixel_vars = np.maximum(pixel_vars, 0.0)
        
        # If ROI is used, only include pixels within ROI for statistics
        if roi_corners:
            # Create ROI mask
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            pts = np.array(roi_corners, dtype=np.int32)
            cv2.fillPoly(roi_mask, [pts], 255)
            
            # Only include ROI pixels
            roi_pixels = roi_mask == 255
            means_flat = pixel_means[roi_pixels]
            vars_flat = pixel_vars[roi_pixels]
            
            # Calculate ROI area
            roi_area = np.sum(roi_pixels)
            print(f"    ROI area: {roi_area:,} pixels")
        else:
            # Use all pixels
            means_flat = pixel_means.flatten()
            vars_flat = pixel_vars.flatten()
            roi_area = means_flat.size
        
        # Calculate overall statistics
        overall_mean = np.mean(means_flat)
        overall_var = np.mean(vars_flat)
        overall_std = np.sqrt(overall_var)
        
        stats = {
            'pixel_means': pixel_means,
            'pixel_vars': pixel_vars,
            'means_flat': means_flat,
            'vars_flat': vars_flat,
            'overall_mean': float(overall_mean),
            'overall_var': float(overall_var),
            'overall_std': float(overall_std),
            'total_pixels': int(roi_area),
            'valid_files': int(np.max(valid_pixels)),
            'image_shape': (height, width),
            'roi_corners': roi_corners,
            'roi_enabled': roi_corners is not None
        }
        
        print(f"  Statistics calculated:")
        print(f"    Total pixels: {stats['total_pixels']:,}")
        print(f"    Valid files: {stats['valid_files']}")
        print(f"    Overall mean: {stats['overall_mean']:.2f}")
        print(f"    Overall variance: {stats['overall_var']:.2f}")
        print(f"    Overall std: {stats['overall_std']:.2f}")
        if roi_corners:
            print(f"    ROI enabled: Yes")
        else:
            print(f"    ROI enabled: No")
        
        return stats
        
    except Exception as e:
        print(f"  Error calculating pixel statistics: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def create_variance_mean_plot(pixel_stats: Dict, output_dir: Path, base_filename: str):
    """
    Create variance vs mean scatter plot
    
    Args:
        pixel_stats: Pixel statistics dictionary
        output_dir: Output directory
        base_filename: Base filename
    """
    print("Creating variance vs mean plot...")
    
    try:
        if 'error' in pixel_stats:
            print(f"  Error in pixel stats: {pixel_stats['error']}")
            return
        
        means_flat = pixel_stats['means_flat']
        vars_flat = pixel_stats['vars_flat']
        
        # Create main scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pixel Variance vs Mean Analysis', fontsize=16)
        
        # Plot 1: Scatter plot of variance vs mean
        axes[0, 0].scatter(means_flat, vars_flat, alpha=0.1, s=0.1, color='blue')
        axes[0, 0].set_xlabel('Pixel Mean')
        axes[0, 0].set_ylabel('Pixel Variance')
        axes[0, 0].set_title('Variance vs Mean (All Pixels)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(means_flat, vars_flat, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(means_flat, p(means_flat), "r--", alpha=0.8, linewidth=2, 
                       label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
        axes[0, 0].legend()
        
        # Set adaptive axis limits with some padding
        mean_range = np.max(means_flat) - np.min(means_flat)
        var_range = np.max(vars_flat) - np.min(vars_flat)
        axes[0, 0].set_xlim(np.min(means_flat) - 0.05 * mean_range, 
                           np.max(means_flat) + 0.05 * mean_range)
        axes[0, 0].set_ylim(np.min(vars_flat) - 0.05 * var_range, 
                           np.max(vars_flat) + 0.05 * var_range)
        
        # Plot 2: Binned scatter plot (reduce data density)
        print("  Creating binned scatter plot...")
        n_bins = 100
        mean_bins = np.linspace(np.min(means_flat), np.max(means_flat), n_bins)
        var_bins = np.linspace(np.min(vars_flat), np.max(vars_flat), n_bins)
        
        # Create 2D histogram
        hist_2d, xedges, yedges = np.histogram2d(means_flat, vars_flat, bins=[mean_bins, var_bins])
        
        # Plot 2D histogram
        im = axes[0, 1].imshow(hist_2d.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                              origin='lower', cmap='viridis', aspect='auto')
        axes[0, 1].set_xlabel('Pixel Mean')
        axes[0, 1].set_ylabel('Pixel Variance')
        axes[0, 1].set_title('Variance vs Mean (Binned)')
        plt.colorbar(im, ax=axes[0, 1], label='Pixel Count')
        
        # Set adaptive axis limits for binned plot
        axes[0, 1].set_xlim(np.min(means_flat), np.max(means_flat))
        axes[0, 1].set_ylim(np.min(vars_flat), np.max(vars_flat))
        
        # Plot 3: Mean distribution
        axes[1, 0].hist(means_flat, bins=256, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Pixel Mean')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Pixel Means')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Set adaptive axis limits for mean distribution
        mean_range = np.max(means_flat) - np.min(means_flat)
        axes[1, 0].set_xlim(np.min(means_flat) - 0.02 * mean_range, 
                           np.max(means_flat) + 0.02 * mean_range)
        
        # Add statistics to mean distribution
        mean_stats_text = f"Mean: {np.mean(means_flat):.2f}\n"
        mean_stats_text += f"Std: {np.std(means_flat):.2f}\n"
        mean_stats_text += f"Min: {np.min(means_flat):.2f}\n"
        mean_stats_text += f"Max: {np.max(means_flat):.2f}"
        axes[1, 0].text(0.02, 0.98, mean_stats_text, transform=axes[1, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Plot 4: Variance distribution
        axes[1, 1].hist(vars_flat, bins=256, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_xlabel('Pixel Variance')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Pixel Variances')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Set adaptive axis limits for variance distribution
        var_range = np.max(vars_flat) - np.min(vars_flat)
        axes[1, 1].set_xlim(np.min(vars_flat) - 0.02 * var_range, 
                           np.max(vars_flat) + 0.02 * var_range)
        
        # Add statistics to variance distribution
        var_stats_text = f"Mean: {np.mean(vars_flat):.2f}\n"
        var_stats_text += f"Std: {np.std(vars_flat):.2f}\n"
        var_stats_text += f"Min: {np.min(vars_flat):.2f}\n"
        var_stats_text += f"Max: {np.max(vars_flat):.2f}"
        axes[1, 1].text(0.02, 0.98, var_stats_text, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plot_path = output_dir / f"{base_filename}_variance_mean_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Variance vs mean plot saved: {plot_path}")
        
        if GENERATE_PLOTS:
            plt.show()
        else:
            plt.close()
        
        # Create detailed scatter plot with different sampling
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Sample data for better visualization (if too many points)
        if len(means_flat) > 100000:
            sample_indices = np.random.choice(len(means_flat), 100000, replace=False)
            means_sample = means_flat[sample_indices]
            vars_sample = vars_flat[sample_indices]
            print(f"  Sampling {len(means_sample):,} points for detailed plot")
        else:
            means_sample = means_flat
            vars_sample = vars_flat
        
        # Create scatter plot with color mapping based on density
        scatter = ax.scatter(means_sample, vars_sample, alpha=0.3, s=1, c=vars_sample, cmap='plasma')
        ax.set_xlabel('Pixel Mean', fontsize=12)
        ax.set_ylabel('Pixel Variance', fontsize=12)
        ax.set_title('Detailed Variance vs Mean Scatter Plot', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Set adaptive axis limits for detailed scatter plot
        mean_range = np.max(means_sample) - np.min(means_sample)
        var_range = np.max(vars_sample) - np.min(vars_sample)
        ax.set_xlim(np.min(means_sample) - 0.05 * mean_range, 
                   np.max(means_sample) + 0.05 * mean_range)
        ax.set_ylim(np.min(vars_sample) - 0.05 * var_range, 
                   np.max(vars_sample) + 0.05 * var_range)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pixel Variance', fontsize=12)
        
        # Add trend line
        z = np.polyfit(means_sample, vars_sample, 1)
        p = np.poly1d(z)
        ax.plot(means_sample, p(means_sample), "r-", alpha=0.8, linewidth=3, 
               label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plot_path = output_dir / f"{base_filename}_detailed_variance_mean.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Detailed variance vs mean plot saved: {plot_path}")
        
        if GENERATE_PLOTS:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"  Error creating variance vs mean plot: {e}")
        import traceback
        traceback.print_exc()

def save_pixel_statistics(pixel_stats: Dict, output_dir: Path, base_filename: str):
    """
    Save pixel statistics to JSON file
    
    Args:
        pixel_stats: Pixel statistics dictionary
        output_dir: Output directory
        base_filename: Base filename
    """
    if not SAVE_DATA:
        return
    
    print("Saving pixel statistics...")
    
    try:
        # Create summary statistics (don't save full arrays to keep file size reasonable)
        summary_stats = {
            'overall_mean': pixel_stats['overall_mean'],
            'overall_var': pixel_stats['overall_var'],
            'overall_std': pixel_stats['overall_std'],
            'total_pixels': pixel_stats['total_pixels'],
            'valid_files': pixel_stats['valid_files'],
            'image_shape': pixel_stats['image_shape'],
            'mean_statistics': {
                'mean': float(np.mean(pixel_stats['means_flat'])),
                'std': float(np.std(pixel_stats['means_flat'])),
                'min': float(np.min(pixel_stats['means_flat'])),
                'max': float(np.max(pixel_stats['means_flat'])),
                'median': float(np.median(pixel_stats['means_flat']))
            },
            'variance_statistics': {
                'mean': float(np.mean(pixel_stats['vars_flat'])),
                'std': float(np.std(pixel_stats['vars_flat'])),
                'min': float(np.min(pixel_stats['vars_flat'])),
                'max': float(np.max(pixel_stats['vars_flat'])),
                'median': float(np.median(pixel_stats['vars_flat']))
            }
        }
        
        # Create complete data structure
        stats_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_type': 'pixel_statistics_across_images',
            'image_dimensions': [IMAGE_WIDTH, IMAGE_HEIGHT],
            'data_type': DATA_TYPE,
            'processing_settings': {
                'dark_subtraction_enabled': DARK_SUBTRACTION_ENABLED,
                'histogram_bins': HISTOGRAM_BINS,
                'clip_negative_values': CLIP_NEGATIVE_VALUES
            },
            'summary_statistics': summary_stats
        }
        
        # Save to JSON file
        data_path = output_dir / f"{base_filename}_pixel_statistics.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Pixel statistics saved: {data_path}")
        
    except Exception as e:
        print(f"  Error saving pixel statistics: {e}")
        import traceback
        traceback.print_exc()

def process_gray_chart_image_with_roi(raw_file: str, dark_data: Optional[np.ndarray], 
                                     width: int, height: int, data_type: str, 
                                     roi_corners: List[Tuple[int, int]]) -> Dict:
    """
    Process gray chart image for whole image noise analysis with ROI
    
    Args:
        raw_file: Path to RAW file
        dark_data: Dark reference data
        width: Image width
        height: Image height
        data_type: Data type
        roi_corners: ROI corner coordinates
        
    Returns:
        Processing result dictionary
    """
    print(f"\n=== Processing Gray Chart Image with ROI ===")
    print(f"File: {raw_file}")
    print(f"ROI corners: {roi_corners}")
    
    try:
        # 1. Read RAW data
        print(f"  1. Reading RAW data...")
        raw_data = read_raw_image(raw_file, width, height, data_type)
        print(f"  1. RAW data loaded: {raw_data.shape}, dtype: {raw_data.dtype}")
        print(f"  1. RAW data range: {np.min(raw_data)} - {np.max(raw_data)}")
        
        # 2. Dark current subtraction
        if DARK_SUBTRACTION_ENABLED and dark_data is not None:
            print(f"  2. Applying dark current subtraction...")
            corrected_data = apply_dark_subtraction(raw_data, dark_data)
        else:
            print(f"  2. Dark current subtraction skipped")
            corrected_data = raw_data.copy()
        
        # 3. Apply ROI mask
        print(f"  3. Applying ROI mask...")
        roi_masked_data = apply_roi_mask(corrected_data, roi_corners)
        
        # Calculate ROI area
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(roi_corners, dtype=np.int32)
        cv2.fillPoly(roi_mask, [pts], 255)
        roi_area = np.sum(roi_mask == 255)
        print(f"  3. ROI area: {roi_area:,} pixels")
        
        # 4. Analyze ROI noise
        print(f"  4. Analyzing ROI noise...")
        image_stats = analyze_whole_image_noise(roi_masked_data)
        
        if 'error' in image_stats:
            print(f"  4. Error: {image_stats['error']}")
            return {
                'filename': raw_file,
                'processing_success': False,
                'error': image_stats['error']
            }
        
        # Update stats with ROI information
        image_stats['roi_corners'] = roi_corners
        image_stats['roi_enabled'] = True
        image_stats['roi_area'] = int(roi_area)
        image_stats['total_pixels'] = int(roi_area)
        
        print(f"    Mean: {image_stats['mean']:.1f}")
        print(f"    Std: {image_stats['std']:.1f}")
        print(f"    SNR: {image_stats['snr']:.1f}")
        print(f"    CV: {image_stats['cv']:.4f}")
        print(f"    ROI area: {roi_area:,} pixels")
        
        # 5. Create output directory
        output_dir = Path(OUTPUT_DIRECTORY) if OUTPUT_DIRECTORY else Path("noise_cali_output")
        output_dir.mkdir(exist_ok=True)
        
        # 6. RGGB channel analysis for ROI
        print(f"  5. RGGB channel analysis for ROI...")
        # Convert ROI corners to rectangle format for RGGB analysis
        # Find bounding rectangle of ROI
        roi_corners_array = np.array(roi_corners)
        x_min = int(np.min(roi_corners_array[:, 0]))
        y_min = int(np.min(roi_corners_array[:, 1]))
        x_max = int(np.max(roi_corners_array[:, 0]))
        y_max = int(np.max(roi_corners_array[:, 1]))
        roi_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Analyze RGGB channel statistics
        rggb_roi_stats = analyze_rggb_roi_statistics(corrected_data, roi_rect)
        
        if 'error' not in rggb_roi_stats:
            print(f"    RGGB channel analysis completed")
            
            # Print simplified RGGB channel statistics
            print(f"    === RGGB Channel Mean and Variance ===")
            for channel_name, stats in rggb_roi_stats['channels'].items():
                print(f"    {channel_name} Channel: Mean={stats['mean']:.1f}, Variance={stats['variance']:.1f}")
            
            # Save RGGB ROI data
            if SAVE_DATA:
                base_filename = f"{Path(raw_file).stem}_roi"
                save_rggb_roi_statistics(rggb_roi_stats, output_dir, base_filename)
        else:
            print(f"    RGGB channel analysis failed: {rggb_roi_stats['error']}")
        
        # 7. Generate plots
        if GENERATE_PLOTS or SAVE_PLOTS:
            print(f"  6. Generating plots...")
            create_whole_image_histogram(roi_masked_data, image_stats, output_dir)
        
        # 8. Save data
        base_filename = f"{Path(raw_file).stem}_roi"
        save_whole_image_data(image_stats, output_dir, base_filename)
        
        # 8. Print summary
        print(f"\n=== ROI Noise Analysis Summary ===")
        print(f"Image size: {image_stats['image_size']}")
        print(f"ROI area: {image_stats['roi_area']:,} pixels")
        print(f"Mean: {image_stats['mean']:.1f}")
        print(f"Standard deviation: {image_stats['std']:.1f}")
        print(f"Min: {image_stats['min']:.1f}")
        print(f"Max: {image_stats['max']:.1f}")
        print(f"Median: {image_stats['median']:.1f}")
        print(f"Signal-to-Noise Ratio: {image_stats['snr']:.2f}")
        print(f"Coefficient of Variation: {image_stats['cv']:.4f}")
        print(f"ROI corners: {roi_corners}")
        
        result = {
            'filename': raw_file,
            'original_data': raw_data,
            'corrected_data': corrected_data,
            'roi_masked_data': roi_masked_data,
            'image_stats': image_stats,
            'roi_corners': roi_corners,
            'processing_success': True,
            'output_directory': str(output_dir)
        }
        
        return result
        
    except Exception as e:
        print(f"  Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return {
            'filename': raw_file,
            'processing_success': False,
            'error': str(e)
        }

def process_gray_chart_image(raw_file: str, dark_data: Optional[np.ndarray], 
                           width: int, height: int, data_type: str) -> Dict:
    """
    Process gray chart image for whole image noise analysis
    
    Args:
        raw_file: Path to RAW file
        dark_data: Dark reference data
        width: Image width
        height: Image height
        data_type: Data type
        
    Returns:
        Processing result dictionary
    """
    print(f"\n=== Processing Gray Chart Image (Whole Image Analysis) ===")
    print(f"File: {raw_file}")
    
    try:
        # 1. Read RAW data
        print(f"  1. Reading RAW data...")
        raw_data = read_raw_image(raw_file, width, height, data_type)
        print(f"  1. RAW data loaded: {raw_data.shape}, dtype: {raw_data.dtype}")
        print(f"  1. RAW data range: {np.min(raw_data)} - {np.max(raw_data)}")
        
        # 2. Dark current subtraction
        if DARK_SUBTRACTION_ENABLED and dark_data is not None:
            print(f"  2. Applying dark current subtraction...")
            corrected_data = apply_dark_subtraction(raw_data, dark_data)
        else:
            print(f"  2. Dark current subtraction skipped")
            corrected_data = raw_data.copy()
        
        # 3. Analyze whole image noise
        print(f"  3. Analyzing whole image noise...")
        image_stats = analyze_whole_image_noise(corrected_data)
        
        if 'error' in image_stats:
            print(f"  3. Error: {image_stats['error']}")
            return {
                'filename': raw_file,
                'processing_success': False,
                'error': image_stats['error']
            }
        
        print(f"    Mean: {image_stats['mean']:.1f}")
        print(f"    Std: {image_stats['std']:.1f}")
        print(f"    SNR: {image_stats['snr']:.1f}")
        print(f"    CV: {image_stats['cv']:.4f}")
        
        # 4. Create output directory
        output_dir = Path(OUTPUT_DIRECTORY) if OUTPUT_DIRECTORY else Path("noise_cali_output")
        output_dir.mkdir(exist_ok=True)
        
        # 5. Generate plots
        if GENERATE_PLOTS or SAVE_PLOTS:
            print(f"  4. Generating plots...")
            create_whole_image_histogram(corrected_data, image_stats, output_dir)
        
        # 6. RGGB channel analysis with ROI selection
        print(f"  5. RGGB channel analysis with ROI selection...")
        
        # First separate RGGB channels from the entire image
        print(f"    Separating RGGB channels from entire image...")
        rggb_channels = separate_rggb_channels(corrected_data)
        print(f"    Separated {len(rggb_channels)} channels: {list(rggb_channels.keys())}")
        
        # Let user select ROI on one of the channels for display
        print(f"    Please select ROI region for RGGB analysis...")
        roi_rect = select_roi_rectangle(corrected_data, "Select ROI for RGGB Analysis")
        
        if roi_rect:
            print(f"    ROI selected: {roi_rect}")
            x, y, w, h = roi_rect
            
            # Extract ROI from each RGGB channel
            print(f"    Extracting ROI from RGGB channels...")
            roi_rggb_data = {}
            
            for channel_name, channel_data in rggb_channels.items():
                # Calculate ROI coordinates for this channel
                # Each channel is half the size of original image
                roi_x = x // 2
                roi_y = y // 2
                roi_w = w // 2
                roi_h = h // 2
                
                # Ensure ROI is within channel bounds
                roi_x = max(0, min(roi_x, channel_data.shape[1] - 1))
                roi_y = max(0, min(roi_y, channel_data.shape[0] - 1))
                roi_w = min(roi_w, channel_data.shape[1] - roi_x)
                roi_h = min(roi_h, channel_data.shape[0] - roi_y)
                
                # Extract ROI from channel
                roi_channel_data = channel_data[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                roi_rggb_data[channel_name] = roi_channel_data
                
                print(f"    {channel_name} channel ROI: shape={roi_channel_data.shape}, pixels={roi_channel_data.size}")
            
            # Calculate mean and variance for each channel
            print(f"    === RGGB Channel Mean and Variance ===")
            channel_stats = {}
            
            for channel_name, roi_data in roi_rggb_data.items():
                # Convert to float for analysis
                roi_float = roi_data.astype(np.float32)
                
                # Calculate statistics
                mean_val = np.mean(roi_float)
                var_val = np.var(roi_float)
                
                channel_stats[channel_name] = {
                    'mean': float(mean_val),
                    'variance': float(var_val)
                }
                
                print(f"    {channel_name} Channel: Mean={mean_val:.1f}, Variance={var_val:.1f}")
            
            # Save RGGB ROI data
            if SAVE_DATA:
                base_filename = Path(raw_file).stem
                rggb_roi_stats = {
                    'roi_rectangle': roi_rect,
                    'channels': channel_stats
                }
                save_rggb_roi_statistics(rggb_roi_stats, output_dir, base_filename)
        else:
            print(f"    ROI selection cancelled")
        
        # 7. Save data
        base_filename = Path(raw_file).stem
        save_whole_image_data(image_stats, output_dir, base_filename)
        
        # 7. Print summary
        print(f"\n=== Whole Image Noise Analysis Summary ===")
        print(f"Image size: {image_stats['image_size']}")
        print(f"Total pixels: {image_stats['total_pixels']:,}")
        print(f"Mean: {image_stats['mean']:.1f}")
        print(f"Standard deviation: {image_stats['std']:.1f}")
        print(f"Min: {image_stats['min']:.1f}")
        print(f"Max: {image_stats['max']:.1f}")
        print(f"Median: {image_stats['median']:.1f}")
        print(f"Signal-to-Noise Ratio: {image_stats['snr']:.2f}")
        print(f"Coefficient of Variation: {image_stats['cv']:.4f}")
        
        result = {
            'filename': raw_file,
            'original_data': raw_data,
            'corrected_data': corrected_data,
            'image_stats': image_stats,
            'processing_success': True,
            'output_directory': str(output_dir)
        }
        
        return result
        
    except Exception as e:
        print(f"  Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return {
            'filename': raw_file,
            'processing_success': False,
            'error': str(e)
        }

def main():
    """Main function"""
    print("=== Noise Calibration Tool (Pixel Statistics Analysis) ===")
    print(f"Input path: {INPUT_PATH}")
    print(f"Dark reference: {DARK_RAW_PATH}")
    print(f"Dimensions: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"Data type: {DATA_TYPE}")
    print(f"Dark subtraction: {DARK_SUBTRACTION_ENABLED}")
    print(f"Analysis type: Pixel statistics across multiple images")
    print(f"Generate plots: {GENERATE_PLOTS}")
    print(f"Save plots: {SAVE_PLOTS}")
    print(f"Save data: {SAVE_DATA}")
    print()
    
    # Check input paths
    input_path = Path(INPUT_PATH)
    dark_path = Path(DARK_RAW_PATH)
    
    if not input_path.exists():
        print(f"Error: Input path not found: {INPUT_PATH}")
        return 1
    
    if DARK_SUBTRACTION_ENABLED and not dark_path.exists():
        print(f"Warning: Dark reference file not found: {DARK_RAW_PATH}")
        print("Continuing without dark subtraction...")
        dark_data = None
    else:
        # Load dark reference
        dark_data = load_dark_reference(DARK_RAW_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
    
    # Find RAW files
    if input_path.is_file():
        raw_files = [str(input_path)]
        print(f"Processing single file: {input_path.name}")
    else:
        raw_files = list(input_path.glob("*.raw"))
        raw_files = [str(f) for f in raw_files]
        print(f"Processing folder: {input_path}")
    
    if not raw_files:
        print("No RAW files found!")
        return 1
    
    print(f"\nFound {len(raw_files)} RAW files")
    
    # Check if we have multiple files for pixel statistics analysis
    if len(raw_files) > 1:
        print(f"\n=== Multi-Image Pixel Statistics Analysis ===")
        print(f"Processing {len(raw_files)} images for pixel-wise statistics...")
        
        # ROI selection for multi-image analysis
        roi_corners = None
        print(f"\n=== ROI Selection ===")
        print(f"Would you like to select a ROI region for analysis?")
        print(f"1. Yes - Select ROI region")
        print(f"2. No - Process entire image")
        
        # For automated processing, we'll use the first image for ROI selection
        try:
            # Read first image for ROI selection
            first_raw_data = read_raw_image(raw_files[0], IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
            if DARK_SUBTRACTION_ENABLED and dark_data is not None:
                first_corrected_data = apply_dark_subtraction(first_raw_data, dark_data)
            else:
                first_corrected_data = first_raw_data.copy()
            
            # Interactive ROI selection
            roi_corners = select_roi_corners(first_corrected_data, "Select ROI for Pixel Statistics")
            
            if roi_corners:
                print(f"ROI selected with corners: {roi_corners}")
            else:
                print(f"ROI selection cancelled, processing entire image")
                
        except Exception as e:
            print(f"Error during ROI selection: {e}")
            print(f"Continuing with entire image processing...")
            roi_corners = None
        
        # Calculate pixel statistics across all images
        pixel_stats = calculate_pixel_statistics(raw_files, dark_data, 
                                               IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE, roi_corners)
        
        if 'error' in pixel_stats:
            print(f"Error in pixel statistics calculation: {pixel_stats['error']}")
            return 1
        
        # Create output directory
        output_dir = Path(OUTPUT_DIRECTORY) if OUTPUT_DIRECTORY else Path("noise_cali_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate variance vs mean plots
        if GENERATE_PLOTS or SAVE_PLOTS:
            print(f"\n=== Generating Variance vs Mean Plots ===")
            roi_suffix = "_roi" if roi_corners else "_full"
            base_filename = f"pixel_stats_{len(raw_files)}_images{roi_suffix}"
            create_variance_mean_plot(pixel_stats, output_dir, base_filename)
        

        # Save pixel statistics
        if SAVE_DATA:
            print(f"\n=== Saving Pixel Statistics ===")
            roi_suffix = "_roi" if roi_corners else "_full"
            base_filename = f"pixel_stats_{len(raw_files)}_images{roi_suffix}"
            save_pixel_statistics(pixel_stats, output_dir, base_filename)
        
        # Print final summary
        print(f"\n=== Pixel Statistics Analysis Complete ===")
        print(f"Total images processed: {len(raw_files)}")
        print(f"Total pixels analyzed: {pixel_stats['total_pixels']:,}")
        print(f"Image dimensions: {pixel_stats['image_shape']}")
        print(f"Overall mean: {pixel_stats['overall_mean']:.2f}")
        print(f"Overall variance: {pixel_stats['overall_var']:.2f}")
        print(f"Overall std: {pixel_stats['overall_std']:.2f}")
        if roi_corners:
            print(f"ROI enabled: Yes")
            print(f"ROI corners: {roi_corners}")
        else:
            print(f"ROI enabled: No")
        print(f"Results saved to: {output_dir}")
        
    else:
        print(f"\n=== Single Image Analysis ===")
        print(f"Processing single image for whole image analysis...")
        
        # ROI selection for single image analysis
        roi_corners = None
        print(f"\n=== ROI Selection ===")
        print(f"Would you like to select a ROI region for analysis?")
        print(f"1. Yes - Select ROI region")
        print(f"2. No - Process entire image")
        
        try:
            # Read image for ROI selection
            raw_data = read_raw_image(raw_files[0], IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
            if DARK_SUBTRACTION_ENABLED and dark_data is not None:
                corrected_data = apply_dark_subtraction(raw_data, dark_data)
            else:
                corrected_data = raw_data.copy()


            #separate rggb channels 
            rggb_channels = separate_rggb_channels(corrected_data)
            print(f"Separated {len(rggb_channels)} channels: {list(rggb_channels.keys())}")
            for channel_name, channel_data in rggb_channels.items():
                print(f"    {channel_name} channel: shape={channel_data.shape}, pixels={channel_data.size}")

            # Interactive ROI selection
            roi_corners = select_roi_corners(rggb_channels['R'], "Select ROI for Single Image Analysis")
            

            if roi_corners:
                print(f"ROI selected with corners: {roi_corners}")
                
                # Extract ROI pixels from each RGGB channel
                print(f"\n=== Extracting ROI Pixels from RGGB Channels ===")
                roi_rggb_data = {}
                
                for channel_name, channel_data in rggb_channels.items():
                    print(f"Processing {channel_name} channel...")
                    
                    # Create mask for ROI region
                    roi_mask = np.zeros(channel_data.shape, dtype=np.uint8)
                    pts = np.array(roi_corners, dtype=np.int32)
                    cv2.fillPoly(roi_mask, [pts], 255)
                    
                    # Extract pixels within ROI
                    roi_pixels = channel_data[roi_mask == 255]
                    roi_rggb_data[channel_name] = roi_pixels
                    
                    print(f"    {channel_name} channel ROI pixels: {len(roi_pixels)} pixels")
                    print(f"    {channel_name} channel ROI range: {np.min(roi_pixels)} - {np.max(roi_pixels)}")
                
                # Calculate mean and variance for each channel
                print(f"\n=== RGGB Channel Mean and Variance ===")
                channel_stats = {}
                
                for channel_name, roi_pixels in roi_rggb_data.items():
                    # Convert to float for analysis
                    roi_float = roi_pixels.astype(np.float32)
                    
                    # Calculate statistics
                    mean_val = np.mean(roi_float)
                    var_val = np.var(roi_float)
                    
                    channel_stats[channel_name] = {
                        'mean': float(mean_val),
                        'variance': float(var_val),
                        'pixel_count': len(roi_pixels)
                    }
                    
                    print(f"    {channel_name} Channel: Mean={mean_val:.1f}, Variance={var_val:.1f}, Pixels={len(roi_pixels)}")
                
                # Save RGGB ROI data
                if SAVE_DATA:
                    output_dir = Path(OUTPUT_DIRECTORY) if OUTPUT_DIRECTORY else Path("noise_cali_output_average")
                    output_dir.mkdir(exist_ok=True)
                    
                    base_filename = Path(raw_files[0]).stem
                    rggb_roi_stats = {
                        'roi_corners': roi_corners,
                        'channels': channel_stats
                    }
                    save_rggb_roi_statistics(rggb_roi_stats, output_dir, base_filename)
                    
                    # Save to Excel file
                    save_rggb_to_excel(channel_stats, roi_corners, base_filename, output_dir)
                    
                    print(f"RGGB ROI statistics saved to: {output_dir}")
                
                # 批量处理父目录下的所有RAW图
                print(f"\n=== 批量处理父目录下的所有RAW图 ===")
                parent_dir = Path(raw_files[0]).parent
                all_raw_files = list(parent_dir.glob("*.raw"))
                all_raw_files = [str(f) for f in all_raw_files]
                all_raw_files.sort()
                
                print(f"找到 {len(all_raw_files)} 个RAW文件:")
                for i, file_path in enumerate(all_raw_files):
                    print(f"  {i+1}. {Path(file_path).name}")
                
                if len(all_raw_files) > 1:
                    print(f"\n开始批量处理 {len(all_raw_files)} 个文件...")
                    
                    # 初始化累积器 - 存储所有像素值
                    all_channel_pixels = {'R': [], 'G1': [], 'G2': [], 'B': []}
                    successful_files = 0
                    failed_files = 0
                    
                    # 处理每个RAW文件
                    for i, raw_file in enumerate(all_raw_files):
                        print(f"\n处理文件 {i+1}/{len(all_raw_files)}: {Path(raw_file).name}")
                        
                        try:
                            # 读取RAW数据
                            raw_data = read_raw_image(raw_file, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
                            
                            # 应用暗电流校正
                            if DARK_SUBTRACTION_ENABLED and dark_data is not None:
                                corrected_data = apply_dark_subtraction(raw_data, dark_data)
                            else:
                                corrected_data = raw_data.copy()
                            
                            # 提取RGGB通道
                            rggb_channels = separate_rggb_channels(corrected_data)
                            
                            if not rggb_channels:
                                print(f"  错误: 无法提取RGGB通道")
                                failed_files += 1
                                continue
                            
                            # 提取ROI区域的像素
                            roi_rggb_data = {}
                            for channel_name, channel_data in rggb_channels.items():
                                # 创建ROI掩码
                                roi_mask = np.zeros(channel_data.shape, dtype=np.uint8)
                                pts = np.array(roi_corners, dtype=np.int32)
                                cv2.fillPoly(roi_mask, [pts], 255)
                                
                                # 提取ROI内的像素
                                roi_pixels = channel_data[roi_mask == 255]
                                roi_rggb_data[channel_name] = roi_pixels
                                
                                # 累积所有像素值
                                if len(roi_pixels) > 0:
                                    all_channel_pixels[channel_name].extend(roi_pixels.tolist())
                                    print(f"    {channel_name}: 提取了 {len(roi_pixels)} 个像素")
                                else:
                                    print(f"    {channel_name}: 无有效像素")
                            
                            successful_files += 1
                            
                        except Exception as e:
                            print(f"  错误处理文件 {raw_file}: {e}")
                            failed_files += 1
                            continue
                    
                    # 计算总体统计量
                    print(f"\n=== 批量处理结果统计 ===")
                    print(f"成功处理: {successful_files} 个文件")
                    print(f"处理失败: {failed_files} 个文件")
                    
                    if successful_files > 0:
                        print(f"\n=== 总体统计量计算 ===")
                        channel_stats = {}
                        
                        for channel_name in ['R', 'G1', 'G2', 'B']:
                            if all_channel_pixels[channel_name]:
                                # 转换为numpy数组
                                pixels_array = np.array(all_channel_pixels[channel_name], dtype=np.float32)
                                
                                # 计算统计量
                                mean_val = np.mean(pixels_array)
                                var_val = np.var(pixels_array)
                                pixel_count = len(pixels_array)
                                
                                channel_stats[channel_name] = {
                                    'mean': float(mean_val),
                                    'variance': float(var_val),
                                    'pixel_count': pixel_count
                                }
                                
                                print(f"  {channel_name}: Mean={mean_val:.1f}, Variance={var_val:.1f}, Pixels={pixel_count}")
                            else:
                                channel_stats[channel_name] = {
                                    'mean': 0.0,
                                    'variance': 0.0,
                                    'pixel_count': 0
                                }
                                print(f"  {channel_name}: 无有效像素")
                        
                        # 保存批量处理结果
                        if SAVE_DATA:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            batch_filename = f"batch_roi_stats_{successful_files}_files_{timestamp}"
                            
                            # 使用save_rggb_to_excel函数保存到Excel
                            save_rggb_to_excel(channel_stats, roi_corners, batch_filename, output_dir)
                            
                            # 保存JSON结果
                            batch_json_data = {
                                'timestamp': timestamp,
                                'roi_corners': roi_corners,
                                'total_files_processed': successful_files,
                                'total_files_failed': failed_files,
                                'channel_statistics': channel_stats
                            }
                            
                            batch_json_path = output_dir / f"{batch_filename}.json"
                            with open(batch_json_path, 'w', encoding='utf-8') as f:
                                json.dump(batch_json_data, f, indent=2, ensure_ascii=False)
                            
                            print(f"批量处理结果已保存到: {batch_json_path}")
                            print(f"Excel文件已保存到: {output_dir}")
                else:
                    print("父目录下只有一个RAW文件，无需批量处理")
            else:
                print(f"ROI selection cancelled, processing entire image")
                

                
        except Exception as e:
            print(f"Error during ROI selection: {e}")
            print(f"Continuing with entire image processing...")
            roi_corners = None




    
    return 0

if __name__ == "__main__":
    exit(main())
