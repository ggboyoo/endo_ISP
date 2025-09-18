#!/usr/bin/env python3
"""
White Balance Calculator with Manual ROI Selection
白平衡计算程序 - 支持手动框选白板区域
读取RAW图，调用ISP流程，去马赛克后手动框选标准白板区域计算白平衡增益
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import argparse

# Import functions from lens_shading.py for correction parameters
try:
    from lens_shading import load_correction_parameters
    from ISP import demosaic_16bit
except ImportError:
    print("Error: lens_shading.py not found in the same directory!")
    print("Please ensure lens_shading.py is in the same directory as this script.")
    exit(1)

# 配置参数
CONFIG = {
    # 输入配置
    'INPUT_IMAGE_PATH': r"F:\ZJU\Picture\wb\25-09-01 160326.raw",  # 标准白板图路径
    'IMAGE_WIDTH': 3840,
    'IMAGE_HEIGHT': 2160,
    'DATA_TYPE': 'uint16',  # 12位数据存储在16位容器中（0-4095）
    
    # ISP配置
    'DARK_RAW_PATH': r"F:\ZJU\Picture\dark\g8\average_dark.raw",  # 暗电流参考图
    'DARK_SUBTRACTION_ENABLED': True,
    'LENS_SHADING_PARAMS_DIR': r"F:\ZJU\Picture\lens shading\new",
    'LENS_SHADING_ENABLED': True,
    'BAYER_PATTERN': 'rggb',
    
    # 白平衡配置
    'WHITE_BALANCE_METHOD': 'manual_roi',  # 手动ROI选择
    'ROI_SIZE': 100,  # 默认ROI大小（像素）
    
    # 输出配置
    'OUTPUT_DIRECTORY': None,  # None表示自动创建
    'SAVE_RESULTS': True,
    'SAVE_IMAGES': True,
    'SAVE_PARAMETERS': True,
}

# 全局变量用于ROI选择
roi_selected = False
roi_coords = None
roi_image = None


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


def subtract_dark_current(raw_data: np.ndarray, dark_data: np.ndarray, 
                         clip_negative: bool = True) -> np.ndarray:
    """Subtract dark current from RAW data"""
    print(f"Subtracting dark current...")
    print(f"  Original range: {np.min(raw_data)} - {np.max(raw_data)}")
    print(f"  Dark range: {np.min(dark_data)} - {np.max(dark_data)}")
    
    # Ensure same data type for subtraction
    if raw_data.dtype != dark_data.dtype:
        raw_data = raw_data.astype(dark_data.dtype)
    
    # Subtract dark current
    corrected_data = raw_data.astype(np.float64) - dark_data.astype(np.float64)
    
    if clip_negative:
        corrected_data = np.clip(corrected_data, 0, None)
        print(f"  Negative values clipped to 0")
    
    print(f"  Corrected range: {np.min(corrected_data)} - {np.max(corrected_data)}")
    
    return corrected_data


def separate_rggb_channels(raw_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Separate RGGB channels from RAW data"""
    height, width = raw_data.shape
    
    # Create channel arrays
    R_channel = np.zeros((height//2, width//2), dtype=raw_data.dtype)
    G1_channel = np.zeros((height//2, width//2), dtype=raw_data.dtype)
    G2_channel = np.zeros((height//2, width//2), dtype=raw_data.dtype)
    B_channel = np.zeros((height//2, width//2), dtype=raw_data.dtype)
    
    # Extract channels (RGGB pattern)
    R_channel = raw_data[0::2, 0::2]  # R at (0,0), (0,2), (2,0), (2,2)...
    G1_channel = raw_data[0::2, 1::2]  # G at (0,1), (0,3), (2,1), (2,3)...
    G2_channel = raw_data[1::2, 0::2]  # G at (1,0), (1,2), (3,0), (3,2)...
    B_channel = raw_data[1::2, 1::2]  # B at (1,1), (1,3), (3,1), (3,3)...
    
    channels = {
        'R': R_channel,
        'G1': G1_channel,
        'G2': G2_channel,
        'B': B_channel
    }
    
    return channels


def reconstruct_corrected_image(corrected_channels: Dict[str, np.ndarray], 
                              corrections: Dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct corrected image from corrected channels"""
    height, width = corrected_channels['R'].shape
    corrected_image = np.zeros((height*2, width*2), dtype=np.uint16)
    
    # Reconstruct RGGB pattern
    corrected_image[0::2, 0::2] = corrected_channels['R']
    corrected_image[0::2, 1::2] = corrected_channels['G1']
    corrected_image[1::2, 0::2] = corrected_channels['G2']
    corrected_image[1::2, 1::2] = corrected_channels['B']
    
    return corrected_image


def apply_demosaicing(raw_data: np.ndarray, bayer_pattern: str = 'rggb') -> np.ndarray:
    """Apply demosaicing to convert RAW to color image"""
    if bayer_pattern.lower() == 'rggb':
        # Use OpenCV demosaicing
        color_image = cv2.cvtColor(raw_data, cv2.COLOR_BayerRG2RGB)
    else:
        # Default to RGGB
        color_image = cv2.cvtColor(raw_data, cv2.COLOR_BayerRG2RGB)
    
    return color_image


def normalize_to_8bit(data: np.ndarray) -> np.ndarray:
    """Normalize data to 8-bit for display"""
    if len(data.shape) == 3:  # Color image
        # Normalize each channel separately
        normalized = np.zeros_like(data, dtype=np.uint8)
        for i in range(3):
            channel = data[:, :, i]
            channel_min = np.min(channel)
            channel_max = np.max(channel)
            if channel_max > channel_min:
                normalized[:, :, i] = ((channel - channel_min) / (channel_max - channel_min) * 255).astype(np.uint8)
    else:  # Grayscale image
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(data, dtype=np.uint8)
    
    return normalized


def load_dark_reference(dark_path: str, width: int, height: int, data_type: str) -> Optional[np.ndarray]:
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


def process_image_with_isp(raw_file: str, config: Dict) -> Dict:
    """
    使用ISP.py中的功能处理图像
    
    Args:
        raw_file: RAW文件路径
        config: 配置参数
        
    Returns:
        处理结果字典
    """
    print("=== ISP Processing using ISP.py ===")
    
    try:
        # 导入ISP模块
        from ISP import process_single_image, load_dark_reference
        
        # 加载暗电流参考数据
        dark_data = None
        if config['DARK_SUBTRACTION_ENABLED'] and Path(config['DARK_RAW_PATH']).exists():
            print("Loading dark reference data...")
            dark_data = load_dark_reference(config['DARK_RAW_PATH'], config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'], config['DATA_TYPE'])
            print(f"Dark reference loaded: {dark_data.shape}, range: {np.min(dark_data)}-{np.max(dark_data)}")
        else:
            print("No dark current correction applied")
        
        # 加载镜头阴影参数
        lens_shading_params = None
        if config['LENS_SHADING_ENABLED'] and Path(config['LENS_SHADING_PARAMS_DIR']).exists():
            print("Loading lens shading parameters...")
            from lens_shading import load_correction_parameters
            lens_shading_params = load_correction_parameters(config['LENS_SHADING_PARAMS_DIR'])
            print("Lens shading parameters loaded")
        else:
            print("No lens shading correction applied")
        
        # 重写简化的ISP流程，只计算到去马赛克
        print("Starting simplified ISP processing...")
        
        # 1. 读取RAW数据
        raw_data = read_raw_image(raw_file, config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'], config['DATA_TYPE'])
        print(f"  1. RAW loaded: {raw_data.shape}, range: {np.min(raw_data)}-{np.max(raw_data)}")
        
        # 2. 暗电流矫正
        if config['DARK_SUBTRACTION_ENABLED'] and dark_data is not None:
            dark_corrected = subtract_dark_current(raw_data, dark_data, clip_negative=True)
            print(f"  2. Dark correction applied")
        else:
            dark_corrected = raw_data.copy()
            print(f"  2. Dark correction skipped")
        
        # 3. Lens shading矫正
        if config['LENS_SHADING_ENABLED'] and lens_shading_params:
            print(f"  3. Applying lens shading correction...")
            # 分离RGGB通道
            channels = separate_rggb_channels(dark_corrected)
            
            # 对每个通道应用lens shading矫正
            corrected_channels = {}
            for channel_name in ['R', 'G1', 'G2', 'B']:
                if channel_name in lens_shading_params:
                    channel_data = channels[channel_name]
                    correction_matrix = lens_shading_params[channel_name]
                    
                    # 调整矫正矩阵尺寸
                    if correction_matrix.shape != channel_data.shape:
                        from scipy.interpolate import RectBivariateSpline
                        grid_h, grid_w = correction_matrix.shape
                        grid_y = np.linspace(0, channel_data.shape[0] - 1, grid_h)
                        grid_x = np.linspace(0, channel_data.shape[1] - 1, grid_w)
                        target_y = np.arange(channel_data.shape[0])
                        target_x = np.arange(channel_data.shape[1])
                        
                        interp_func = RectBivariateSpline(grid_y, grid_x, correction_matrix, kx=3, ky=3)
                        correction_matrix = interp_func(target_y, target_x)
                    
                    # 应用矫正
                    corrected_channel = channel_data.astype(np.float64) * correction_matrix
                    corrected_channels[channel_name] = np.clip(corrected_channel, 0, 4095).astype(np.uint16)
                else:
                    corrected_channels[channel_name] = channels[channel_name]
            
            # 重建矫正后的图像
            lens_corrected = reconstruct_corrected_image(corrected_channels, lens_shading_params)
            print(f"  3. Lens shading correction applied")
        else:
            lens_corrected = dark_corrected.copy()
            print(f"  3. Lens shading correction skipped")
        
        # 4. 去马赛克（Demosaicing）
        print(f"  4. Applying demosaicing...")
        color_image = demosaic_16bit(lens_corrected, 'rggb')
        print(f"  4. Demosaicing completed: {color_image.shape}")


        # 5. 转换为8位用于显示
        print(f"  5. Converting to 8-bit for display...")
        # 检查图像的实际最大值，使用动态范围而不是固定的4095
        max_val = np.max(color_image)
        print(f"  5. Color image range: {np.min(color_image)}-{max_val}")
        if max_val > 0:
            img_8bit = (color_image.astype(np.float32) / max_val * 255.0).astype(np.uint8)
        else:
            img_8bit = np.zeros_like(color_image, dtype=np.uint8)
        print(f"  5. 8-bit color image: {img_8bit.shape}, range: 0-255")
        
        print("Simplified ISP processing completed successfully")
        return {
            'raw_image': raw_data,
            'dark_corrected': dark_corrected,
            'lens_corrected': lens_corrected,
            'color_image': color_image,
            'img_8bit': img_8bit,
            'success': True
        }
            
    except Exception as e:
        print(f"Error in ISP processing: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def mouse_callback(event, x, y, flags, param):
    """
    鼠标回调函数用于ROI选择
    
    Args:
        event: 鼠标事件
        x, y: 鼠标坐标
        flags: 鼠标标志
        param: 额外参数
    """
    global roi_selected, roi_coords, roi_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 开始选择ROI
        roi_coords = (x, y)
        roi_selected = False
        
    elif event == cv2.EVENT_MOUSEMOVE and roi_coords is not None:
        # 更新ROI预览
        if roi_image is not None:
            preview = roi_image.copy()
            cv2.rectangle(preview, roi_coords, (x, y), (0, 255, 0), 2)
            cv2.imshow('Select White Board ROI', preview)
            
    elif event == cv2.EVENT_LBUTTONUP:
        # 完成ROI选择
        if roi_coords is not None:
            roi_coords = (min(roi_coords[0], x), min(roi_coords[1], y), 
                         max(roi_coords[0], x), max(roi_coords[1], y))
            roi_selected = True
            print(f"ROI selected: {roi_coords}")

def select_white_board_roi(color_image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    手动选择白板ROI区域
    
    Args:
        color_image: 彩色图像
        
    Returns:
        ROI坐标 (x1, y1, x2, y2)
    """
    global roi_selected, roi_coords, roi_image
    
    roi_selected = False
    roi_coords = None
    roi_image = color_image.copy()
    
    # 创建窗口
    cv2.namedWindow('Select White Board ROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select White Board ROI', 1200, 800)
    
    # 设置鼠标回调
    cv2.setMouseCallback('Select White Board ROI', mouse_callback)
    
    # 显示图像
    cv2.imshow('Select White Board ROI', color_image)
    
    print("\n=== Manual ROI Selection ===")
    print("Instructions:")
    print("1. Click and drag to select the white board region")
    print("2. Press 'Enter' to confirm selection")
    print("3. Press 'Esc' to cancel")
    print("4. Press 'r' to reset selection")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            if roi_selected and roi_coords is not None:
                break
            else:
                print("Please select a region first!")
                
        elif key == 27:  # Esc key
            print("ROI selection cancelled")
            cv2.destroyAllWindows()
            return None
            
        elif key == ord('r'):  # Reset
            roi_selected = False
            roi_coords = None
            cv2.imshow('Select White Board ROI', color_image)
            print("Selection reset")
    
    cv2.destroyAllWindows()
    
    if roi_coords is not None:
        x1, y1, x2, y2 = roi_coords
        print(f"ROI selected: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"ROI size: {x2-x1} x {y2-y1}")
        return roi_coords
    else:
        return None

def calculate_white_balance_from_roi_bayer(raw_bayer: np.ndarray, roi_coords: Tuple[int, int, int, int], bayer_pattern: str = 'rggb') -> Tuple[float, float, float]:
    """
    在 RGGB 拜尔阵列上直接计算白平衡参数（以 G 为基准）。

    Args:
        raw_bayer: 原始拜尔图 (H, W) uint16
        roi_coords: ROI坐标 (x1, y1, x2, y2)（像素坐标）
        bayer_pattern: 拜尔模式，默认 'rggb'

    Returns:
        (B_gain, G_gain, R_gain)
    """
    x1, y1, x2, y2 = roi_coords
    # 对齐到 R 起点（偶数行、偶数列）并保证宽高为偶数，避免破坏 2x2 CFA
    x1 = max(0, x1 - (x1 % 2))
    y1 = max(0, y1 - (y1 % 2))
    x2 = x2 - (x2 % 2)
    y2 = y2 - (y2 % 2)
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        raise ValueError('ROI too small after Bayer alignment')

    roi = raw_bayer[y1:y2, x1:x2]
    print(f"Bayer ROI shape (aligned): {roi.shape}")

    pat = bayer_pattern.lower()
    if pat != 'rggb':
        print(f"Warning: only 'rggb' is fully supported here; treating as 'rggb'.")

    # RGGB 索引
    R = roi[0::2, 0::2]
    G1 = roi[0::2, 1::2]
    G2 = roi[1::2, 0::2]
    B = roi[1::2, 1::2]

    # 使用均值（也可改用中位数以增强鲁棒性）
    r_mean = float(np.mean(R))
    g_mean = float((np.mean(G1) + np.mean(G2)) * 0.5)
    b_mean = float(np.mean(B))

    print(f"Bayer ROI means: R={r_mean:.2f}, G={g_mean:.2f}, B={b_mean:.2f}")

    # 以 G 为基准计算增益
    b_gain = (g_mean / b_mean) if b_mean > 0 else 1.0
    g_gain = 1.0
    r_gain = (g_mean / r_mean) if r_mean > 0 else 1.0

    return b_gain, g_gain, r_gain

def apply_white_balance(image: np.ndarray, b_gain: float, g_gain: float, r_gain: float) -> np.ndarray:
    """
    应用白平衡校正
    
    Args:
        image: 输入图像
        b_gain: B通道增益
        g_gain: G通道增益
        r_gain: R通道增益
        
    Returns:
        白平衡校正后的图像
    """
    corrected = image.copy().astype(np.float32)
    
    # 应用白平衡增益
    corrected[:, :, 0] *= b_gain  # B通道
    corrected[:, :, 1] *= g_gain  # G通道
    corrected[:, :, 2] *= r_gain  # R通道
    
    # 裁剪到有效范围
    corrected = np.clip(corrected, 0, 255)
    
    return corrected.astype(np.uint8)

def create_wb_analysis_plots(original_image: np.ndarray, corrected_image: np.ndarray,
                           wb_gains: Tuple[float, float, float], roi_coords: Tuple[int, int, int, int],
                           output_dir: Path) -> None:
    """
    创建白平衡分析图表
    
    Args:
        original_image: 原始图像
        corrected_image: 校正后图像
        wb_gains: 白平衡增益
        roi_coords: ROI坐标
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('White Balance Analysis - Manual ROI Selection', fontsize=16)
    
    # 原始图像
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 在原始图像上标记ROI
    if roi_coords is not None:
        x1, y1, x2, y2 = roi_coords
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(rect)
    
    # 校正后图像
    axes[0, 1].imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('White Balance Corrected')
    axes[0, 1].axis('off')
    
    # 差异图像
    diff = cv2.absdiff(original_image, corrected_image)
    axes[0, 2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Difference (Original - Corrected)')
    axes[0, 2].axis('off')
    
    # 原始图像直方图
    for i, (color, channel) in enumerate(zip(['Blue', 'Green', 'Red'], [0, 1, 2])):
        hist = cv2.calcHist([original_image], [channel], None, [256], [0, 256])
        axes[1, i].plot(hist, color=color.lower())
        axes[1, i].set_title(f'Original {color} Channel Histogram')
        axes[1, i].set_xlabel('Pixel Value')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
    
    # 校正后图像直方图
    for i, (color, channel) in enumerate(zip(['Blue', 'Green', 'Red'], [0, 1, 2])):
        hist = cv2.calcHist([corrected_image], [channel], None, [256], [0, 256])
        axes[1, i].plot(hist, color=color.lower(), linestyle='--')
        axes[1, i].set_title(f'Corrected {color} Channel Histogram')
        axes[1, i].set_xlabel('Pixel Value')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
    
    # 添加白平衡增益信息
    b_gain, g_gain, r_gain = wb_gains
    gain_text = f'WB Gains:\nB: {b_gain:.3f}\nG: {g_gain:.3f}\nR: {r_gain:.3f}'
    axes[1, 1].text(0.02, 0.98, gain_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = output_dir / f'wb_analysis_manual_roi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"White balance analysis plot saved: {plot_path}")
    
    plt.show()

def save_wb_parameters(wb_gains: Tuple[float, float, float], roi_coords: Tuple[int, int, int, int],
                      output_dir: Path) -> Path:
    """
    保存白平衡参数
    
    Args:
        wb_gains: 白平衡增益
        roi_coords: ROI坐标
        output_dir: 输出目录
        
    Returns:
        参数文件路径
    """
    b_gain, g_gain, r_gain = wb_gains
    
    parameters = {
        'white_balance_gains': {
            'b_gain': float(b_gain),
            'g_gain': float(g_gain),
            'r_gain': float(r_gain)
        },
        'method': 'manual_roi',
        'roi_coordinates': {
            'x1': int(roi_coords[0]),
            'y1': int(roi_coords[1]),
            'x2': int(roi_coords[2]),
            'y2': int(roi_coords[3])
        },
        'timestamp': datetime.now().isoformat(),
        'description': 'White balance parameters calculated using manual ROI selection method'
    }
    
    # 保存为JSON文件
    params_path = output_dir / f'wb_parameters_manual_roi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(parameters, f, indent=2, ensure_ascii=False)
    
    print(f"White balance parameters saved: {params_path}")
    return params_path

def save_wb_images(original_image: np.ndarray, corrected_image: np.ndarray,
                  output_dir: Path) -> None:
    """
    保存白平衡图像
    
    Args:
        original_image: 原始图像
        corrected_image: 校正后图像
        output_dir: 输出目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存原始图像
    original_path = output_dir / f'original_image_{timestamp}.png'
    cv2.imwrite(str(original_path), original_image)
    print(f"Original image saved: {original_path}")
    
    # 保存校正后图像
    corrected_path = output_dir / f'wb_corrected_manual_roi_{timestamp}.png'
    cv2.imwrite(str(corrected_path), corrected_image)
    print(f"White balance corrected image saved: {corrected_path}")

def calculate_white_balance(input_image_path: str, config: Dict) -> Dict:
    """
    计算白平衡参数
    
    Args:
        input_image_path: 输入图像路径
        config: 配置参数
        
    Returns:
        白平衡结果字典
    """
    print("=== White Balance Calculator with Manual ROI Selection ===")
    print(f"Input image: {input_image_path}")
    print(f"Image dimensions: {config['IMAGE_WIDTH']} x {config['IMAGE_HEIGHT']}")
    print(f"Data type: {config['DATA_TYPE']}")
    print(f"Bayer pattern: {config['BAYER_PATTERN']}")
    
    # 创建输出目录
    if config['OUTPUT_DIRECTORY'] is None:
        output_dir = Path(input_image_path).parent / f"wb_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(config['OUTPUT_DIRECTORY'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    try:
        # 使用ISP流程处理图像
        isp_result = process_image_with_isp(input_image_path, config)
        
        if not isp_result['success']:
            raise Exception("ISP processing failed")
        
        color_image = isp_result['img_8bit']
        
        # 手动选择白板ROI区域（用于可视化），但增益改在拜尔域上计算
        print("\n=== Manual ROI Selection ===")
        roi_coords = select_white_board_roi(color_image)
        
        if roi_coords is None:
            raise Exception("ROI selection cancelled or failed")
        
        # 在拜尔域上计算白平衡参数
        print("\n=== White Balance Calculation on Bayer (RGGB) ===")
        raw_bayer = isp_result['raw_image'].astype(np.uint16)
        wb_gains = calculate_white_balance_from_roi_bayer(raw_bayer, roi_coords, config['BAYER_PATTERN'])
        
        b_gain, g_gain, r_gain = wb_gains
        print(f"White balance gains: B={b_gain:.3f}, G={g_gain:.3f}, R={r_gain:.3f}")
        
        # 应用白平衡校正
        print("Applying white balance correction...")
        corrected_image = apply_white_balance(color_image, b_gain, g_gain, r_gain)
        
        # 保存结果
        if config['SAVE_RESULTS']:
            print("\nSaving results...")
            
            if config['SAVE_PARAMETERS']:
                save_wb_parameters(wb_gains, roi_coords, output_dir)
            
            if config['SAVE_IMAGES']:
                save_wb_images(color_image, corrected_image, output_dir)
            
            # 创建分析图表
            create_wb_analysis_plots(color_image, corrected_image, wb_gains, roi_coords, output_dir)
        
        # 返回结果
        result = {
            'success': True,
            'white_balance_gains': {
                'b_gain': float(b_gain),
                'g_gain': float(g_gain),
                'r_gain': float(r_gain)
            },
            'method': 'manual_roi',
            'roi_coordinates': {
                'x1': int(roi_coords[0]),
                'y1': int(roi_coords[1]),
                'x2': int(roi_coords[2]),
                'y2': int(roi_coords[3])
            },
            'output_directory': str(output_dir),
            'image_info': {
                'original_shape': isp_result['raw_image'].shape,
                'original_dtype': str(isp_result['raw_image'].dtype),
                'original_range': [int(np.min(isp_result['raw_image'])), int(np.max(isp_result['raw_image']))],
                'color_shape': color_image.shape,
                'roi_shape': (roi_coords[2]-roi_coords[0], roi_coords[3]-roi_coords[1])
            }
        }
        
        print("\n=== White Balance Calculation Complete ===")
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'success': False,
            'error': str(e),
            'output_directory': str(output_dir)
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='White Balance Calculator with Manual ROI Selection')
    parser.add_argument('--input', '-i', type=str, default=CONFIG['INPUT_IMAGE_PATH'],
                       help='Input RAW image path')
    parser.add_argument('--width', '-w', type=int, default=CONFIG['IMAGE_WIDTH'],
                       help='Image width')
    parser.add_argument('--height', type=int, default=CONFIG['IMAGE_HEIGHT'],
                       help='Image height')
    parser.add_argument('--pattern', '-p', type=str, default=CONFIG['BAYER_PATTERN'],
                       choices=['rggb', 'bggr', 'grbg', 'gbrg'],
                       help='Bayer pattern')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--dark', '-d', type=str, default=CONFIG['DARK_RAW_PATH'],
                       help='Dark reference image path')
    parser.add_argument('--lens-shading', '-l', type=str, default=CONFIG['LENS_SHADING_PARAMS_DIR'],
                       help='Lens shading parameters directory')
    parser.add_argument('--no-dark', dest='dark_enabled', action='store_false', default=True,
                       help='Disable dark current correction')
    parser.add_argument('--no-lens-shading', dest='lens_shading_enabled', action='store_false', default=True,
                       help='Disable lens shading correction')
    
    args = parser.parse_args()
    
    # 更新配置
    config = CONFIG.copy()
    config['INPUT_IMAGE_PATH'] = args.input
    config['IMAGE_WIDTH'] = args.width
    config['IMAGE_HEIGHT'] = args.height
    config['BAYER_PATTERN'] = args.pattern
    config['OUTPUT_DIRECTORY'] = args.output
    config['DARK_RAW_PATH'] = args.dark
    config['LENS_SHADING_PARAMS_DIR'] = args.lens_shading
    config['DARK_SUBTRACTION_ENABLED'] = args.dark_enabled
    config['LENS_SHADING_ENABLED'] = args.lens_shading_enabled
    
    # 计算白平衡
    result = calculate_white_balance(args.input, config)
    
    if result['success']:
        print(f"\nWhite balance gains:")
        print(f"  B gain: {result['white_balance_gains']['b_gain']:.3f}")
        print(f"  G gain: {result['white_balance_gains']['g_gain']:.3f}")
        print(f"  R gain: {result['white_balance_gains']['r_gain']:.3f}")
        print(f"\nROI coordinates:")
        print(f"  ({result['roi_coordinates']['x1']}, {result['roi_coordinates']['y1']}) to ({result['roi_coordinates']['x2']}, {result['roi_coordinates']['y2']})")
        print(f"\nResults saved to: {result['output_directory']}")
    else:
        print(f"\nError: {result['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
