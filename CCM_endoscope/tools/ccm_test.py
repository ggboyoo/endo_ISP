#!/usr/bin/env python3
"""
CCM Endoscope Calibration Program
色卡CCM标定程序
读取色卡RAW图，调用ISP处理，进行CCM标定
"""

import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('TkAgg')  # 设置matplotlib后端
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import argparse

# 导入ISP处理功能
try:
    from ISP import process_single_image, load_dark_reference
except ImportError:
    print("Error: ISP.py not found in the same directory!")
    exit(1)

# 导入CCM计算功能
try:
    from ccm_calculator import (
        srgb_to_linear, linear_to_srgb, 
        xyz_to_rgb_linear, rgb_linear_to_xyz,
        lab_to_xyz, xyz_to_lab,
        compute_delta_e_loss, solve_ccm_gradient_optimization,
        apply_white_balance_only
    )
except ImportError:
    print("Error: ccm_calculator.py not found in the same directory!")
    exit(1)

# 配置参数
# 支持文件夹平均功能：
# - 如果路径是文件：直接使用该文件
# - 如果路径是文件夹：自动加载文件夹内所有.raw/.RAW文件并平均，用于去除噪声
CONFIG = {
    # 输入配置
    'INPUT_IMAGE_PATH':r"F:\ZJU\Picture\ccm\ccm_2\dark_24",  # 色卡RAW图路径（文件或文件夹）
    'ILLUMINATION_IMAGE_PATH': r"F:\ZJU\Picture\ccm\ccm_2\dark_ill",  # 照明图RAW路径（文件或文件夹），None表示不使用照明调整
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
    'WHITE_BALANCE_ENABLED': True,
    'WB_PARAMETERS_PATH': r"F:\ZJU\Picture\wb\wb_output\wb_parameters_manual_roi_20250902_114009.json",
    
    # CCM配置
    'CCM_METHOD': 'gradient_optimization',  # CCM计算方法：gradient_optimization, linear_regression
    'WHITE_BALANCE_ENABLED_CCM': True,  # CCM计算时是否启用白平衡
    'LUMINANCE_NORMALIZATION': True,  # 是否启用亮度归一化
    'WHITE_PRESERVATION_CONSTRAINT': True,  # 是否启用白色保持约束
    'PATCH_19_20_WHITE_BALANCE': False,  # 是否使用第19、20色块进行白平衡
    
    # 角度校正配置
    'ENABLE_ANGLE_CORRECTION': True,  # 是否启用透视变换校正
    'PERSPECTIVE_TRANSFORM': True,    # 是否使用透视变换（True）或旋转（False）
    
    # 照明调整配置
    'ILLUMINATION_CORRECTION_ENABLED': True,  # 是否启用照明调整
    'ILLUMINATION_GRID_SIZE': 8,  # 照明调整网格大小（像素）
    'ILLUMINATION_SMOOTHING': True,  # 是否对照明调整进行平滑处理
    'ILLUMINATION_REFERENCE_METHOD': 'center',  # 照明参考方法：'center', 'max', 'mean'
    'ILLUMINATION_ADJUSTMENT_STRENGTH': 1.0,  # 照明调整强度（0.0-1.0）
    
    # 输出配置
    'OUTPUT_DIRECTORY': None,  # None表示自动创建
    'SAVE_RESULTS': True,
    'SAVE_IMAGES': True,
    'SAVE_PARAMETERS': True,
    'GENERATE_PLOTS': True,
}

# ============================================================================
# 照明调整相关函数
# ============================================================================

def load_illumination_image(illumination_path: str, width: int, height: int, data_type: str) -> Optional[np.ndarray]:
    """
    加载照明图RAW数据，支持单个文件或文件夹平均
    
    Args:
        illumination_path: 照明图路径（文件或文件夹）
        width: 图像宽度
        height: 图像高度
        data_type: 数据类型
        
    Returns:
        照明图数据或None
    """
    try:
        illumination_path = Path(illumination_path)
        
        if illumination_path.is_file():
            # 单个文件
            print(f"Loading single illumination image: {illumination_path}")
            return load_single_raw_file(illumination_path, width, height, data_type)
        elif illumination_path.is_dir():
            # 文件夹，加载所有RAW文件并平均
            print(f"Loading illumination images from folder: {illumination_path}")
            return load_and_average_raw_files(illumination_path, width, height, data_type)
        else:
            print(f"Illumination path not found: {illumination_path}")
            return None
            
    except Exception as e:
        print(f"Error loading illumination image: {e}")
        return None

def load_single_raw_file(file_path: Path, width: int, height: int, data_type: str) -> Optional[np.ndarray]:
    """加载单个RAW文件"""
    try:
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        
        # 读取RAW数据
        if data_type == "uint16":
            raw_data = np.fromfile(str(file_path), dtype=np.uint16)
        elif data_type == "uint8":
            raw_data = np.fromfile(str(file_path), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # 调整大小
        expected_size = width * height
        if len(raw_data) != expected_size:
            print(f"Warning: Expected {expected_size} pixels, got {len(raw_data)}")
            if len(raw_data) > expected_size:
                raw_data = raw_data[:expected_size]
            else:
                raw_data = np.pad(raw_data, (0, expected_size - len(raw_data)), 'constant')
        
        illumination_data = raw_data.reshape((height, width))
        print(f"Single RAW file loaded: {illumination_data.shape}, range: {np.min(illumination_data)}-{np.max(illumination_data)}")
        return illumination_data
        
    except Exception as e:
        print(f"Error loading single RAW file: {e}")
        return None

def load_and_average_raw_files(folder_path: Path, width: int, height: int, data_type: str) -> Optional[np.ndarray]:
    """加载文件夹内所有RAW文件并平均"""
    try:
        # 查找所有RAW文件
        raw_files = list(folder_path.glob("*.raw")) + list(folder_path.glob("*.RAW"))
        
        if not raw_files:
            print(f"No RAW files found in folder: {folder_path}")
            return None
        
        print(f"Found {len(raw_files)} RAW files in folder")
        
        # 加载所有RAW文件
        raw_images = []
        for i, raw_file in enumerate(raw_files):
            print(f"Loading RAW file {i+1}/{len(raw_files)}: {raw_file.name}")
            raw_data = load_single_raw_file(raw_file, width, height, data_type)
            if raw_data is not None:
                raw_images.append(raw_data)
            else:
                print(f"Failed to load: {raw_file.name}")
        
        if not raw_images:
            print("No valid RAW files loaded")
            return None
        
        print(f"Successfully loaded {len(raw_images)} RAW files")
        
        # 转换为numpy数组进行平均
        raw_images_array = np.array(raw_images)
        
        # 计算平均值
        averaged_data = np.mean(raw_images_array, axis=0).astype(raw_images_array.dtype)
        
        print(f"Averaged illumination data: {averaged_data.shape}, range: {np.min(averaged_data)}-{np.max(averaged_data)}")
        print(f"Standard deviation: {np.std(averaged_data):.2f}")
        
        return averaged_data
        
    except Exception as e:
        print(f"Error loading and averaging RAW files: {e}")
        return None

def process_illumination_image(illumination_data: np.ndarray, dark_data: np.ndarray, 
                              lens_shading_params: Dict, config: Dict) -> np.ndarray:
    """
    处理照明图，应用暗电流矫正和镜头阴影矫正
    
    Args:
        illumination_data: 照明图RAW数据
        dark_data: 暗电流数据
        lens_shading_params: 镜头阴影矫正参数
        config: 配置参数
        
    Returns:
        处理后的照明图
    """
    try:
        # 1. 暗电流矫正
        if config['DARK_SUBTRACTION_ENABLED'] and dark_data is not None:
            print("Applying dark correction to illumination image...")
            corrected_data = illumination_data.astype(np.float64) - dark_data.astype(np.float64)
            corrected_data = np.clip(corrected_data, 0, None)
        else:
            corrected_data = illumination_data.astype(np.float64)
        
        # 2. 镜头阴影矫正
        if config['LENS_SHADING_ENABLED'] and lens_shading_params:
            print("Applying lens shading correction to illumination image...")
            from lens_shading import shading_correct
            corrected_data = shading_correct(corrected_data, lens_shading_params)
        
        # 3. 去马赛克
        print("Demosaicing illumination image...")
        corrected_data_uint16 = corrected_data.astype(np.uint16)
        demosaiced = cv2.cvtColor(corrected_data_uint16, cv2.COLOR_BayerRG2BGR)
        
        # 4. 转换为灰度图
        gray_illumination = cv2.cvtColor(demosaiced, cv2.COLOR_BGR2GRAY)
        
        print(f"Illumination processing completed: {gray_illumination.shape}, range: {np.min(gray_illumination)}-{np.max(gray_illumination)}")
        return gray_illumination
        
    except Exception as e:
        print(f"Error processing illumination image: {e}")
        return illumination_data.astype(np.float64)

def create_illumination_correction_grid(illumination_image: np.ndarray, config: Dict) -> np.ndarray:
    """
    根据照明图创建网格化的亮度调整矩阵
    
    Args:
        illumination_image: 处理后的照明图（灰度图）
        config: 配置参数
        
    Returns:
        亮度调整矩阵
    """
    try:
        # 处理输入图像，确保是灰度图
        if len(illumination_image.shape) == 3:
            # 如果是彩色图像，转换为灰度图
            illumination_gray = cv2.cvtColor(illumination_image, cv2.COLOR_BGR2GRAY)
            print(f"Converted 3D illumination image to grayscale: {illumination_gray.shape}")
        else:
            illumination_gray = illumination_image
            print(f"Using grayscale illumination image: {illumination_gray.shape}")
        
        height, width = illumination_gray.shape
        grid_size = config['ILLUMINATION_GRID_SIZE']
        reference_method = config['ILLUMINATION_REFERENCE_METHOD']
        adjustment_strength = config['ILLUMINATION_ADJUSTMENT_STRENGTH']
        
        print(f"Creating illumination correction grid: {grid_size}x{grid_size} pixels")
        print(f"Illumination image dimensions: {width}x{height}")
        
        # 计算网格数量
        grid_h = height // grid_size
        grid_w = width // grid_size
        
        print(f"Grid dimensions: {grid_w}x{grid_h} (width x height)")
        print(f"Grid size: {grid_size} pixels")
        
        # 检查网格数量是否有效
        if grid_h <= 0 or grid_w <= 0:
            print(f"ERROR: Invalid grid dimensions! Image size: {width}x{height}, Grid size: {grid_size}")
            print(f"Calculated grid: {grid_w}x{grid_h}")
            return np.ones((1, 1), dtype=np.float64)
        
        # 创建网格调整矩阵
        correction_grid = np.ones((grid_h, grid_w), dtype=np.float64)
        print(f"Initial correction grid shape: {correction_grid.shape}")
        
        # 计算参考亮度值
        if reference_method == 'center':
            # 计算中心网格的平均亮度
            center_h, center_w = height // 2, width // 2
            center_grid_h = center_h // grid_size
            center_grid_w = center_w // grid_size
            
            # 确保中心网格索引在有效范围内
            center_grid_h = min(center_grid_h, grid_h - 1)
            center_grid_w = min(center_grid_w, grid_w - 1)
            
            # 计算中心网格区域
            start_h = center_grid_h * grid_size
            end_h = min((center_grid_h + 1) * grid_size, height)
            start_w = center_grid_w * grid_size
            end_w = min((center_grid_w + 1) * grid_size, width)
            
            # 计算中心网格的平均亮度
            center_grid_region = illumination_gray[start_h:end_h, start_w:end_w]
            reference_brightness = np.mean(center_grid_region)
            
            print(f"Center grid region: ({start_h}:{end_h}, {start_w}:{end_w})")
            print(f"Center grid size: {end_h-start_h}x{end_w-start_w}")
        elif reference_method == 'max':
            reference_brightness = np.max(illumination_gray)
        elif reference_method == 'mean':
            reference_brightness = np.mean(illumination_gray)
        else:
            reference_brightness = np.mean(illumination_gray)
        
        print(f"Reference brightness ({reference_method}): {reference_brightness}")
        print(f"Processing {grid_h * grid_w} grid cells...")
        
        # 计算每个网格的调整系数
        processed_cells = 0
        for i in range(grid_h):
            for j in range(grid_w):
                # 计算网格区域
                start_h = i * grid_size
                end_h = min((i + 1) * grid_size, height)
                start_w = j * grid_size
                end_w = min((j + 1) * grid_size, width)
                
                # 计算网格平均亮度
                grid_region = illumination_gray[start_h:end_h, start_w:end_w]
                grid_brightness = np.mean(grid_region)
                
                # 计算调整系数
                if grid_brightness > 50:
                    correction_factor = reference_brightness / grid_brightness
                    # 应用调整强度
                    correction_factor = 1.0 + (correction_factor - 1.0) * adjustment_strength
                    correction_grid[i, j] = correction_factor
                else:
                    correction_grid[i, j] = 1.0
                
                processed_cells += 1
                if processed_cells % 100 == 0:  # 每100个网格打印一次进度
                    print(f"Processed {processed_cells}/{grid_h * grid_w} grid cells...")
        
        # 平滑处理
        if config['ILLUMINATION_SMOOTHING']:
            from scipy.ndimage import gaussian_filter
            correction_grid = gaussian_filter(correction_grid, sigma=5.0)
        
        print(f"Illumination correction grid created: {correction_grid.shape}")
        print(f"Correction range: {np.min(correction_grid):.3f} - {np.max(correction_grid):.3f}")
        
        return correction_grid
        
    except Exception as e:
        print(f"Error creating illumination correction grid: {e}")
        return np.ones((1, 1), dtype=np.float64)

def apply_illumination_correction(color_image: np.ndarray, correction_grid: np.ndarray, 
                                 grid_size: int) -> np.ndarray:
    """
    应用照明调整到彩色图像
    
    Args:
        color_image: 彩色图像
        correction_grid: 调整网格
        grid_size: 网格大小
        
    Returns:
        调整后的彩色图像
    """
    try:
        height, width = color_image.shape[:2]
        corrected_image = color_image.copy().astype(np.float64)
        
        grid_h, grid_w = correction_grid.shape
        
        # 对每个网格应用调整
        for i in range(grid_h):
            for j in range(grid_w):
                # 计算网格区域
                start_h = i * grid_size
                end_h = min((i + 1) * grid_size, height)
                start_w = j * grid_size
                end_w = min((j + 1) * grid_size, width)
                
                # 应用调整系数
                correction_factor = correction_grid[i, j]
                corrected_image[start_h:end_h, start_w:end_w] *= correction_factor
        
        # 限制到有效范围
        corrected_image = np.clip(corrected_image, 0, 4095)
        
        print(f"Illumination correction applied to color image")
        print(f"Original range: {np.min(color_image)} - {np.max(color_image)}")
        print(f"Corrected range: {np.min(corrected_image)} - {np.max(corrected_image)}")
        
        return corrected_image.astype(np.uint16)
        
    except Exception as e:
        print(f"Error applying illumination correction: {e}")
        return color_image

# 24色卡标准值 (sRGB D65)
SRGB_24PATCH_D65_8BIT = np.array([
    [115, 82, 68],    # 1. Dark skin
    [194, 150, 130],  # 2. Light skin
    [98, 122, 157],   # 3. Blue sky
    [87, 108, 67],    # 4. Foliage
    [133, 128, 177],  # 5. Blue flower
    [103, 189, 170],  # 6. Bluish green
    [214, 126, 44],   # 7. Orange
    [80, 91, 166],    # 8. Purplish blue
    [193, 90, 99],    # 9. Moderate red
    [94, 60, 108],    # 10. Purple
    [157, 188, 64],   # 11. Yellow green
    [224, 163, 46],   # 12. Orange yellow
    [56, 61, 150],    # 13. Blue
    [70, 148, 73],    # 14. Green
    [175, 54, 60],    # 15. Red
    [231, 199, 31],   # 16. Yellow
    [187, 86, 149],   # 17. Magenta
    [8, 133, 161],    # 18. Cyan
    [243, 243, 242],  # 19. White
    [200, 200, 200],  # 20. Neutral 8
    [160, 160, 160],  # 21. Neutral 6.5
    [122, 122, 121],  # 22. Neutral 5
    [85, 85, 85],     # 23. Neutral 3.5
    [52, 52, 52]      # 24. Black
], dtype=np.float64)

def normalize_16bit_to_float(data: np.ndarray, data_type: str) -> np.ndarray:
    """
    将16位数据归一化到浮点范围(0, 255)
    
    Args:
        data: 16位数据
        data_type: 原始数据类型
        
    Returns:
        归一化到(0, 255)范围的浮点数据
    """
    if data_type == 'uint16':
        # 12位数据存储在16位容器中，最大值是4095
        max_val = 4095.0
    elif data_type == 'uint8':
        max_val = 255.0
    else:
        max_val = float(np.max(data))
    
    # 归一化到(0, 255)范围，使用float64确保精度
    normalized = data.astype(np.float64)
    normalized = np.clip(normalized, 0, max_val)
    normalized = (normalized / max_val * 255.0)
    
    return normalized

def draw_patch_position_info(preview_image, patch_x, patch_y, patch_w, patch_h):
    """在预览图像上绘制色块位置信息"""
    # 添加位置信息（右下角）
    position_text = f"({patch_x},{patch_y})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos_font_scale = 0.5
    pos_thickness = 1
    
    # 获取位置文字尺寸
    (pos_text_width, pos_text_height), _ = cv2.getTextSize(position_text, font, pos_font_scale, pos_thickness)
    
    # 计算位置文字位置（右下角）
    pos_text_x = patch_x + patch_w - pos_text_width - 5
    pos_text_y = patch_y + patch_h - 5
    
    # 绘制位置信息的黑色描边
    cv2.putText(preview_image, position_text, (pos_text_x - 1, pos_text_y - 1), font, pos_font_scale, (0, 0, 0), pos_thickness + 1)
    cv2.putText(preview_image, position_text, (pos_text_x + 1, pos_text_y + 1), font, pos_font_scale, (0, 0, 0), pos_thickness + 1)
    cv2.putText(preview_image, position_text, (pos_text_x - 1, pos_text_y + 1), font, pos_font_scale, (0, 0, 0), pos_thickness + 1)
    cv2.putText(preview_image, position_text, (pos_text_x + 1, pos_text_y - 1), font, pos_font_scale, (0, 0, 0), pos_thickness + 1)
    
    # 绘制位置信息的白色文字
    cv2.putText(preview_image, position_text, (pos_text_x, pos_text_y), font, pos_font_scale, (255, 255, 255), pos_thickness)

def select_colorcard_corners(color_image):
    """
    交互式选择色卡的4个顶点
    
    Args:
        color_image: 彩色图像
        
    Returns:
        4个顶点的坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    print("\n=== Color Card Corner Selection ===")
    print("Please select 4 corners of the color card in order:")
    print("1. Top-left corner")
    print("2. Top-right corner") 
    print("3. Bottom-right corner")
    print("4. Bottom-left corner")
    print("Click on each corner, then press 'Enter' to confirm")
    print("Press 'r' to reset, 'Esc' to cancel")
    
    corners = []
    temp_image = color_image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal corners, temp_image
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 4:
                corners.append((x, y))
                print(f"Corner {len(corners)}: ({x}, {y})")
                
                # 绘制已选择的点
                cv2.circle(temp_image, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(temp_image, str(len(corners)), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 绘制连线
                if len(corners) > 1:
                    cv2.line(temp_image, corners[-2], corners[-1], (0, 255, 0), 2)
                if len(corners) == 4:
                    # 闭合四边形
                    cv2.line(temp_image, corners[-1], corners[0], (0, 255, 0), 2)
    
    cv2.namedWindow('Color Card Corner Selection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Color Card Corner Selection', 1200, 800)
    cv2.setMouseCallback('Color Card Corner Selection', mouse_callback)
    
    while True:
        cv2.imshow('Color Card Corner Selection', temp_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter键
            if len(corners) == 4:
                break
            else:
                print(f"Please select all 4 corners. Currently selected: {len(corners)}")
        elif key == ord('r'):  # Reset
            corners = []
            temp_image = color_image.copy()
            print("Reset corner selection")
        elif key == 27:  # Esc键
            cv2.destroyAllWindows()
            raise Exception("User cancelled corner selection")
    
    cv2.destroyAllWindows()
    
    print(f"Selected corners: {corners}")
    return corners

def calculate_perspective_transform(corners):
    """
    根据4个顶点计算透视变换矩阵，将色卡变形为矩形
    
    Args:
        corners: 4个顶点的坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        顺序: [top_left, top_right, bottom_right, bottom_left]
        
    Returns:
        (transform_matrix, target_size): 透视变换矩阵和目标尺寸
    """
    top_left, top_right, bottom_right, bottom_left = corners
    
    # 计算色卡的实际尺寸
    # 上边长度
    top_length = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
    # 下边长度
    bottom_length = np.sqrt((bottom_right[0] - bottom_left[0])**2 + (bottom_right[1] - bottom_left[1])**2)
    # 左边长度
    left_length = np.sqrt((bottom_left[0] - top_left[0])**2 + (bottom_left[1] - top_left[1])**2)
    # 右边长度
    right_length = np.sqrt((bottom_right[0] - top_right[0])**2 + (bottom_right[1] - top_right[1])**2)
    
    # 计算平均宽度和高度
    avg_width = (top_length + bottom_length) / 2
    avg_height = (left_length + right_length) / 2
    
    print(f"Color card dimensions:")
    print(f"  Top edge: {top_length:.1f}px")
    print(f"  Bottom edge: {bottom_length:.1f}px")
    print(f"  Left edge: {left_length:.1f}px")
    print(f"  Right edge: {right_length:.1f}px")
    print(f"  Average width: {avg_width:.1f}px")
    print(f"  Average height: {avg_height:.1f}px")
    
    # 定义目标矩形的四个顶点（标准矩形）
    target_width = int(avg_width)
    target_height = int(avg_height)
    
    # 目标矩形的四个顶点（左上、右上、右下、左下）
    target_corners = np.float32([
        [0, 0],                           # 左上
        [target_width, 0],                # 右上
        [target_width, target_height],    # 右下
        [0, target_height]                # 左下
    ])
    
    # 源顶点
    source_corners = np.float32(corners)
    
    # 计算透视变换矩阵
    transform_matrix = cv2.getPerspectiveTransform(source_corners, target_corners)
    
    print(f"Target rectangle size: {target_width}x{target_height}")
    print(f"Perspective transform matrix calculated")
    
    return transform_matrix, (target_width, target_height)

def preview_perspective_transform(image, corners, transform_matrix, target_size):
    """
    预览透视变换效果
    
    Args:
        image: 输入图像
        corners: 4个顶点坐标
        transform_matrix: 透视变换矩阵
        target_size: 目标尺寸 (width, height)
    """
    print(f"\n=== Perspective Transform Preview ===")
    print(f"Previewing perspective transform to {target_size[0]}x{target_size[1]}...")
    
    # 创建透视变换后的图像
    transformed_image = cv2.warpPerspective(image, transform_matrix, target_size)
    
    # 创建对比显示
    h1, w1 = image.shape[:2]
    h2, w2 = transformed_image.shape[:2]
    
    # 调整图像大小以便并排显示
    max_h = max(h1, h2)
    display_w = w1 + w2 + 50  # 中间留50像素间隔
    display_h = max_h + 100   # 上下留空间添加标题
    
    display_image = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    display_image[:] = (50, 50, 50)  # 深灰背景
    
    # 放置原图（左侧）
    y_offset = 60
    display_image[y_offset:y_offset+h1, 0:w1] = image
    
    # 在原图上绘制选择的顶点
    for i, corner in enumerate(corners):
        cv2.circle(display_image, corner, 8, (0, 255, 0), -1)
        cv2.putText(display_image, str(i+1), (corner[0]+10, corner[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 绘制连接线
    for i in range(4):
        cv2.line(display_image, corners[i], corners[(i+1)%4], (0, 255, 0), 2)
    
    # 放置变换后的图（右侧）
    display_image[y_offset:y_offset+h2, w1+50:w1+50+w2] = transformed_image
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # 主标题
    main_title = f"Perspective Transform Preview: {target_size[0]}x{target_size[1]}"
    (title_width, title_height), _ = cv2.getTextSize(main_title, font, font_scale, thickness)
    title_x = (display_w - title_width) // 2
    title_y = 30
    
    cv2.putText(display_image, main_title, (title_x-1, title_y-1), font, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(display_image, main_title, (title_x+1, title_y+1), font, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(display_image, main_title, (title_x, title_y), font, font_scale, (255, 255, 255), thickness)
    
    # 子标题
    font_scale_sub = 0.7
    thickness_sub = 2
    
    # "Original"标题
    original_title = "Original (with corners)"
    (orig_width, orig_height), _ = cv2.getTextSize(original_title, font, font_scale_sub, thickness_sub)
    orig_x = (w1 - orig_width) // 2
    orig_y = y_offset - 10
    
    cv2.putText(display_image, original_title, (orig_x-1, orig_y-1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
    cv2.putText(display_image, original_title, (orig_x+1, orig_y+1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
    cv2.putText(display_image, original_title, (orig_x, orig_y), font, font_scale_sub, (255, 255, 255), thickness_sub)
    
    # "Transformed"标题
    transformed_title = "Transformed Rectangle"
    (trans_width, trans_height), _ = cv2.getTextSize(transformed_title, font, font_scale_sub, thickness_sub)
    trans_x = w1 + 50 + (w2 - trans_width) // 2
    trans_y = y_offset - 10
    
    cv2.putText(display_image, transformed_title, (trans_x-1, trans_y-1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
    cv2.putText(display_image, transformed_title, (trans_x+1, trans_y+1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
    cv2.putText(display_image, transformed_title, (trans_x, trans_y), font, font_scale_sub, (255, 255, 255), thickness_sub)
    
    # 添加分割线
    line_y = y_offset - 20
    cv2.line(display_image, (0, line_y), (display_w, line_y), (100, 100, 100), 2)
    
    # 显示预览
    cv2.namedWindow('Perspective Transform Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Perspective Transform Preview', 1400, 800)
    
    print("Please check the perspective transform preview:")
    print("Press 'Enter' to confirm transform")
    print("Press 'r' to reselect corners")
    print("Press 'Esc' to cancel transform")
    
    while True:
        cv2.imshow('Perspective Transform Preview', display_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter键
            break
        elif key == ord('r'):  # r键重新选择顶点
            cv2.destroyAllWindows()
            return 'recalculate'
        elif key == 27:  # Esc键
            cv2.destroyAllWindows()
            return 'cancel'
    
    cv2.destroyAllWindows()
    return 'confirm'

def apply_perspective_transform(image, transform_matrix, target_size):
    """
    应用透视变换将图像变形为矩形
    
    Args:
        image: 输入图像
        transform_matrix: 透视变换矩阵
        target_size: 目标尺寸 (width, height)
        
    Returns:
        透视变换后的图像
    """
    print(f"Applying perspective transform to {target_size[0]}x{target_size[1]}...")
    
    # 执行透视变换
    transformed_image = cv2.warpPerspective(image, transform_matrix, target_size, 
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    print(f"Image transformed: {image.shape} -> {transformed_image.shape}")
    return transformed_image

def show_center_regions_window(center_regions, patch_size=50):
    """创建窗口显示24个center_region"""
    print("\n=== Center Regions Display ===")
    print(f"Showing {len(center_regions)} center regions with size: {patch_size}x{patch_size}")
    print("Please check the center regions used for color calculation:")
    print("Press 'Enter' to continue with calibration")
    print("Press 'Esc' to cancel")
    
    # 24色卡通常是4x6排列
    rows, cols = 4, 6
    
    # 计算显示窗口大小
    display_width = cols * patch_size
    display_height = rows * patch_size
    
    # 创建显示图像
    display_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    print(f"Display window size: {display_width}x{display_height}")
    
    # 绘制center_region
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            
            if idx < len(center_regions) and center_regions[idx] is not None:
                # 计算色块位置
                patch_x = j * patch_size
                patch_y = i * patch_size
                
                # 获取center_region
                center_region = center_regions[idx]
                
                # 调整center_region大小到patch_size
                if center_region.shape[:2] != (patch_size, patch_size):
                    center_region_resized = cv2.resize(center_region, (patch_size, patch_size))
                else:
                    center_region_resized = center_region
                
                # 将center_region放置到display_image中
                display_image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size] = center_region_resized
                
                # 绘制边界框（白色边框）
                cv2.rectangle(display_image, (patch_x, patch_y), 
                             (patch_x + patch_size, patch_y + patch_size), (255, 255, 255), 2)
                
                # 绘制内边框（黑色边框，增强对比度）
                cv2.rectangle(display_image, (patch_x + 1, patch_y + 1), 
                             (patch_x + patch_size - 1, patch_y + patch_size - 1), (0, 0, 0), 1)
                
                # 添加编号（白色文字，黑色描边）
                text = str(idx + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                
                # 获取文字尺寸
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # 计算文字位置（左上角）
                text_x = patch_x + 5
                text_y = patch_y + text_height + 5
                
                # 绘制黑色描边
                cv2.putText(display_image, text, (text_x - 1, text_y - 1), font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(display_image, text, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(display_image, text, (text_x - 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(display_image, text, (text_x + 1, text_y - 1), font, font_scale, (0, 0, 0), thickness + 1)
                
                # 绘制白色文字
                cv2.putText(display_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # 显示center_region窗口
    cv2.namedWindow('Center Regions Display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Center Regions Display', display_width + 100, display_height + 100)
    
    while True:
        cv2.imshow('Center Regions Display', display_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter键
            break
        elif key == 27:  # Esc键
            cv2.destroyAllWindows()
            raise Exception("User cancelled center regions preview")
    
    cv2.destroyAllWindows()

def show_extracted_patches_window(patches, patch_size=50):
    """创建单独窗口显示提取的色块"""
    print("\n=== Extracted Patches Display ===")
    print(f"Showing {len(patches)} extracted patches with size: {patch_size}x{patch_size}")
    print("Please check if the patches are pure colors:")
    print("Press 'Enter' to continue with calibration")
    print("Press 'Esc' to cancel")
    
    # 24色卡通常是4x6排列
    rows, cols = 4, 6
    
    # 计算显示窗口大小
    display_width = cols * patch_size
    display_height = rows * patch_size
    
    # 创建显示图像
    display_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    print(f"Display window size: {display_width}x{display_height}")
    
    # 绘制提取的色块
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            
            if idx < len(patches):
                # 计算色块位置
                patch_x = j * patch_size
                patch_y = i * patch_size
                
                # 获取色块颜色（BGR格式用于OpenCV显示）
                patch_color = patches[idx]
                if len(patch_color) == 3:
                    # 确保颜色值在有效范围内
                    bgr_color = (
                        int(np.clip(patch_color[2], 0, 255)),  # B
                        int(np.clip(patch_color[1], 0, 255)),  # G
                        int(np.clip(patch_color[0], 0, 255))   # R
                    )
                    
                    # 填充色块区域
                    display_image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size] = bgr_color
                    
                    # 绘制边界框（白色边框）
                    cv2.rectangle(display_image, (patch_x, patch_y), 
                                 (patch_x + patch_size, patch_y + patch_size), (255, 255, 255), 2)
                    
                    # 绘制内边框（黑色边框，增强对比度）
                    cv2.rectangle(display_image, (patch_x + 1, patch_y + 1), 
                                 (patch_x + patch_size - 1, patch_y + patch_size - 1), (0, 0, 0), 1)
                    
                    # 添加编号（白色文字，黑色描边）
                    text = str(idx + 1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    
                    # 获取文字尺寸
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # 计算文字位置（左上角）
                    text_x = patch_x + 5
                    text_y = patch_y + text_height + 5
                    
                    # 绘制黑色描边
                    cv2.putText(display_image, text, (text_x - 1, text_y - 1), font, font_scale, (0, 0, 0), thickness + 1)
                    cv2.putText(display_image, text, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1)
                    cv2.putText(display_image, text, (text_x - 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1)
                    cv2.putText(display_image, text, (text_x + 1, text_y - 1), font, font_scale, (0, 0, 0), thickness + 1)
                    
                    # 绘制白色文字
                    cv2.putText(display_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # 显示提取的色块窗口
    cv2.namedWindow('Extracted Color Patches', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Extracted Color Patches', display_width + 100, display_height + 100)
    
    while True:
        cv2.imshow('Extracted Color Patches', display_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter键
            break
        elif key == 27:  # Esc键
            cv2.destroyAllWindows()
            raise Exception("User cancelled extracted patches preview")
    
    cv2.destroyAllWindows()

def extract_color_patches_direct(color_image: np.ndarray) -> np.ndarray:
    """
    从透视变换后的色卡图像中直接提取24个色块
    
    Args:
        color_image: 透视变换后的彩色图像（标准矩形）
        
    Returns:
        24个色块的RGB值
    """
    h, w = color_image.shape[:2]
    print(f"\n=== Direct Color Patch Extraction ===")
    print(f"Processing transformed color card image: {w}x{h}")
    
    # 24色卡通常是4x6排列
    rows, cols = 4, 6
    
    # 计算每个色块的大小
    patch_h = h // rows
    patch_w = w // cols
    
    print(f"Patch grid: {rows}x{cols}, patch size: {patch_w}x{patch_h}")
    
    # 提取色块
    patches = []
    patch_positions = []
    center_regions = []
    
    for i in range(rows):
        for j in range(cols):
            # 计算色块在图像中的位置
            patch_x = j * patch_w
            patch_y = i * patch_h
            
            # 计算中间1/2区域的边界（避免边界影响）
            center_start_x = patch_x + patch_w // 4
            center_end_x = patch_x + 3 * patch_w // 4
            center_start_y = patch_y + patch_h // 4
            center_end_y = patch_y + 3 * patch_h // 4
            
            # 提取中间1/2区域
            center_region = color_image[center_start_y:center_end_y, center_start_x:center_end_x]
            
            # 保存center_region
            center_regions.append(center_region)
            
            # 计算色块平均RGB值
            if center_region.size > 0:
                # 转换BGR到RGB
                patch_rgb = cv2.cvtColor(center_region, cv2.COLOR_BGR2RGB)
                avg_rgb = np.mean(patch_rgb.reshape(-1, 3), axis=0)
                patches.append(avg_rgb)
                patch_positions.append((patch_x, patch_y, patch_w, patch_h))
                print(f"Patch {i*cols + j + 1}: Center region {center_region.shape}, RGB: {avg_rgb}")
            else:
                patches.append([0, 0, 0])
                patch_positions.append((patch_x, patch_y, patch_w, patch_h))
                print(f"Patch {i*cols + j + 1}: Empty center region")
    
    # 显示提取的色块
    show_center_regions_window(center_regions, 50)
    
    # 显示patches位置信息
    print("\n=== Patch Positions ===")
    print("Extracted patch positions:")
    for i, (patch_x, patch_y, patch_w, patch_h) in enumerate(patch_positions):
        print(f"Patch {i+1:2d}: Position=({patch_x:4d}, {patch_y:4d}), Size=({patch_w:3d}x{patch_h:3d})")
    
    # 打印色块RGB值
    print("\nExtracted color patch RGB values:")
    for i, patch in enumerate(patches):
        print(f"Patch {i+1:2d}: R={patch[0]:6.1f}, G={patch[1]:6.1f}, B={patch[2]:6.1f}")
    
    print(f"Successfully extracted {len(patches)} color patches")
    return np.array(patches, dtype=np.float64), patch_positions

def extract_color_patches_16bit_direct(color_image: np.ndarray) -> np.ndarray:
    """
    从透视变换后的16位色卡图像中直接提取24个色块
    
    Args:
        color_image: 透视变换后的16位彩色图像（标准矩形）
        
    Returns:
        24个色块的RGB值
    """
    h, w = color_image.shape[:2]
    print(f"\n=== Direct Color Patch Extraction (16-bit) ===")
    print(f"Processing transformed 16-bit color card image: {w}x{h}")
    
    # 24色卡通常是4x6排列
    rows, cols = 4, 6
    
    # 计算每个色块的大小
    patch_h = h // rows
    patch_w = w // cols
    
    print(f"Patch grid: {rows}x{cols}, patch size: {patch_w}x{patch_h}")
    
    # 提取色块
    patches = []
    patch_positions = []
    center_regions = []
    
    for i in range(rows):
        for j in range(cols):
            # 计算色块在图像中的位置
            patch_x = j * patch_w
            patch_y = i * patch_h
            
            # 计算中间1/2区域的边界（避免边界影响）
            center_start_x = patch_x + patch_w // 3
            center_end_x = patch_x + 2 * patch_w // 3
            center_start_y = patch_y + patch_h // 3
            center_end_y = patch_y + 2 * patch_h // 3
            
            # 提取中间1/2区域（16位）
            center_region = color_image[center_start_y:center_end_y, center_start_x:center_end_x]
            
            # 转换为8位用于显示
            center_region_8bit = (center_region.astype(np.float32) / 4095.0 * 255.0).astype(np.uint8)
            
            # 保存center_region（8位用于显示）
            center_regions.append(center_region_8bit)
            
            # 计算色块平均RGB值（保持16位精度）
            if center_region.size > 0:
                # 转换BGR到RGB
                patch_rgb = cv2.cvtColor(center_region, cv2.COLOR_BGR2RGB)
                avg_rgb = np.mean(patch_rgb.reshape(-1, 3), axis=0)
                patches.append(avg_rgb)
                patch_positions.append((patch_x, patch_y, patch_w, patch_h))
                print(f"Patch {i*cols + j + 1}: Center region {center_region.shape}, RGB: {avg_rgb}")
            else:
                patches.append([0, 0, 0])
                patch_positions.append((patch_x, patch_y, patch_w, patch_h))
                print(f"Patch {i*cols + j + 1}: Empty center region")
    
    # 显示提取的色块
    show_center_regions_window(center_regions, 50)
    
    # 显示patches位置信息
    print("\n=== Patch Positions ===")
    print("Extracted patch positions:")
    for i, (patch_x, patch_y, patch_w, patch_h) in enumerate(patch_positions):
        print(f"Patch {i+1:2d}: Position=({patch_x:4d}, {patch_y:4d}), Size=({patch_w:3d}x{patch_h:3d})")
    
    # 打印色块RGB值
    print("\nExtracted color patch RGB values (16-bit):")
    for i, patch in enumerate(patches):
        print(f"Patch {i+1:2d}: R={patch[0]:6.1f}, G={patch[1]:6.1f}, B={patch[2]:6.1f}")
    
    print(f"Successfully extracted {len(patches)} color patches from 16-bit image")
    return np.array(patches, dtype=np.float64), patch_positions


def load_and_average_colorcheck_images(input_path: str, width: int, height: int, data_type: str) -> Optional[np.ndarray]:
    """加载色卡图文件夹内所有RAW文件并平均"""
    try:
        input_path = Path(input_path)
        
        if input_path.is_file():
            # 单个文件
            print(f"Loading single colorcheck image: {input_path}")
            return load_single_raw_file(input_path, width, height, data_type)
        elif input_path.is_dir():
            # 文件夹，加载所有RAW文件并平均
            print(f"Loading colorcheck images from folder: {input_path}")
            return load_and_average_raw_files(input_path, width, height, data_type)
        else:
            print(f"Colorcheck path not found: {input_path}")
            return None
            
    except Exception as e:
        print(f"Error loading colorcheck images: {e}")
        return None

def process_colorcheck_image(input_image_path: str, config: Dict) -> Dict:
    """
    处理色卡图像并计算CCM
    
    Args:
        input_image_path: 输入图像路径
        config: 配置参数
        
    Returns:
        处理结果字典
    """
    print("=== CCM Endoscope Calibration ===")
    print(f"Input image: {input_image_path}")
    print(f"Image dimensions: {config['IMAGE_WIDTH']} x {config['IMAGE_HEIGHT']}")
    print(f"Data type: {config['DATA_TYPE']}")
    print(f"CCM method: {config['CCM_METHOD']}")
    
    # 创建输出目录
    if config['OUTPUT_DIRECTORY'] is None:
        output_dir = Path(input_image_path).parent / f"ccm_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(config['OUTPUT_DIRECTORY'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    try:
        # 1. 加载色卡图数据（支持文件夹平均）
        print("\n=== Loading Colorcheck Images ===")
        colorcheck_raw_data = load_and_average_colorcheck_images(
            input_image_path, 
            config['IMAGE_WIDTH'], 
            config['IMAGE_HEIGHT'], 
            config['DATA_TYPE']
        )
        
        if colorcheck_raw_data is None:
            raise Exception("Failed to load colorcheck images")
        
        # 2. 使用ISP流程处理图像
        print("\n=== ISP Processing ===")
        
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
        
        # 加载白平衡参数
        wb_params = None
        if config['WHITE_BALANCE_ENABLED'] and Path(config['WB_PARAMETERS_PATH']).exists():
            print("Loading white balance parameters...")
            from ISP import load_white_balance_parameters
            wb_params = load_white_balance_parameters(config['WB_PARAMETERS_PATH'])
            print("White balance parameters loaded")
        else:
            print("No white balance correction applied")
        
        # 将平均后的数据保存为临时文件
        temp_raw_file = output_dir / "temp_averaged_colorcheck.raw"
        colorcheck_raw_data.flatten().tofile(str(temp_raw_file))
        print(f"Saved averaged colorcheck data to temporary file: {temp_raw_file}")
        
        # 使用ISP处理
        isp_result = process_single_image(
            raw_file=str(temp_raw_file),
            dark_data=dark_data,
            lens_shading_params=lens_shading_params,
            width=config['IMAGE_WIDTH'],
            height=config['IMAGE_HEIGHT'],
            data_type=config['DATA_TYPE'],
            wb_params=wb_params,
            ccm_enabled=False,
            gamma_correction_enabled=False,
        )
        
        if not isp_result['processing_success']:
            raise Exception("ISP processing failed")
        
        print("ISP processing completed successfully")
        
        # 3. 照明调整（如果启用）
        illumination_correction_grid = None
        if config['ILLUMINATION_CORRECTION_ENABLED'] and config.get('ILLUMINATION_IMAGE_PATH'):
            print("\n=== Illumination Correction ===")
            illumination_path = config['ILLUMINATION_IMAGE_PATH']
            
            # 加载照明图数据（支持文件夹平均）
            print(f"Loading illumination images: {illumination_path}")
            illumination_raw_data = load_illumination_image(
                illumination_path, 
                config['IMAGE_WIDTH'], 
                config['IMAGE_HEIGHT'], 
                config['DATA_TYPE']
            )
            
            if illumination_raw_data is not None:
                # 将平均后的照明数据保存为临时文件
                temp_illumination_file = output_dir / "temp_averaged_illumination.raw"
                illumination_raw_data.flatten().tofile(str(temp_illumination_file))
                print(f"Saved averaged illumination data to temporary file: {temp_illumination_file}")
                
                # 处理照明图
                print(f"Processing illumination image: {temp_illumination_file}")
                illumination_isp_result = process_single_image(
                    raw_file=str(temp_illumination_file),
                    dark_data=dark_data,
                    lens_shading_params=lens_shading_params,
                    width=config['IMAGE_WIDTH'],
                    height=config['IMAGE_HEIGHT'],
                    data_type=config['DATA_TYPE'],
                    wb_params=wb_params
                )
            else:
                print("Failed to load illumination images")
                illumination_isp_result = {'processing_success': False}
            
            if not illumination_isp_result['processing_success']:
                print("Illumination image ISP processing failed, skipping illumination correction")
                illumination_correction_grid = None
            else:
                # 获取16位彩色图像并转换为灰度图
                if illumination_isp_result['color_img_16bit'] is not None:
                    processed_illumination = illumination_isp_result['color_img_16bit']
                    print(f"Illumination image processed successfully: {processed_illumination.shape}")
                else:
                    print("No 16-bit color image available from illumination processing")
                    illumination_correction_grid = None
                
                # 创建照明调整网格
                if processed_illumination is not None:
                    illumination_correction_grid = create_illumination_correction_grid(
                        processed_illumination, 
                        config
                    )
                else:
                    illumination_correction_grid = None
                
                # 保存照明调整网格图像（用于调试）
                if config['SAVE_IMAGES'] and illumination_correction_grid is not None:
                    # 创建调整网格的可视化
                    grid_vis = (illumination_correction_grid - 1.0) * 50 + 128  # 将调整系数转换为可视化图像
                    grid_vis = np.clip(grid_vis, 0, 255).astype(np.uint8)
                    
                    # 放大网格图像以便查看
                    grid_vis_large = cv2.resize(grid_vis, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
                    
                    illumination_grid_path = output_dir / f'illumination_correction_grid_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                    cv2.imwrite(str(illumination_grid_path), grid_vis_large)
                    print(f"Illumination correction grid saved: {illumination_grid_path}")
        else:
            print("Illumination correction disabled or no illumination image provided")
        

        # 4. 应用照明调整（如果启用）
        illumination_corrected_8bit = None
        illumination_before_8bit = None
        
        if illumination_correction_grid is not None:
            # 保存照明矫正前的8位图像
            if isp_result['color_img'] is not None:
                illumination_before_8bit = isp_result['color_img'].copy()
                print("Saved pre-illumination correction 8-bit image for comparison")
            print("\n=== Applying Illumination Correction ===")
            if isp_result['color_img_16bit'] is not None:
                print("Applying illumination correction to 16-bit color image...")
                corrected_16bit = apply_illumination_correction(
                    isp_result['color_img_16bit'], 
                    illumination_correction_grid, 
                    config['ILLUMINATION_GRID_SIZE']
                )
                isp_result['color_img_16bit'] = corrected_16bit
                print("16-bit illumination correction applied")
                
                # 保存照明矫正后的8位图像用于显示
                illumination_corrected_8bit = (corrected_16bit/4095.0*255).astype(np.uint8)
                isp_result['color_img'] = illumination_corrected_8bit
                print("8-bit illumination correction applied")
        else:
            print("No illumination correction applied")


        # 5. 色卡透视变换校正
        if config['ENABLE_ANGLE_CORRECTION']:
            print("\n=== Color Card Perspective Transform ===")
            # 使用8位彩色图像进行透视变换（更快的处理速度）
            if isp_result['color_img'] is not None:
                color_image = isp_result['color_img']
                print(f"Using 8-bit color image for transform: {color_image.shape}")
                
                # 选择色卡4个顶点
                corners = select_colorcard_corners(color_image)
                
                # 计算透视变换矩阵
                transform_matrix, target_size = calculate_perspective_transform(corners)
                
                # 显示透视变换预览
                while True:
                    preview_result = preview_perspective_transform(color_image, corners, transform_matrix, target_size)
                    
                    if preview_result == 'confirm':
                        # 用户确认变换
                        print("User confirmed perspective transform, applying correction...")
                        
                        # 应用透视变换到8位图像
                        color_image_transformed = apply_perspective_transform(color_image, transform_matrix, target_size)
                        
                        # 如果有16位图像，也进行透视变换
                        if isp_result['color_img_16bit'] is not None:
                            color_image_16bit_transformed = apply_perspective_transform(isp_result['color_img_16bit'], transform_matrix, target_size)
                            print(f"16-bit image also transformed: {color_image_16bit_transformed.shape}")
                        else:
                            color_image_16bit_transformed = None
                        
                        # 更新ISP结果
                        isp_result['color_img'] = color_image_transformed
                        isp_result['color_img_16bit'] = color_image_16bit_transformed
                        
                        print("Color card perspective transform completed")
                        break
                        
                    elif preview_result == 'recalculate':
                        # 用户要求重新选择顶点
                        print("User requested corner reselection...")
                        corners = select_colorcard_corners(color_image)
                        transform_matrix, target_size = calculate_perspective_transform(corners)
                        
                    elif preview_result == 'cancel':
                        # 用户取消变换
                        print("User cancelled perspective transform, skipping correction...")
                        break
            else:
                print("No color image available for transform")
        else:
            print("\n=== Color Card Perspective Transform ===")
            print("Perspective transform disabled in configuration")
        

        
        # 6. 提取色块数据
        print("\n=== Color Patch Extraction ===")
        # 使用16位彩色图像进行色块提取，保持更高精度
        if isp_result['color_img_16bit'] is not None:
            color_image_16bit = isp_result['color_img_16bit']
            print(f"Using 16-bit color image for patch extraction: {color_image_16bit.shape}")
            print(f"16-bit color image range: {np.min(color_image_16bit)}-{np.max(color_image_16bit)}")
            
            # 提取24个色块（使用16位图像）
            measured_patches, patch_positions = extract_color_patches_16bit_direct(color_image_16bit)
        else:
            color_image = isp_result['color_img']
            print(f"Using 8-bit color image for patch extraction: {color_image.shape}")
            
            # 提取24个色块（使用8位图像）
            measured_patches, patch_positions = extract_color_patches_direct(color_image)
        
        print(f"Extracted {len(measured_patches)} color patches")
        print(f"Measured patches shape: {measured_patches.shape}")
        
        # 7. 16位数据归一化到浮点范围(0, 255)
        print("\n=== Data Normalization ===")
        measured_patches_normalized = normalize_16bit_to_float(measured_patches, config['DATA_TYPE'])
        print(f"Normalized patches range: {np.min(measured_patches_normalized):.2f} - {np.max(measured_patches_normalized):.2f}")
        

        #准备参考数据
        reference_patches = SRGB_24PATCH_D65_8BIT.copy()
        # 参考数据是非线性的sRGB，需要线性化
        reference_linear = (reference_patches / 255.0).astype(np.float64)  # 归一化到[0,1]
        reference_patches = reference_linear

        # 参考数据是sRGB，需要线性化
        print("Linearizing reference data (sRGB to linear RGB)...")
        reference_linear = srgb_to_linear(reference_linear)

        # 5. 亮度调整（基于第19块）
        print("\n=== Luminance Adjustment ===")
        measured_patches_adjusted = adjust_luminance_with_patch19(measured_patches_normalized, reference_linear*255)
        print(f"Luminance adjusted patches range: {np.min(measured_patches_adjusted):.2f} - {np.max(measured_patches_adjusted):.2f}")
        
        # 6. CCM计算
        print("\n=== CCM Calculation ===")
        
        # 计算CCM
        if config['CCM_METHOD'] == 'gradient_optimization':
            print("Using gradient optimization method...")
            # 调用ccm_calculator.py中的compute_ccm_from_bgr函数
            # 但我们需要先转换数据格式
            
            # 将测量和参考数据转换为线性RGB
            # 测量数据是线性的，不需要线性化
            measured_linear = (measured_patches_adjusted / 255.0).astype(np.float64)  # 归一化到[0,1]

            
            print(f"Data types - measured_linear: {measured_linear.dtype}, reference_linear: {reference_linear.dtype}")
            print(f"Data shapes - measured_linear: {measured_linear.shape}, reference_linear: {reference_linear.shape}")
            print(f"Data ranges - measured_linear: [{np.min(measured_linear):.3f}, {np.max(measured_linear):.3f}], reference_linear: [{np.min(reference_linear):.3f}, {np.max(reference_linear):.3f}]")
            

            print(f"After sRGB linearization - reference_linear: {reference_linear.dtype}, range: [{np.min(reference_linear):.3f}, {np.max(reference_linear):.3f}]")
            
            # 创建CCM配置
            from ccm_calculator import CCMSolveConfig
            ccm_config = CCMSolveConfig(
                model='linear3x3',
                lambda_reg=0.0,
                regularize_bias=False,
                use_gradient_optimization=True,
                max_iterations=1000,
                tolerance=1e-6,
                preserve_white=config['WHITE_PRESERVATION_CONSTRAINT']
            )
            
            # 计算CCM之前的DeltaE误差
            print("Calculating DeltaE error before CCM correction...")
            measured_xyz = rgb_linear_to_xyz(measured_linear)
            reference_xyz = rgb_linear_to_xyz(reference_linear)
            
            measured_lab = xyz_to_lab(measured_xyz)
            reference_lab = xyz_to_lab(reference_xyz)
            
            # 计算CCM之前的DeltaE
            from ccm_calculator import delta_e_cie76
            delta_e_before = delta_e_cie76(measured_lab, reference_lab)
            delta_e_before_mean = float(np.mean(delta_e_before))
            delta_e_before_median = float(np.median(delta_e_before))
            delta_e_before_max = float(np.max(delta_e_before))
            
            print(f"DeltaE before CCM correction:")
            print(f"  Mean: {delta_e_before_mean:.3f}")
            print(f"  Median: {delta_e_before_median:.3f}")
            print(f"  Max: {delta_e_before_max:.3f}")
            
            # 迭代优化：每次移除误差最大的色块，直到均值<10或仅剩3块
            print("Starting iterative CCM optimization with outlier removal...")
            kept_indices = list(range(measured_linear.shape[0]))
            removed_indices = []
            iter_history = []
            best_state = None  # 记录最佳(mean最小)状态
            target_mean = 10.0
            max_iters = max(1, len(kept_indices) - 3)

            def _apply_ccm(ml_arr: np.ndarray, ccm_arr: np.ndarray) -> np.ndarray:
                if ccm_arr.shape == (3, 3):
                    out = ml_arr @ ccm_arr.T
                elif ccm_arr.shape == (3, 4):
                    ones = np.ones((ml_arr.shape[0], 1), dtype=ml_arr.dtype)
                    X = np.concatenate([ml_arr, ones], axis=1)
                    out = X @ ccm_arr.T
                elif ccm_arr.shape == (4, 3):
                    ones = np.ones((ml_arr.shape[0], 1), dtype=ml_arr.dtype)
                    X = np.concatenate([ml_arr, ones], axis=1)
                    out = X @ ccm_arr
                else:
                    raise ValueError(f"Unsupported CCM shape: {ccm_arr.shape}")
                return np.clip(out, 0.0, 1.0)

            for it in range(max_iters):
                try:
                    ml = measured_linear[kept_indices]
                    rl = reference_linear[kept_indices]

                    if ml.size == 0 or rl.size == 0 or ml.shape[0] != rl.shape[0] or ml.shape[1] != 3 or rl.shape[1] != 3:
                        raise ValueError(f"Invalid patch arrays: ml{ml.shape}, rl{rl.shape}")
                    if not np.isfinite(ml).all() or not np.isfinite(rl).all():
                        raise ValueError("NaN/Inf detected in input patches")

                    # 拟合本轮CCM（提高第15块权重，索引14）
                w15 = int(max(1, CONFIG.get('PATCH15_WEIGHT', 1))) if 'CONFIG' in globals() else 1
                if w15 > 1 and len(kept_indices) >= 15 and 14 in kept_indices:
                    local15 = kept_indices.index(14)
                    rep_m = np.repeat(ml[local15:local15+1, :], w15-1, axis=0)
                    rep_r = np.repeat(rl[local15:local15+1, :], w15-1, axis=0)
                    ml_fit = np.concatenate([ml, rep_m], axis=0)
                    rl_fit = np.concatenate([rl, rep_r], axis=0)
                else:
                    ml_fit = ml
                    rl_fit = rl

                    ccm_matrix_it = solve_ccm_gradient_optimization(ml_fit, rl_fit, ccm_config)

                    # 应用并评估
                    corrected_it = _apply_ccm(ml, ccm_matrix_it)
                    corr_xyz = rgb_linear_to_xyz(corrected_it)
                    corr_lab = xyz_to_lab(corr_xyz)
                    ref_lab = xyz_to_lab(rgb_linear_to_xyz(rl))
                    de_it = delta_e_cie76(corr_lab, ref_lab)
                    mean_de_it = float(np.mean(de_it))

                    iter_history.append({'iter': it, 'mean': mean_de_it, 'num_patches': len(kept_indices)})
                    print(f"  Iter {it}: mean DeltaE = {mean_de_it:.3f}, patches = {len(kept_indices)}")

                    # 记录最佳
                    if best_state is None or mean_de_it < best_state['mean']:
                        best_state = {
                            'mean': mean_de_it,
                            'ccm': ccm_matrix_it,
                            'kept': kept_indices.copy(),
                            'de': de_it.copy(),
                            'corrected_linear': corrected_it.copy(),
                        }

                    # 满足阈值或无法再移除
                    if mean_de_it <= target_mean or len(kept_indices) <= 3:
                        break

                    # 移除本轮误差最大的一个色块
                    worst_local = int(np.argmax(de_it))
                    worst_global = kept_indices[worst_local]
                    removed_indices.append(worst_global)
                    kept_indices.pop(worst_local)
                except Exception as e:
                    import traceback
                    print(f"  Iter {it} failed: {e}")
                    traceback.print_exc()
                    break

            # 使用最佳结果作为最终结果，避免退化
            ccm_matrix = best_state['ccm']
            corrected_linear = best_state['corrected_linear']
            delta_e_after_mean = best_state['mean']
            # 重新基于最佳纠正值计算完整统计
            corrected_xyz = rgb_linear_to_xyz(corrected_linear)
            corrected_lab = xyz_to_lab(corrected_xyz)
            reference_lab_kept = xyz_to_lab(rgb_linear_to_xyz(reference_linear[best_state['kept']]))
            delta_e_after = delta_e_cie76(corrected_lab, reference_lab_kept)
            delta_e_after_median = float(np.median(delta_e_after))
            delta_e_after_max = float(np.max(delta_e_after))
            
            print(f"DeltaE after CCM correction:")
            print(f"  Mean: {delta_e_after_mean:.3f}")
            print(f"  Median: {delta_e_after_median:.3f}")
            print(f"  Max: {delta_e_after_max:.3f}")
            
            # 计算改善程度
            improvement_mean = delta_e_before_mean - delta_e_after_mean
            improvement_percent = (improvement_mean / delta_e_before_mean) * 100 if delta_e_before_mean > 0 else 0
            
            print(f"DeltaE improvement:")
            print(f"  Mean improvement: {improvement_mean:.3f} ({improvement_percent:.1f}%)")
            
            # 构建结果字典
            ccm_result = {
                'ccm_matrix': ccm_matrix,
                'ccm_type': 'linear3x3',
                'delta_e_before': {
                    'mean': delta_e_before_mean,
                    'median': delta_e_before_median,
                    'max': delta_e_before_max,
                    'errors': delta_e_before.tolist()
                },
                'delta_e_after': {
                    'mean': delta_e_after_mean,
                    'median': delta_e_after_median,
                    'max': delta_e_after_max,
                    'errors': delta_e_after.tolist()
                },
                'delta_e_improvement': {
                    'mean_improvement': improvement_mean,
                    'improvement_percent': improvement_percent
                },
                'delta_e_error': delta_e_after_mean,  # 保持向后兼容性
                'delta_e_errors': delta_e_after.tolist(),  # 保持向后兼容性
                'luminance_normalization': config['LUMINANCE_NORMALIZATION'],
                'white_preservation_constraint': config['WHITE_PRESERVATION_CONSTRAINT'],
            }
            
        else:
            raise ValueError(f"Unsupported CCM method: {config['CCM_METHOD']}")
        
        print(f"CCM calculation completed")
        print(f"CCM matrix shape: {ccm_result['ccm_matrix'].shape}")
        print(f"DeltaE error (after CCM): {ccm_result['delta_e_error']:.3f}")
        
        # 显示详细的DeltaE统计信息
        if 'delta_e_before' in ccm_result:
            print(f"\n=== DeltaE Analysis Summary ===")
            print(f"Before CCM correction:")
            print(f"  Mean: {ccm_result['delta_e_before']['mean']:.3f}")
            print(f"  Median: {ccm_result['delta_e_before']['median']:.3f}")
            print(f"  Max: {ccm_result['delta_e_before']['max']:.3f}")
            print(f"After CCM correction:")
            print(f"  Mean: {ccm_result['delta_e_after']['mean']:.3f}")
            print(f"  Median: {ccm_result['delta_e_after']['median']:.3f}")
            print(f"  Max: {ccm_result['delta_e_after']['max']:.3f}")
            print(f"Improvement:")
            print(f"  Mean improvement: {ccm_result['delta_e_improvement']['mean_improvement']:.3f}")
            print(f"  Improvement percentage: {ccm_result['delta_e_improvement']['improvement_percent']:.1f}%")
        
        # 6.5. 应用CCM矫正到完整图像
        print("\n=== Applying CCM Correction to Full Image ===")
        ccm_corrected_image = apply_ccm_to_image(isp_result, ccm_result['ccm_matrix'])
        
        # 7. 保存结果
        if config['SAVE_RESULTS']:
            print("\n=== Saving Results ===")
            
            # 保存CCM参数
            if config['SAVE_PARAMETERS']:
                save_ccm_parameters(ccm_result, output_dir)
            
            # 保存图像
            if config['SAVE_IMAGES']:
                save_ccm_images(isp_result, measured_patches_adjusted, reference_patches, output_dir, patch_positions, ccm_corrected_image, illumination_before_8bit, illumination_corrected_8bit)
            
            # 生成图表
            if config['GENERATE_PLOTS']:
                create_ccm_analysis_plots(measured_patches_adjusted, reference_patches*255, ccm_result, output_dir)
        
        # 返回结果
        result = {
            'success': True,
            'ccm_result': ccm_result,
            'measured_patches': measured_patches_adjusted,
            'reference_patches': reference_patches,
            'isp_result': isp_result,
            'output_directory': str(output_dir)
        }
        
        print("\n=== CCM Calibration Complete ===")
        
        # 清理临时文件
        temp_files = [
            output_dir / "temp_averaged_colorcheck.raw",
            output_dir / "temp_averaged_illumination.raw"
        ]
        
        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    print(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_file}: {e}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        
        # 清理临时文件（即使出错也要清理）
        temp_files = [
            output_dir / "temp_averaged_colorcheck.raw",
            output_dir / "temp_averaged_illumination.raw"
        ]
        
        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    print(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_file}: {e}")
        
        return {
            'success': False,
            'error': str(e),
            'output_directory': str(output_dir)
        }

def save_ccm_parameters(ccm_result: Dict, output_dir: Path) -> Path:
    """
    保存CCM参数
    
    Args:
        ccm_result: CCM计算结果
        output_dir: 输出目录
        
    Returns:
        参数文件路径
    """
    parameters = {
        'ccm_matrix': ccm_result['ccm_matrix'].tolist(),
        'ccm_type': ccm_result['ccm_type'],
        'delta_e_error': float(ccm_result['delta_e_error']),
        'white_balance_gains': ccm_result.get('wb_gains', {}),
        'luminance_normalization': ccm_result.get('luminance_normalization', False),
        'white_preservation_constraint': ccm_result.get('white_preservation_constraint', False),
        'timestamp': datetime.now().isoformat(),
        'description': 'CCM calibration parameters calculated using gradient optimization'
    }
    
    # 保存为JSON文件
    params_path = output_dir / f'ccm_parameters_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(parameters, f, indent=2, ensure_ascii=False)
    
    print(f"CCM parameters saved: {params_path}")
    return params_path

def save_ccm_images(isp_result: Dict, measured_patches: np.ndarray, reference_patches: np.ndarray, output_dir: Path, patch_positions: list = None, ccm_corrected_image: Dict = None, illumination_before_8bit: np.ndarray = None, illumination_corrected_8bit: np.ndarray = None) -> None:
    """
    保存CCM相关图像
    
    Args:
        isp_result: ISP处理结果
        measured_patches: 测量的色块数据
        reference_patches: 参考色块数据
        output_dir: 输出目录
        patch_positions: 色块位置信息
        ccm_corrected_image: CCM矫正后的图像
        illumination_before_8bit: 照明矫正前的8位图像
        illumination_corrected_8bit: 照明矫正后的8位图像
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存ISP处理后的图像
    if isp_result['color_img'] is not None:
        color_img_path = output_dir / f'colorcheck_processed_{timestamp}.png'
        cv2.imwrite(str(color_img_path), isp_result['color_img'])
        print(f"Colorcheck image saved: {color_img_path}")
        
        # 如果有色块位置信息，保存带色块框的图像
        if patch_positions is not None:
            color_img_with_patches = isp_result['color_img'].copy()
            
            # 绘制色块边界和编号
            for idx, (patch_x, patch_y, patch_w, patch_h) in enumerate(patch_positions):
                # 绘制边界框（白色边框）
                cv2.rectangle(color_img_with_patches, (patch_x, patch_y), 
                             (patch_x + patch_w, patch_y + patch_h), (255, 255, 255), 2)
                
                # 绘制内边框（黑色边框，增强对比度）
                cv2.rectangle(color_img_with_patches, (patch_x + 1, patch_y + 1), 
                             (patch_x + patch_w - 1, patch_y + patch_h - 1), (0, 0, 0), 1)
                
                # 添加编号（白色文字，黑色描边）
                text = str(idx + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                
                # 获取文字尺寸
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # 计算文字位置（左上角）
                text_x = patch_x + 5
                text_y = patch_y + text_height + 5
                
                # 绘制黑色描边
                cv2.putText(color_img_with_patches, text, (text_x - 1, text_y - 1), font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(color_img_with_patches, text, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(color_img_with_patches, text, (text_x - 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(color_img_with_patches, text, (text_x + 1, text_y - 1), font, font_scale, (0, 0, 0), thickness + 1)
                
                # 绘制白色文字
                cv2.putText(color_img_with_patches, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                
                # 绘制位置信息
                draw_patch_position_info(color_img_with_patches, patch_x, patch_y, patch_w, patch_h)
            
            # 保存带色块框的图像
            color_img_with_patches_path = output_dir / f'colorcheck_with_patches_{timestamp}.png'
            cv2.imwrite(str(color_img_with_patches_path), color_img_with_patches)
            print(f"Colorcheck image with patches saved: {color_img_with_patches_path}")
    
    # 保存CCM矫正后的图像
    if ccm_corrected_image is not None and ccm_corrected_image.get('success', False):
        print("Saving CCM corrected images...")
        
        # 保存8位CCM矫正图像
        if ccm_corrected_image['ccm_corrected_8bit'] is not None:
            ccm_8bit = ccm_corrected_image['ccm_corrected_8bit']
            ccm_8bit_path = output_dir / f'colorcheck_ccm_corrected_8bit_{timestamp}.png'
            cv2.imwrite(str(ccm_8bit_path), ccm_8bit)
            print(f"CCM corrected 8-bit image saved: {ccm_8bit_path}")
            
            # 另存一份gamma矫正后的8位图
            try:
                gamma = 2.2
                ccm_8bit_float = np.clip(ccm_8bit.astype(np.float32) / 255.0, 0.0, 1.0)
                ccm_8bit_gamma = np.power(ccm_8bit_float, 1.0 / gamma)
                ccm_8bit_gamma_u8 = np.clip(ccm_8bit_gamma * 255.0, 0, 255).astype(np.uint8)
                ccm_8bit_gamma_path = output_dir / f'colorcheck_ccm_corrected_8bit_gamma_{timestamp}.png'
                cv2.imwrite(str(ccm_8bit_gamma_path), ccm_8bit_gamma_u8)
                print(f"CCM corrected 8-bit (gamma) image saved: {ccm_8bit_gamma_path}")
            except Exception as e:
                print(f"Warning: failed to save gamma-corrected 8-bit image: {e}")
        
        # 保存16位CCM矫正图像
        if ccm_corrected_image['ccm_corrected_16bit'] is not None:
            ccm_16bit_path = output_dir / f'colorcheck_ccm_corrected_16bit_{timestamp}.png'
            cv2.imwrite(str(ccm_16bit_path), ccm_corrected_image['ccm_corrected_16bit'])
            print(f"CCM corrected 16-bit image saved: {ccm_16bit_path}")
            
            # 保存16位CCM矫正RAW数据
            ccm_raw_path = output_dir / f'colorcheck_ccm_corrected_16bit_{timestamp}.raw'
            with open(ccm_raw_path, 'wb') as f:
                ccm_corrected_image['ccm_corrected_16bit'].tofile(f)
            print(f"CCM corrected 16-bit RAW saved: {ccm_raw_path}")
    else:
        print("No CCM corrected images to save")
    
    # 保存照明矫正对比图（如果启用）
    if illumination_before_8bit is not None and illumination_corrected_8bit is not None:
        create_illumination_correction_comparison(illumination_before_8bit, illumination_corrected_8bit, output_dir, timestamp)
        # 显示照明矫正对比图
        show_illumination_correction_plot(illumination_before_8bit, illumination_corrected_8bit)
    
    # 保存CCM矫正前后对比图
    if ccm_corrected_image is not None and ccm_corrected_image.get('success', False):
        create_ccm_before_after_comparison(isp_result, ccm_corrected_image, output_dir, timestamp)
        # 显示CCM矫正前后对比图
        show_ccm_before_after_plot(isp_result, ccm_corrected_image)
    else:
        print("Warning: No CCM corrected image available for comparison")
    
    # 保存色块对比图
    create_patch_comparison_image(measured_patches, reference_patches, output_dir, timestamp)

def show_ccm_before_after_plot(isp_result: Dict, ccm_corrected_image: Dict) -> None:
    """
    使用matplotlib显示CCM矫正前后对比图
    
    Args:
        isp_result: ISP处理结果
        ccm_corrected_image: CCM矫正后的图像
    """
    try:
        print("\n=== Displaying CCM Before/After Comparison ===")
        
        # 检查是否有必要的数据
        if not ccm_corrected_image.get('success', False):
            print("No CCM corrected image available for display")
            return
        
        # 获取矫正前的图像（ISP处理后的8位图像）
        before_image = isp_result.get('color_img')
        if before_image is None:
            print("No before image (ISP processed) available for display")
            return
        
        # 获取矫正后的图像（8位）
        after_image = ccm_corrected_image.get('ccm_corrected_8bit')
        if after_image is None:
            print("No after image (CCM corrected) available for display")
            return
        
        print(f"Displaying before image shape: {before_image.shape}")
        print(f"Displaying after image shape: {after_image.shape}")
        
        # 确保两个图像尺寸相同
        if before_image.shape != after_image.shape:
            print("Warning: Before and after images have different shapes")
            min_h = min(before_image.shape[0], after_image.shape[0])
            min_w = min(before_image.shape[1], after_image.shape[1])
            before_image = before_image[:min_h, :min_w]
            after_image = after_image[:min_h, :min_w]
            print(f"Resized to: {before_image.shape}")
        
        # 手动转换BGR到RGB用于matplotlib显示（不使用cv2.cvtColor）
        before_rgb = before_image[:, :, [2, 1, 0]]  # BGR -> RGB: 交换通道0和2
        after_rgb = after_image[:, :, [2, 1, 0]]    # BGR -> RGB: 交换通道0和2
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle('CCM Correction Comparison', fontsize=16, fontweight='bold')
        
        # 显示矫正前的图像
        axes[0].imshow(before_rgb)
        axes[0].set_title('Before CCM (ISP Processed)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 显示矫正后的图像
        axes[1].imshow(after_rgb)
        axes[1].set_title('After CCM (Color Corrected)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图像
        plt.show()
        
        print("CCM before/after comparison displayed successfully")
        
    except Exception as e:
        print(f"Error displaying CCM before/after comparison: {e}")
        import traceback
        traceback.print_exc()

def create_ccm_before_after_comparison(isp_result: Dict, ccm_corrected_image: Dict, output_dir: Path, timestamp: str) -> None:
    """
    创建CCM矫正前后对比图像
    
    Args:
        isp_result: ISP处理结果
        ccm_corrected_image: CCM矫正后的图像
        output_dir: 输出目录
        timestamp: 时间戳
    """
    try:
        print("\n=== Creating CCM Before/After Comparison ===")
        print(f"Output directory: {output_dir}")
        print(f"Timestamp: {timestamp}")
        
        # 检查是否有必要的数据
        if not ccm_corrected_image.get('success', False):
            print("No CCM corrected image available for comparison")
            return
        
        # 获取矫正前的图像（ISP处理后的8位图像）
        before_image = isp_result.get('color_img')
        if before_image is None:
            print("No before image (ISP processed) available for comparison")
            return
        
        # 获取矫正后的图像（8位）
        after_image = ccm_corrected_image.get('ccm_corrected_8bit')
        if after_image is None:
            print("No after image (CCM corrected) available for comparison")
            return
        
        print(f"Before image shape: {before_image.shape}")
        print(f"After image shape: {after_image.shape}")
        
        # 确保两个图像尺寸相同
        if before_image.shape != after_image.shape:
            print("Warning: Before and after images have different shapes")
            # 调整尺寸到相同大小（使用较小的尺寸）
            min_h = min(before_image.shape[0], after_image.shape[0])
            min_w = min(before_image.shape[1], after_image.shape[1])
            before_image = before_image[:min_h, :min_w]
            after_image = after_image[:min_h, :min_w]
            print(f"Resized to: {before_image.shape}")
        
        # 创建水平拼接的对比图
        h, w = before_image.shape[:2]
        
        # 创建对比图像（左右拼接）
        comparison_width = w * 2 + 40  # 中间留40像素间隔
        comparison_height = h + 80     # 上下留空间添加标题
        comparison_image = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)
        
        # 设置背景为深灰色
        comparison_image[:] = (50, 50, 50)
        
        # 放置矫正前的图像（左侧）
        y_offset = 60  # 为标题留空间
        comparison_image[y_offset:y_offset+h, 0:w] = before_image
        
        # 放置矫正后的图像（右侧）
        comparison_image[y_offset:y_offset+h, w+40:w*2+40] = after_image
        
        # 添加标题文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        
        # 主标题
        main_title = "CCM Correction Comparison"
        (title_width, title_height), _ = cv2.getTextSize(main_title, font, font_scale, thickness)
        title_x = (comparison_width - title_width) // 2
        title_y = 30
        
        # 绘制主标题（白色文字，黑色描边）
        cv2.putText(comparison_image, main_title, (title_x-1, title_y-1), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(comparison_image, main_title, (title_x+1, title_y+1), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(comparison_image, main_title, (title_x, title_y), font, font_scale, (255, 255, 255), thickness)
        
        # 子标题
        font_scale_sub = 0.8
        thickness_sub = 2
        
        # "Before CCM"标题
        before_title = "Before CCM (ISP Processed)"
        (before_width, before_height), _ = cv2.getTextSize(before_title, font, font_scale_sub, thickness_sub)
        before_x = (w - before_width) // 2
        before_y = y_offset - 10
        
        cv2.putText(comparison_image, before_title, (before_x-1, before_y-1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
        cv2.putText(comparison_image, before_title, (before_x+1, before_y+1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
        cv2.putText(comparison_image, before_title, (before_x, before_y), font, font_scale_sub, (255, 255, 255), thickness_sub)
        
        # "After CCM"标题
        after_title = "After CCM (Color Corrected)"
        (after_width, after_height), _ = cv2.getTextSize(after_title, font, font_scale_sub, thickness_sub)
        after_x = w + 40 + (w - after_width) // 2
        after_y = y_offset - 10
        
        cv2.putText(comparison_image, after_title, (after_x-1, after_y-1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
        cv2.putText(comparison_image, after_title, (after_x+1, after_y+1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
        cv2.putText(comparison_image, after_title, (after_x, after_y), font, font_scale_sub, (255, 255, 255), thickness_sub)
        
        # 添加分割线
        line_y = y_offset - 20
        cv2.line(comparison_image, (0, line_y), (comparison_width, line_y), (100, 100, 100), 2)
        
        # 保存对比图像
        comparison_path = output_dir / f'ccm_before_after_comparison_{timestamp}.png'
        print(f"Attempting to save comparison image to: {comparison_path}")
        success = cv2.imwrite(str(comparison_path), comparison_image)
        if success:
            print(f"✓ CCM before/after comparison saved: {comparison_path}")
        else:
            print(f"✗ Failed to save comparison image: {comparison_path}")
        
        # 另外保存单独的矫正前后图像（方便单独查看）
        before_path = output_dir / f'ccm_before_{timestamp}.png'
        after_path = output_dir / f'ccm_after_{timestamp}.png'
        
        print(f"Attempting to save before image to: {before_path}")
        before_success = cv2.imwrite(str(before_path), before_image)
        if before_success:
            print(f"✓ CCM before image saved: {before_path}")
        else:
            print(f"✗ Failed to save before image: {before_path}")
        
        print(f"Attempting to save after image to: {after_path}")
        after_success = cv2.imwrite(str(after_path), after_image)
        if after_success:
            print(f"✓ CCM after image saved: {after_path}")
        else:
            print(f"✗ Failed to save after image: {after_path}")
        
        # 验证文件是否真的存在
        if comparison_path.exists():
            print(f"✓ Comparison file exists: {comparison_path}")
        else:
            print(f"✗ Comparison file does not exist: {comparison_path}")
        
        if before_path.exists():
            print(f"✓ Before file exists: {before_path}")
        else:
            print(f"✗ Before file does not exist: {before_path}")
        
        if after_path.exists():
            print(f"✓ After file exists: {after_path}")
        else:
            print(f"✗ After file does not exist: {after_path}")
        
        # 注意：只处理8位数据对比，16位对比已移除
        
    except Exception as e:
        print(f"Error creating CCM before/after comparison: {e}")
        import traceback
        traceback.print_exc()



def create_patch_comparison_image(measured_patches: np.ndarray, reference_patches: np.ndarray, output_dir: Path, timestamp: str) -> None:
    """
    创建色块对比图像（仅保存，不显示）
    
    Args:
        measured_patches: 测量的色块数据
        reference_patches: 参考色块数据
        output_dir: 输出目录
        timestamp: 时间戳
    """
    # 创建色块对比图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 测量色块
    patch_size = 50
    measured_image = np.zeros((4 * patch_size, 6 * patch_size, 3), dtype=np.uint8)
    for i in range(4):
        for j in range(6):
            idx = i * 6 + j
            if idx < len(measured_patches):
                start_h = i * patch_size
                end_h = (i + 1) * patch_size
                start_w = j * patch_size
                end_w = (j + 1) * patch_size
                # measured_patches已经是RGB格式，直接使用
                measured_image[start_h:end_h, start_w:end_w] = measured_patches[idx].astype(np.uint8)
    
    axes[0].imshow(measured_image)
    axes[0].set_title('Measured Color Patches')
    axes[0].axis('off')
    
    # 参考色块
    reference_image = np.zeros((4 * patch_size, 6 * patch_size, 3), dtype=np.uint8)
    for i in range(4):
        for j in range(6):
            idx = i * 6 + j
            if idx < len(reference_patches):
                start_h = i * patch_size
                end_h = (i + 1) * patch_size
                start_w = j * patch_size
                end_w = (j + 1) * patch_size
                # reference_patches已经是RGB格式，直接使用
                reference_image[start_h:end_h, start_w:end_w] = reference_patches[idx].astype(np.uint8)
    
    axes[1].imshow(reference_image)
    axes[1].set_title('Reference Color Patches')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = output_dir / f'patch_comparison_{timestamp}.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Patch comparison image saved: {comparison_path}")
    
    plt.close()

def create_ccm_analysis_plots(measured_patches: np.ndarray, reference_patches: np.ndarray, ccm_result: Dict, output_dir: Path) -> None:
    """
    创建CCM分析图表
    
    Args:
        measured_patches: 测量的色块数据
        reference_patches: 参考色块数据
        ccm_result: CCM计算结果
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CCM Calibration Analysis', fontsize=16)
    
    # 1. 色块对比散点图
    axes[0, 0].scatter(measured_patches[:, 0], reference_patches[:, 0], c='red', alpha=0.7, label='R')
    axes[0, 0].scatter(measured_patches[:, 1], reference_patches[:, 1], c='green', alpha=0.7, label='G')
    axes[0, 0].scatter(measured_patches[:, 2], reference_patches[:, 2], c='blue', alpha=0.7, label='B')
    axes[0, 0].plot([0, 255], [0, 255], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('Measured Values')
    axes[0, 0].set_ylabel('Reference Values')
    axes[0, 0].set_title('Color Channel Correlation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. DeltaE误差分布
    delta_e_errors = ccm_result.get('delta_e_errors', [])
    if delta_e_errors:
        axes[0, 1].hist(delta_e_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('DeltaE Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('DeltaE Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. CCM矩阵可视化
    ccm_matrix = ccm_result['ccm_matrix']
    im = axes[1, 0].imshow(ccm_matrix, cmap='RdBu', aspect='auto')
    axes[1, 0].set_title('CCM Matrix')
    axes[1, 0].set_xlabel('Output Channel')
    axes[1, 0].set_ylabel('Input Channel')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. 处理摘要
    summary_text = f"CCM Calibration Summary:\n"
    summary_text += f"• Method: {ccm_result['ccm_type']}\n"
    summary_text += f"• DeltaE Error: {ccm_result['delta_e_error']:.3f}\n"
    summary_text += f"• Patches: {len(measured_patches)}\n"
    summary_text += f"• Matrix Shape: {ccm_matrix.shape}\n"
    
    if 'wb_gains' in ccm_result:
        wb_gains = ccm_result['wb_gains']
        summary_text += f"• White Balance:\n"
        summary_text += f"  R: {wb_gains.get('r_gain', 1.0):.3f}\n"
        summary_text += f"  G: {wb_gains.get('g_gain', 1.0):.3f}\n"
        summary_text += f"  B: {wb_gains.get('b_gain', 1.0):.3f}\n"
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 1].set_title('Calibration Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = output_dir / f'ccm_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"CCM analysis plot saved: {plot_path}")
    
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CCM Endoscope Calibration Program')
    parser.add_argument('--input', '-i', type=str, default=CONFIG['INPUT_IMAGE_PATH'],
                       help='Input colorcheck RAW image path')
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
    parser.add_argument('--wb-params', type=str, default=CONFIG['WB_PARAMETERS_PATH'],
                       help='White balance parameters file path')
    parser.add_argument('--method', '-m', type=str, default=CONFIG['CCM_METHOD'],
                       choices=['gradient_optimization', 'linear_regression'],
                       help='CCM calculation method')
    parser.add_argument('--no-dark', dest='dark_enabled', action='store_false', default=True,
                       help='Disable dark current correction')
    parser.add_argument('--no-lens-shading', dest='lens_shading_enabled', action='store_false', default=True,
                       help='Disable lens shading correction')
    parser.add_argument('--no-wb', dest='wb_enabled', action='store_false', default=True,
                       help='Disable white balance correction')
    parser.add_argument('--no-angle-correction', dest='angle_correction_enabled', action='store_false', default=True,
                       help='Disable perspective transform correction')
    parser.add_argument('--use-rotation', dest='use_perspective_transform', action='store_false', default=True,
                       help='Use rotation instead of perspective transform')
    
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
    config['WB_PARAMETERS_PATH'] = args.wb_params
    config['CCM_METHOD'] = args.method
    config['DARK_SUBTRACTION_ENABLED'] = args.dark_enabled
    config['LENS_SHADING_ENABLED'] = args.lens_shading_enabled
    config['WHITE_BALANCE_ENABLED'] = args.wb_enabled
    config['ENABLE_ANGLE_CORRECTION'] = args.angle_correction_enabled
    config['PERSPECTIVE_TRANSFORM'] = args.use_perspective_transform
    
    # 执行CCM标定
    result = process_colorcheck_image(args.input, config)
    
    if result['success']:
        ccm_result = result['ccm_result']
        print(f"\n=== CCM Calibration Results ===")
        print(f"CCM Matrix Shape: {ccm_result['ccm_matrix'].shape}")
        print(f"CCM Type: {ccm_result['ccm_type']}")
        
        # 显示详细的DeltaE信息
        if 'delta_e_before' in ccm_result:
            print(f"\nDeltaE Analysis:")
            print(f"  Before CCM correction:")
            print(f"    Mean: {ccm_result['delta_e_before']['mean']:.3f}")
            print(f"    Median: {ccm_result['delta_e_before']['median']:.3f}")
            print(f"    Max: {ccm_result['delta_e_before']['max']:.3f}")
            print(f"  After CCM correction:")
            print(f"    Mean: {ccm_result['delta_e_after']['mean']:.3f}")
            print(f"    Median: {ccm_result['delta_e_after']['median']:.3f}")
            print(f"    Max: {ccm_result['delta_e_after']['max']:.3f}")
            print(f"  Improvement:")
            print(f"    Mean improvement: {ccm_result['delta_e_improvement']['mean_improvement']:.3f}")
            print(f"    Improvement percentage: {ccm_result['delta_e_improvement']['improvement_percent']:.1f}%")
        else:
            print(f"DeltaE Error (after CCM): {ccm_result['delta_e_error']:.3f}")
        
        if 'wb_gains' in ccm_result:
            wb_gains = ccm_result['wb_gains']
            print(f"\nWhite Balance Gains:")
            print(f"  R: {wb_gains.get('r_gain', 1.0):.3f}")
            print(f"  G: {wb_gains.get('g_gain', 1.0):.3f}")
            print(f"  B: {wb_gains.get('b_gain', 1.0):.3f}")
        
        print(f"\nResults saved to: {result['output_directory']}")
    else:
        print(f"\nError: {result['error']}")
        return 1
    
    return 0

def apply_ccm_to_image(isp_result: Dict, ccm_matrix: np.ndarray) -> Dict:
    """
    将CCM矩阵应用到完整图像上
    
    Args:
        isp_result: ISP处理结果
        ccm_matrix: CCM矫正矩阵
        
    Returns:
        包含CCM矫正后图像的字典
    """
    print("Applying CCM correction to full image...")
    
    try:
        # 优先使用16位彩色图像，如果没有则使用8位图像
        if isp_result['color_img_16bit'] is not None:
            color_img = isp_result['color_img_16bit']
            max_value = 4095.0
            print(f"Using 16-bit color image for CCM correction: {color_img.shape}")
        elif isp_result['color_img'] is not None:
            color_img = isp_result['color_img']
            max_value = 255.0
            print(f"Using 8-bit color image for CCM correction: {color_img.shape}")
        else:
            print("No color image available for CCM correction")
            return {
                'ccm_corrected_16bit': None,
                'ccm_corrected_8bit': None,
                'success': False,
                'error': 'No color image available'
            }
        
        # 转换为float32进行CCM计算
        img_float = color_img.astype(np.float32) / max_value  # 归一化到[0,1]
        
        # 手动转换BGR到RGB（不使用cv2.cvtColor）
        img_rgb = img_float[:, :, [2, 1, 0]]  # BGR -> RGB: 交换通道0和2
        
        # 重塑为(N, 3)格式进行矩阵运算
        h, w, c = img_rgb.shape
        img_reshaped = img_rgb.reshape(-1, 3)
        
        # 应用CCM矩阵
        corrected_reshaped = np.clip(img_reshaped @ ccm_matrix.T, 0.0, 1.0)
        
        # 重塑回原始形状
        corrected_rgb = corrected_reshaped.reshape(h, w, c)
        
        # 手动转换回BGR（不使用cv2.cvtColor）
        corrected_bgr = corrected_rgb[:, :, [2, 1, 0]]  # RGB -> BGR: 交换通道0和2
        
        # 转换为8位用于显示
        corrected_8bit = (corrected_bgr * 255.0).astype(np.uint8)
        
        # 如果有16位输入，也生成16位输出
        if max_value == 4095.0:
            corrected_16bit = (corrected_bgr * 4095.0).astype(np.uint16)
            print(f"CCM correction applied successfully")
            print(f"16-bit corrected image: {corrected_16bit.shape}, range: {np.min(corrected_16bit)}-{np.max(corrected_16bit)}")
            print(f"8-bit corrected image: {corrected_8bit.shape}, range: {np.min(corrected_8bit)}-{np.max(corrected_8bit)}")
            
            return {
                'ccm_corrected_16bit': corrected_16bit,
                'ccm_corrected_8bit': corrected_8bit,
                'success': True
            }
        else:
            print(f"CCM correction applied successfully")
            print(f"8-bit corrected image: {corrected_8bit.shape}, range: {np.min(corrected_8bit)}-{np.max(corrected_8bit)}")
            
            return {
                'ccm_corrected_16bit': None,
                'ccm_corrected_8bit': corrected_8bit,
                'success': True
            }
            
    except Exception as e:
        print(f"Error applying CCM correction: {e}")
        import traceback
        traceback.print_exc()
        return {
            'ccm_corrected_16bit': None,
            'ccm_corrected_8bit': None,
            'success': False,
            'error': str(e)
        }

def adjust_luminance_with_patch19(measured_patches: np.ndarray, reference_patches: np.ndarray) -> np.ndarray:
    """
    根据第19块（白色块）的亮度与标准值的差别来调整测量值的亮度
    
    Args:
        measured_patches: 测量的色块数据 (24, 3)
        reference_patches: 参考色块数据 (24, 3)
        
    Returns:
        亮度调整后的测量色块数据
    """
    print("Adjusting luminance based on patch 19 (white patch)...")
    
    try:
        # 第19块是索引18（0-based）
        patch19_idx = 18
        
        if patch19_idx >= len(measured_patches) or patch19_idx >= len(reference_patches):
            print(f"Warning: Patch 19 index {patch19_idx} out of range, skipping luminance adjustment")
            return measured_patches
        
        # 获取第19块的测量值和参考值
        measured_patch19 = measured_patches[patch19_idx]  # RGB值
        reference_patch19 = reference_patches[patch19_idx]  # RGB值
        
        print(f"Patch 19 measured RGB: {measured_patch19}")
        print(f"Patch 19 reference RGB: {reference_patch19}")
        
        # 计算亮度（使用标准亮度公式：Y = 0.299*R + 0.587*G + 0.114*B）
        measured_luminance = 0.299 * measured_patch19[0] + 0.587 * measured_patch19[1] + 0.114 * measured_patch19[2]
        reference_luminance = 0.299 * reference_patch19[0] + 0.587 * reference_patch19[1] + 0.114 * reference_patch19[2]
        
        print(f"Patch 19 measured luminance: {measured_luminance:.2f}")
        print(f"Patch 19 reference luminance: {reference_luminance:.2f}")
        
        # 计算亮度比例
        if measured_luminance > 0:
            luminance_ratio = reference_luminance / measured_luminance
            print(f"Luminance ratio: {luminance_ratio:.3f}")
            
            # 应用亮度调整到所有色块
            adjusted_patches = measured_patches * luminance_ratio
            
            # 确保值在有效范围内
            adjusted_patches = np.clip(adjusted_patches, 0, 255)
            
            print(f"Luminance adjustment applied to all patches")
            print(f"Adjusted patch 19 RGB: {adjusted_patches[patch19_idx]}")
            
            # 验证调整后的第19块亮度
            adjusted_luminance = 0.299 * adjusted_patches[patch19_idx][0] + 0.587 * adjusted_patches[patch19_idx][1] + 0.114 * adjusted_patches[patch19_idx][2]
            print(f"Adjusted patch 19 luminance: {adjusted_luminance:.2f}")
            print(f"Luminance difference after adjustment: {abs(adjusted_luminance - reference_luminance):.2f}")
            
            return adjusted_patches
        else:
            print("Warning: Measured patch 19 luminance is 0, skipping luminance adjustment")
            return measured_patches
            
    except Exception as e:
        print(f"Error in luminance adjustment: {e}")
        import traceback
        traceback.print_exc()
        return measured_patches

def create_illumination_correction_comparison(illumination_before: np.ndarray, illumination_after: np.ndarray, output_dir: Path, timestamp: str) -> None:
    """
    创建照明矫正前后对比图
    
    Args:
        illumination_before: 照明矫正前的8位图像
        illumination_after: 照明矫正后的8位图像
        output_dir: 输出目录
        timestamp: 时间戳
    """
    try:
        print("Creating illumination correction comparison image...")
        
        # 获取图像尺寸
        h, w = illumination_before.shape[:2]
        
        # 创建对比图像
        comparison_width = w * 2 + 40  # 两张图像 + 中间间隔
        comparison_height = h + 80     # 图像高度 + 标题空间
        comparison_image = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)
        
        # 设置背景为深灰色
        comparison_image[:] = (50, 50, 50)
        
        # 放置矫正前的图像（左侧）
        y_offset = 60  # 为标题留空间
        comparison_image[y_offset:y_offset+h, 0:w] = illumination_before
        
        # 放置矫正后的图像（右侧）
        comparison_image[y_offset:y_offset+h, w+40:w*2+40] = illumination_after
        
        # 添加标题文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        
        # 主标题
        main_title = "Illumination Correction Comparison"
        (title_width, title_height), _ = cv2.getTextSize(main_title, font, font_scale, thickness)
        title_x = (comparison_width - title_width) // 2
        title_y = 30
        
        # 绘制主标题（白色文字，黑色描边）
        cv2.putText(comparison_image, main_title, (title_x-1, title_y-1), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(comparison_image, main_title, (title_x+1, title_y+1), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(comparison_image, main_title, (title_x, title_y), font, font_scale, (255, 255, 255), thickness)
        
        # 子标题
        font_scale_sub = 0.8
        thickness_sub = 2
        
        # "Before Illumination Correction"标题
        before_title = "Before Illumination Correction"
        (before_width, before_height), _ = cv2.getTextSize(before_title, font, font_scale_sub, thickness_sub)
        before_x = (w - before_width) // 2
        before_y = y_offset - 10
        
        cv2.putText(comparison_image, before_title, (before_x-1, before_y-1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
        cv2.putText(comparison_image, before_title, (before_x+1, before_y+1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
        cv2.putText(comparison_image, before_title, (before_x, before_y), font, font_scale_sub, (255, 255, 255), thickness_sub)
        
        # "After Illumination Correction"标题
        after_title = "After Illumination Correction"
        (after_width, after_height), _ = cv2.getTextSize(after_title, font, font_scale_sub, thickness_sub)
        after_x = w + 40 + (w - after_width) // 2
        after_y = y_offset - 10
        
        cv2.putText(comparison_image, after_title, (after_x-1, after_y-1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
        cv2.putText(comparison_image, after_title, (after_x+1, after_y+1), font, font_scale_sub, (0, 0, 0), thickness_sub+1)
        cv2.putText(comparison_image, after_title, (after_x, after_y), font, font_scale_sub, (255, 255, 255), thickness_sub)
        
        # 添加分割线
        line_y = y_offset - 20
        cv2.line(comparison_image, (0, line_y), (comparison_width, line_y), (100, 100, 100), 2)
        
        # 保存对比图像
        comparison_path = output_dir / f'illumination_correction_comparison_{timestamp}.png'
        print(f"Attempting to save illumination comparison image to: {comparison_path}")
        success = cv2.imwrite(str(comparison_path), comparison_image)
        if success:
            print(f"✓ Illumination correction comparison saved: {comparison_path}")
        else:
            print(f"✗ Failed to save illumination comparison image: {comparison_path}")
        
        # 另外保存单独的矫正前后图像（方便单独查看）
        before_path = output_dir / f'illumination_before_{timestamp}.png'
        after_path = output_dir / f'illumination_after_{timestamp}.png'
        
        print(f"Attempting to save before image to: {before_path}")
        before_success = cv2.imwrite(str(before_path), illumination_before)
        if before_success:
            print(f"✓ Illumination before image saved: {before_path}")
        else:
            print(f"✗ Failed to save illumination before image: {before_path}")
            
        print(f"Attempting to save after image to: {after_path}")
        after_success = cv2.imwrite(str(after_path), illumination_after)
        if after_success:
            print(f"✓ Illumination after image saved: {after_path}")
        else:
            print(f"✗ Failed to save illumination after image: {after_path}")
            
    except Exception as e:
        print(f"Error creating illumination correction comparison: {e}")

def show_illumination_correction_plot(illumination_before: np.ndarray, illumination_after: np.ndarray) -> None:
    """
    显示照明矫正前后对比图
    
    Args:
        illumination_before: 照明矫正前的8位图像
        illumination_after: 照明矫正后的8位图像
    """
    try:
        print("Displaying illumination correction comparison plot...")
        
        # 创建matplotlib图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Illumination Correction Comparison', fontsize=16, fontweight='bold')
        
        # 显示矫正前的图像
        axes[0].imshow(cv2.cvtColor(illumination_before, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Before Illumination Correction', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 显示矫正后的图像
        axes[1].imshow(cv2.cvtColor(illumination_after, cv2.COLOR_BGR2RGB))
        axes[1].set_title('After Illumination Correction', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # 显示差异图
        difference = cv2.absdiff(illumination_before, illumination_after)
        difference_normalized = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
        
        im_diff = axes[2].imshow(difference_normalized, cmap='hot')
        axes[2].set_title('Difference (Hot Colormap)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # 添加颜色条
        plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图像
        plt.show()
        
        # 计算并显示统计信息
        print("\n=== Illumination Correction Statistics ===")
        print(f"Before correction - Mean: {np.mean(illumination_before):.2f}, Std: {np.std(illumination_before):.2f}")
        print(f"After correction - Mean: {np.mean(illumination_after):.2f}, Std: {np.std(illumination_after):.2f}")
        print(f"Difference - Mean: {np.mean(difference):.2f}, Max: {np.max(difference):.2f}")
        
    except Exception as e:
        print(f"Error displaying illumination correction plot: {e}")

if __name__ == "__main__":
    exit(main())
