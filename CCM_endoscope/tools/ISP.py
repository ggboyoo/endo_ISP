#!/usr/bin/env python3
"""
ISP (Image Signal Processing) Script
Reads RAW files and subtracts dark current images for image correction
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

# Import functions from raw_reader.py
try:
    from raw_reader import read_raw_image, demosaic_image_corrected_fixed
    from demosaic_easy import demosaic_easy
except ImportError:
    print("Error: raw_reader.py not found in the same directory!")
    print("Please ensure raw_reader.py is in the same directory as this script.")
    exit(1)

# Import lens shading correction functions
try:
    from lens_shading import load_correction_parameters, shading_correct
except ImportError:
    print("Error: lens_shading.py not found in the same directory!")
    print("Please ensure lens_shading.py is in the same directory as this script.")
    exit(1)

# ============================================================================
# 配置文件路径 - 直接在这里修改，无需交互输入
# ============================================================================

# 输入路径配置
INPUT_PATH = r"F:\ZJU\Picture\invert_isp\inverted_output.raw"# 待处理的RAW图像路径
DARK_RAW_PATH = r"F:\ZJU\Picture\dark\g3\average_dark.raw"  # 暗电流图像路径
LENS_SHADING_PARAMS_DIR = r"F:\ZJU\Picture\lens shading\new"  # 镜头阴影矫正参数目录

# 图像参数配置
IMAGE_WIDTH = 3840      # 图像宽度
IMAGE_HEIGHT = 2160     # 图像高度
DATA_TYPE = 'uint16'    # 数据类型

# 输出配置
OUTPUT_DIRECTORY = Path(INPUT_PATH).parent # 输出目录（None为自动生成）
GENERATE_PLOTS = True   # 是否生成对比图表
SAVE_IMAGES = True     # 是否保存处理后的图像

# 处理选项
DARK_SUBTRACTION_ENABLED = True    # 是否启用暗电流减法
LENS_SHADING_ENABLED = True        # 是否启用镜头阴影矫正
WHITE_BALANCE_ENABLED = True      # 是否启用白平衡矫正
CCM_ENABLED = True                # 是否启用CCM矫正
GAMMA_CORRECTION_ENABLED = True   # 是否启用伽马变换
DEMOSAIC_OUTPUT = True             # 是否输出去马赛克后的彩色图像

# CCM矫正参数
CCM_MATRIX_PATH = r" F:\ZJU\Picture\ccm\ccm_2\ccm_output_20250905_162714"  # CCM矩阵文件路径
CCM_MATRIX = [
    [1.7801320111582375, -0.7844420268663381, 0.004310015708100662],
    [-0.24377094860030846, 2.4432181685707977, -1.1994472199704893],
    [-0.4715762768203783, -0.7105721829898775, 2.182148459810256]
]  # CCM矩阵（如果提供则优先使用，不需要从文件加载）

# 白平衡参数
WB_PARAMS_PATH = r"F:\ZJU\Picture\wb\wb_output"   # 白平衡参数文件路径   

# 伽马变换参数
GAMMA_VALUE = 2.2                  # 伽马值（2.2为sRGB标准）

# ============================================================================
# 函数定义
# ============================================================================

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

def apply_lens_shading_correction(raw_data: np.ndarray, lens_shading_params: Dict) -> np.ndarray:
    """Apply lens shading correction to RAW data"""
    print("Applying lens shading correction...")
    
    try:
        # Use the shading_correct function from lens_shading.py
        corrected_data = shading_correct(raw_data, lens_shading_params)
        print(f"Lens shading correction applied successfully")
        return corrected_data
    except Exception as e:
        print(f"Error applying lens shading correction: {e}")
        return raw_data

def find_json_file(path: str, filename_pattern: str = None) -> Optional[str]:
    """Find JSON file in directory or return path if it's a file"""
    path_obj = Path(path)
    
    if path_obj.is_file():
        if path_obj.suffix.lower() == '.json':
            return str(path_obj)
        else:
            print(f"File is not a JSON file: {path}")
            return None
    elif path_obj.is_dir():
        # Look for JSON files in directory
        json_files = list(path_obj.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in directory: {path}")
            return None
        
        if filename_pattern:
            # Look for specific pattern
            pattern_files = [f for f in json_files if filename_pattern in f.name.lower()]
            if pattern_files:
                return str(pattern_files[0])
        
        # Return the first JSON file found
        return str(json_files[0])
    else:
        print(f"Path does not exist: {path}")
        return None

def load_white_balance_parameters(wb_params_path: str) -> Optional[Dict]:
    """Load white balance parameters from JSON file or directory"""
    try:
        # Find JSON file
        json_file = find_json_file(wb_params_path, "wb")
        if not json_file:
            print(f"White balance JSON file not found in: {wb_params_path}")
            return None
        
        print(f"Loading white balance parameters from: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            wb_data = json.load(f)
        
        # Extract white balance gains
        if 'white_balance_gains' in wb_data:
            wb_params = wb_data['white_balance_gains']
            print(f"White balance parameters loaded: {wb_params}")
            return wb_params
        else:
            print(f"No white_balance_gains found in JSON file: {json_file}")
            return None
            
    except Exception as e:
        print(f"Error loading white balance parameters: {e}")
        return None

def demosaic_16bit(raw_data: np.ndarray, bayer_pattern: str = 'rggb') -> np.ndarray:
    """
    Demosaic 16-bit RAW data to 16-bit color image using fixed demosaicing
    
    Args:
        raw_data: 16-bit RAW data
        bayer_pattern: Bayer pattern ('rggb', 'bggr', 'grbg', 'gbrg')
        
    Returns:
        16-bit color image (BGR format)
    """
    print("Demosaicing 16-bit RAW data with fixed demosaicing...")
    
    try:
        # For RGGB pattern, apply the fixed demosaicing logic directly on 16-bit data
        if bayer_pattern == 'rggb':
            print("Processing RGGB pattern with channel correction on 16-bit data...")
            
            # Convert to uint16 for OpenCV demosaicing
            raw_data_uint16 = raw_data.astype(np.uint16)
            print(f"  Converted to uint16: {raw_data_uint16.shape}, dtype: {raw_data_uint16.dtype}")
            
            # Direct demosaicing on 16-bit data
            demosaiced = cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerRG2BGR)
            
            # Apply the R/B channel swap correction (same as demosaic_image_corrected_fixed)
            corrected = demosaiced.copy()
            corrected[:, :, 0] = demosaiced[:, :, 2]  # B = R
            corrected[:, :, 2] = demosaiced[:, :, 0]  # R = B
            
            color_16bit = corrected
            
        elif bayer_pattern == 'bggr':
            raw_data_uint16 = raw_data.astype(np.uint16)
            color_16bit = cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerBG2BGR)
        elif bayer_pattern == 'grbg':
            raw_data_uint16 = raw_data.astype(np.uint16)
            color_16bit = cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerGR2BGR)
        elif bayer_pattern == 'gbrg':
            raw_data_uint16 = raw_data.astype(np.uint16)
            color_16bit = cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerGB2BGR)
        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
        
        print(f"  Demosaicing completed: {color_16bit.shape}, range: {np.min(color_16bit)}-{np.max(color_16bit)}")
        return color_16bit
        
    except Exception as e:
        print(f"  Error in demosaicing: {e}")
        return None

def apply_white_balance_correction_16bit(color_image: np.ndarray, wb_params: Dict) -> np.ndarray:
    """Apply white balance correction to 16-bit color image (12-bit data in 16-bit container)"""
    print("Applying white balance correction to 16-bit image...")
    
    try:
        # Extract white balance gains
        r_gain = wb_params.get('r_gain', 1.0)
        g_gain = wb_params.get('g_gain', 1.0)
        b_gain = wb_params.get('b_gain', 1.0)
        
        print(f"  White balance gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")
        
        # Apply gains to each channel
        corrected = color_image.copy().astype(np.float64)
        corrected[:, :, 2] *= r_gain  # Red channel
        corrected[:, :, 1] *= g_gain  # Green channel
        corrected[:, :, 0] *= b_gain  # Blue channel
        
        # Clip to valid range (12-bit data in 16-bit container)
        corrected = np.clip(corrected, 0, 4095).astype(np.uint16)
        
        print(f"  White balance correction applied: {corrected.shape}, range: {np.min(corrected)}-{np.max(corrected)}")
        return corrected
        
    except Exception as e:
        print(f"  Error applying white balance correction: {e}")
        return color_image

def load_ccm_matrix(ccm_path: str) -> Optional[Tuple[np.ndarray, str]]:
    """Load CCM matrix from JSON file or directory"""
    try:
        # Find JSON file
        json_file = find_json_file(ccm_path, "ccm")
        if not json_file:
            print(f"CCM JSON file not found in: {ccm_path}")
            return None
        
        print(f"Loading CCM matrix from: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            ccm_data = json.load(f)
        
        # Extract CCM matrix and type
        if 'ccm_matrix' in ccm_data:
            matrix = np.array(ccm_data['ccm_matrix'])
            matrix_type = ccm_data.get('ccm_type', 'linear3x3')
        elif 'matrix' in ccm_data:
            # Fallback for old format
            matrix = np.array(ccm_data['matrix'])
            matrix_type = ccm_data.get('type', 'linear3x3')
        else:
            print(f"No CCM matrix found in JSON file: {json_file}")
            return None
        
        print(f"CCM matrix loaded: {matrix.shape}, type: {matrix_type}")
        return matrix, matrix_type
        
    except Exception as e:
        print(f"Error loading CCM matrix: {e}")
        return None

def apply_gamma_correction_16bit(color_image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Apply gamma correction to 16-bit color image (12-bit data in 16-bit container)
    
    Args:
        color_image: 16-bit color image containing 12-bit data (0-4095 range)
        gamma: Gamma value (default 2.2 for sRGB)
        
    Returns:
        Gamma corrected 16-bit image (12-bit data in 16-bit container)
    """
    try:
        # Convert to float and normalize to [0, 1]
        img_float = color_image.astype(np.float64) / 4095.0
        
        # Apply gamma correction
        img_gamma = np.power(img_float, 1.0 / gamma)
        
        # Convert back to 16-bit
        img_corrected = (img_gamma * 4095.0).astype(np.uint16)
        
        print(f"  Gamma correction applied: gamma={gamma}, range: {np.min(img_corrected)}-{np.max(img_corrected)}")
        return img_corrected
        
    except Exception as e:
        print(f"  Error applying gamma correction: {e}")
        return color_image

def apply_ccm_16bit(color_image: np.ndarray, ccm_matrix: np.ndarray, matrix_type: str) -> np.ndarray:
    """Apply CCM correction to 16-bit color image (12-bit data in 16-bit container)"""
    print(f"Applying CCM correction ({matrix_type}) to 16-bit image...")
    
    try:
        # Convert to float for matrix operations
        img_float = color_image.astype(np.float64)
        img_rgb = img_float[:, :, [2, 1, 0]]
        if matrix_type == 'linear3x3':
            # 3x3 linear transformation
            corrected = np.dot(img_rgb.reshape(-1, 3), ccm_matrix.T).reshape(img_float.shape)
        elif matrix_type == 'affine3x4':
            # 3x4 affine transformation (with bias)
            img_rgb = img_rgb.reshape(-1, 3)
            img_with_bias = np.column_stack([img_rgb, np.ones(img_rgb.shape[0])])
            corrected_flat = np.dot(img_with_bias, ccm_matrix.T)
            corrected = corrected_flat.reshape(img_float.shape)
        else:
            raise ValueError(f"Unsupported matrix type: {matrix_type}")
        
        # Clip to valid range (12-bit data in 16-bit container)
        corrected = np.clip(corrected, 0, 4095).astype(np.uint16)

        corrected = corrected[:, :, [2, 1, 0]]
        print(f"  CCM correction applied: {corrected.shape}, range: {np.min(corrected)}-{np.max(corrected)}")
        return corrected
        
    except Exception as e:
        print(f"  Error applying CCM correction: {e}")
        return color_image

def process_single_image(raw_file: str, dark_data: np.ndarray, lens_shading_params: Dict, 
                        width: int, height: int, data_type: str, wb_params: Optional[Dict] = None,
                        dark_subtraction_enabled: bool = True, lens_shading_enabled: bool = True,
                        white_balance_enabled: bool = True, ccm_enabled: bool = True,
                        ccm_matrix_path: Optional[str] = None, ccm_matrix: Optional[np.ndarray] = None,
                        gamma_correction_enabled: bool = True, gamma_value: float = 2.2, 
                        demosaic_output: bool = True) -> Dict:
    """
    Process a single RAW image through the complete ISP pipeline
    
    Args:
        raw_file: Path to RAW file
        dark_data: Dark reference data
        lens_shading_params: Lens shading correction parameters
        width: Image width
        height: Image height
        data_type: Data type ('uint8' or 'uint16')
        wb_params: White balance parameters (optional)
        dark_subtraction_enabled: Enable dark current subtraction
        lens_shading_enabled: Enable lens shading correction
        white_balance_enabled: Enable white balance correction
        ccm_enabled: Enable CCM correction
        ccm_matrix_path: Path to CCM matrix file (optional if ccm_matrix provided)
        ccm_matrix: CCM matrix as numpy array (optional, takes priority over ccm_matrix_path)
        gamma_correction_enabled: Enable gamma correction
        gamma_value: Gamma value for correction (default 2.2)
        demosaic_output: Enable demosaicing output
        
    Returns:
        Dictionary containing processing results
    """
    try:
        print(f"Processing: {os.path.basename(raw_file)}")
        
        # 1. Load RAW image
        print(f"  1. Loading RAW image...")
        raw_data = read_raw_image(raw_file, width, height, data_type)
        if raw_data is None:
            return {'processing_success': False, 'error': 'Failed to load RAW image'}
        
        print(f"  1. RAW loaded: {raw_data.shape}, range: {np.min(raw_data)}-{np.max(raw_data)}")
        
        # 2. Dark current subtraction
        if dark_subtraction_enabled and dark_data is not None:
            print(f"  2. Applying dark current subtraction...")
            dark_corrected = subtract_dark_current(raw_data, dark_data, clip_negative=True)
            print(f"  2. Dark current subtraction applied")
        else:
            dark_corrected = raw_data.copy()
            print(f"  2. Dark current subtraction skipped")
        
        # 3. Lens shading correction
        if lens_shading_enabled and lens_shading_params:
            print(f"  3. Applying lens shading correction...")
            lens_corrected = apply_lens_shading_correction(dark_corrected, lens_shading_params)
            print(f"  3. Lens shading correction applied")
        else:
            lens_corrected = dark_corrected.copy()
            print(f"  3. Lens shading correction skipped")
        
        # 4. Demosaic in 16-bit domain
        if demosaic_output:
            print(f"  4. Demosaicing in 16-bit domain...")
            # color_img_16bit = demosaic_easy(lens_corrected,'rggb')
            color_img_16bit = demosaic_16bit(lens_corrected, 'rggb')
            if color_img_16bit is not None:
                print(f"  4. 16-bit color image: {color_img_16bit.shape}, range: {np.min(color_img_16bit)}-{np.max(color_img_16bit)}")
                
                # 保存去马赛克后的图像用于对比
                demosaiced_8bit = (color_img_16bit.astype(np.float32) / 4095 * 255.0).astype(np.uint8)
                
                # 5. White balance correction in 16-bit domain
                if white_balance_enabled and wb_params is not None:
                    print(f"  5. Applying white balance correction in 16-bit domain...")
                    color_img_16bit = apply_white_balance_correction_16bit(color_img_16bit, wb_params)
                    print(f"  5. White balance correction applied to 16-bit image")
                else:
                    print(f"  5. White balance correction skipped")
                
                # 保存白平衡后的图像用于对比
                wb_corrected_8bit = (color_img_16bit.astype(np.float32) / 4095 * 255.0).astype(np.uint8)
                
                # 6. CCM correction in 16-bit domain
                if ccm_enabled:
                    print(f"  6. Applying CCM correction in 16-bit domain...")
                    
                    # Priority: use ccm_matrix if provided, otherwise load from file
                    if ccm_matrix is not None:
                        print(f"  6. Using provided CCM matrix: {ccm_matrix.shape}")
                        matrix_type = 'linear3x3'  # Default to linear3x3 for direct matrix input
                        color_img_16bit = apply_ccm_16bit(color_img_16bit, ccm_matrix, matrix_type)
                        print(f"  6. CCM correction applied to 16-bit image using provided matrix")
                    elif ccm_matrix_path is not None:
                        ccm_result = load_ccm_matrix(ccm_matrix_path)
                        if ccm_result is not None:
                            ccm_matrix_loaded, matrix_type = ccm_result
                            color_img_16bit = apply_ccm_16bit(color_img_16bit, ccm_matrix_loaded, matrix_type)
                            print(f"  6. CCM correction applied to 16-bit image using loaded matrix")
                        else:
                            print(f"  6. CCM correction failed, skipping...")
                    else:
                        print(f"  6. No CCM matrix provided, skipping CCM correction...")
                else:
                    print(f"  6. CCM correction skipped")
                
                # 保存CCM矫正后的图像用于对比
                ccm_corrected_8bit = (color_img_16bit.astype(np.float32) / 4095 * 255.0).astype(np.uint8)
                
                # 7. Gamma correction (convert to nonlinear domain)
                if gamma_correction_enabled:
                    print(f"  7. Applying gamma correction...")
                    color_img_16bit = apply_gamma_correction_16bit(color_img_16bit, gamma=gamma_value)
                    print(f"  7. Gamma correction applied to 16-bit image")
                else:
                    print(f"  7. Gamma correction skipped")
                
                # 保存伽马矫正后的图像用于对比
                gamma_corrected_8bit = (color_img_16bit.astype(np.float32) / 4095 * 255.0).astype(np.uint8)
                
                # 8. Convert to 8-bit for display/saving
                print(f"  8. Converting to 8-bit for display...")
                # 
                max_val = np.max(color_img_16bit)
                if max_val > 0:
                    color_img_8bit = (color_img_16bit.astype(np.float32) / 4095 * 255.0).astype(np.uint8)
                else:
                    color_img_8bit = np.zeros_like(color_img_16bit, dtype=np.uint8)
                print(f"  8. 8-bit color image: {color_img_8bit.shape}, range: {np.min(color_img_8bit)}-{np.max(color_img_8bit)}")
            else:
                print(f"  4. Demosaicing failed!")
                color_img_16bit = None
                color_img_8bit = None
                demosaiced_8bit = None
                wb_corrected_8bit = None
                ccm_corrected_8bit = None
                gamma_corrected_8bit = None
        else:
            color_img_16bit = None
            color_img_8bit = None
            demosaiced_8bit = None
            wb_corrected_8bit = None
            ccm_corrected_8bit = None
            gamma_corrected_8bit = None
        
        return {
            'processing_success': True,
            'original_data': raw_data,
            'dark_corrected': dark_corrected,
            'lens_corrected': lens_corrected,
            'color_img': color_img_8bit,
            'color_img_16bit': color_img_16bit,
            'demosaiced_8bit': demosaiced_8bit,
            'wb_corrected_8bit': wb_corrected_8bit,
            'ccm_corrected_8bit': ccm_corrected_8bit,
            'gamma_corrected_8bit': gamma_corrected_8bit
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {
            'processing_success': False,
            'error': str(e)
        }

def create_comparison_plot(original: np.ndarray, dark_corrected: np.ndarray, 
                          lens_corrected: np.ndarray, color_img: Optional[np.ndarray] = None,
                          demosaiced_8bit: Optional[np.ndarray] = None,
                          wb_corrected_8bit: Optional[np.ndarray] = None,
                          ccm_corrected_8bit: Optional[np.ndarray] = None,
                          gamma_corrected_8bit: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None) -> None:
    """Create comparison plot of processing steps"""
    if not GENERATE_PLOTS:
        return
    
    # Create a larger subplot grid to accommodate all steps
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('ISP Processing Steps Comparison', fontsize=16, fontweight='bold')
    
    # Row 1: RAW processing steps
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('1. Original RAW', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(dark_corrected, cmap='gray')
    axes[0, 1].set_title('2. Dark Corrected', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(lens_corrected, cmap='gray')
    axes[0, 2].set_title('3. Lens Shading Corrected', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Color processing steps
    if demosaiced_8bit is not None:
        demosaiced_rgb = cv2.cvtColor(demosaiced_8bit, cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(demosaiced_rgb)
        axes[1, 0].set_title('4. Demosaiced', fontsize=12, fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('4. Demosaiced', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    if wb_corrected_8bit is not None:
        wb_rgb = cv2.cvtColor(wb_corrected_8bit, cv2.COLOR_BGR2RGB)
        axes[1, 1].imshow(wb_rgb)
        axes[1, 1].set_title('5. White Balance', fontsize=12, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('5. White Balance', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    if ccm_corrected_8bit is not None:
        ccm_rgb = cv2.cvtColor(ccm_corrected_8bit, cv2.COLOR_BGR2RGB)
        axes[1, 2].imshow(ccm_rgb)
        axes[1, 2].set_title('6. CCM Corrected', fontsize=12, fontweight='bold')
    else:
        axes[1, 2].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('6. CCM Corrected', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Row 3: Final processing steps
    if gamma_corrected_8bit is not None:
        gamma_rgb = cv2.cvtColor(gamma_corrected_8bit, cv2.COLOR_BGR2RGB)
        axes[2, 0].imshow(gamma_rgb)
        axes[2, 0].set_title('7. Gamma Corrected', fontsize=12, fontweight='bold')
    else:
        axes[2, 0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('7. Gamma Corrected', fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    
    # Final result
    if color_img is not None:
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        axes[2, 1].imshow(color_rgb)
        axes[2, 1].set_title('8. Final Result', fontsize=12, fontweight='bold')
    else:
        axes[2, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('8. Final Result', fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')
    
    # Hide the last subplot
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    
    plt.show()

def main():
    """Main function"""
    print("=" * 60)
    print("ISP (Image Signal Processing) Pipeline")
    print("=" * 60)
    print(f"Input path: {INPUT_PATH}")
    print(f"Dark reference: {DARK_RAW_PATH}")
    print(f"Lens shading params: {LENS_SHADING_PARAMS_DIR}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Data type: {DATA_TYPE}")
    print("=" * 60)
    
    # Create output directory
    if OUTPUT_DIRECTORY is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"isp_output_{timestamp}")
    else:
        output_dir = Path(OUTPUT_DIRECTORY)
    
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load dark reference
    dark_data = None
    dark_subtraction_enabled = DARK_SUBTRACTION_ENABLED
    if dark_subtraction_enabled:
        print(f"\nLoading dark reference...")
        dark_data = load_dark_reference(DARK_RAW_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
        if dark_data is None:
            print("Warning: Failed to load dark reference, continuing without dark subtraction")
            dark_subtraction_enabled = False
    
    # Load lens shading parameters
    lens_shading_params = None
    lens_shading_enabled = LENS_SHADING_ENABLED
    if lens_shading_enabled:
        print(f"\nLoading lens shading parameters...")
        lens_shading_params = load_correction_parameters(LENS_SHADING_PARAMS_DIR)
        if lens_shading_params is None:
            print("Warning: Failed to load lens shading parameters, continuing without lens shading correction")
            lens_shading_enabled = False
    
    # Load white balance parameters
    wb_params = None
    white_balance_enabled = WHITE_BALANCE_ENABLED
    if white_balance_enabled and WB_PARAMS_PATH:
        print(f"\nLoading white balance parameters...")
        wb_params = load_white_balance_parameters(WB_PARAMS_PATH)
        if wb_params is None:
            print("Warning: Failed to load white balance parameters, continuing without white balance correction")
            white_balance_enabled = False
    
    # CCM parameters
    ccm_enabled = CCM_ENABLED
    ccm_matrix_path = CCM_MATRIX_PATH
    ccm_matrix = np.array(CCM_MATRIX) if CCM_MATRIX else None
    
    # Process images
    input_path = Path(INPUT_PATH)
    if input_path.is_file():
        # Single file
        raw_files = [input_path]
    else:
        # Directory
        raw_files = list(input_path.glob("*.raw")) 
    
    if not raw_files:
        print("No RAW files found!")
        return
    
    print(f"\nFound {len(raw_files)} RAW files to process")
    
    # Process each file
    for i, raw_file in enumerate(raw_files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(raw_files)}: {raw_file.name}")
        print(f"{'='*60}")
        
        result = process_single_image(
            raw_file=str(raw_file),
            dark_data=dark_data,
            lens_shading_params=lens_shading_params,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            data_type=DATA_TYPE,
            wb_params=wb_params,
            dark_subtraction_enabled=dark_subtraction_enabled,
            lens_shading_enabled=lens_shading_enabled,
            white_balance_enabled=white_balance_enabled,
            ccm_enabled=ccm_enabled,
            ccm_matrix_path=ccm_matrix_path,
            ccm_matrix=ccm_matrix,
            gamma_correction_enabled=GAMMA_CORRECTION_ENABLED,
            gamma_value=GAMMA_VALUE,
            demosaic_output=DEMOSAIC_OUTPUT
        )
        
        if result['processing_success']:
            print(f"Processing completed successfully!")
            
            # Save images
            if SAVE_IMAGES:
                # Save 8-bit color image
                if result['color_img'] is not None:
                    output_file = output_dir / f"{raw_file.stem}_processed.jpg"
                    cv2.imwrite(str(output_file), result['color_img'])
                    print(f"8-bit color image saved: {output_file}")
                
                # Save 16-bit color image
                if result['color_img_16bit'] is not None:
                    output_file_16bit = output_dir / f"{raw_file.stem}_processed_16bit.png"
                    cv2.imwrite(str(output_file_16bit), result['color_img'])
                    print(f"16-bit color image saved: {output_file_16bit}")
            
            # Create comparison plot
            if GENERATE_PLOTS:
                plot_path = output_dir / f"{raw_file.stem}_comparison.png"
                create_comparison_plot(
                    original=result['original_data'],
                    dark_corrected=result['dark_corrected'],
                    lens_corrected=result['lens_corrected'],
                    color_img=result['color_img'],
                    demosaiced_8bit=result.get('demosaiced_8bit'),
                    wb_corrected_8bit=result.get('wb_corrected_8bit'),
                    ccm_corrected_8bit=result.get('ccm_corrected_8bit'),
                    gamma_corrected_8bit=result.get('gamma_corrected_8bit'),
                    save_path=str(plot_path)
                )
        else:
            print(f"Processing failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*60}")
    print("ISP Processing Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
