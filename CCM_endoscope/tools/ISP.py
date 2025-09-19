            #!/usr/bin/env python3
"""
            ccm_matrix_path=None,  # No path needed when using global CCM_MATRIXand subtracts dark current images for image correction
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

# Optional demosaicing library (colour-demosaicing)
try:
    from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004 as _colour_demosaic
except Exception:
    _colour_demosaic = None

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
INPUT_PATH = r"F:\ZJU\Picture\ccm\ccm_1\25-09-05 101445.raw"# 待处理的RAW图像路径
DARK_RAW_PATH = r"F:\ZJU\Picture\dark\g3\average_dark.raw"  # 暗电流图像路径
LENS_SHADING_PARAMS_DIR = r"F:\ZJU\Picture\lens shading\regress"  # 镜头阴影矫正参数目录

# 图像参数配置
RESOLUTION = '4k'       # 分辨率选择: '1k', '4k', 'auto'
IMAGE_WIDTH = None      # 图像宽度（根据分辨率设置或自动检测）
IMAGE_HEIGHT = None     # 图像高度（根据分辨率设置或自动检测）
DATA_TYPE = 'uint16'    # 数据类型（固定为uint16）

# 分辨率定义
RESOLUTIONS = {
    '1k': {'width': 1920, 'height': 1080, 'name': '1K (1920x1080)'},
    '4k': {'width': 3840, 'height': 2160, 'name': '4K (3840x2160)'}
}

# 输出配置
OUTPUT_DIRECTORY = Path(INPUT_PATH).parent # 输出目录（None为自动生成）
GENERATE_PLOTS = True   # 是否生成对比图表
SAVE_IMAGES = True     # 是否保存处理后的图像

# 处理选项
DARK_SUBTRACTION_ENABLED = True    # 是否启用暗电流减法
LENS_SHADING_ENABLED = True       # 是否启用镜头阴影矫正
WHITE_BALANCE_ENABLED = True      # 是否启用白平衡矫正
CCM_ENABLED = True                # 是否启用CCM矫正
GAMMA_CORRECTION_ENABLED = True   # 是否启用伽马变换
DEMOSAIC_OUTPUT = True             # 是否输出去马赛克后的彩色图像

# 尺寸检查选项
CHECK_DIMENSIONS = True            # 是否检查尺寸匹配
SKIP_ON_DIMENSION_MISMATCH = True  # 尺寸不匹配时是否跳过该步骤

# CCM矫正参数
CCM_MATRIX_PATH = r"F:\ZJU\Picture\ccm\ccm_2\ccm_output_20250918_151644"  # CCM矩阵文件路径
CCM_MATRIX = None
# CCM_MATRIX = [
#     [
#       1.6023577244053917,
#       -0.7021584601302941,
#       0.09980073572490247
#     ],
#     [
#       -0.18188330118239007,
#       2.4686994973519805,
#       -1.2868161961695905
#     ],
#     [
#       -0.41185887887176487,
#       -0.7704932869035739,
#       2.1823521657753386
#     ]
# ]  # CCM矩阵（如果提供则优先使用，不需要从文件加载）
 

# 白平衡参数
WB_PARAMS_PATH = r"F:\ZJU\Picture\wb\wb_output"   # 白平衡参数文件路径   

# 伽马变换参数
GAMMA_VALUE = 2.2                  # 伽马值（2.2为sRGB标准）

# ============================================================================
# 函数定义
# ============================================================================

def set_resolution_config(resolution: str) -> None:
    """
    根据分辨率设置全局配置参数
    
    Args:
        resolution: 分辨率选择 ('1k', '4k', 'auto')
    """
    global IMAGE_WIDTH, IMAGE_HEIGHT
    
    if resolution in RESOLUTIONS:
        res_info = RESOLUTIONS[resolution]
        IMAGE_WIDTH = res_info['width']
        IMAGE_HEIGHT = res_info['height']
        print(f"设置分辨率: {res_info['name']}")
    elif resolution == 'auto':
        print("使用自动检测分辨率")
        IMAGE_WIDTH = None
        IMAGE_HEIGHT = None
    else:
        print(f"未知分辨率: {resolution}，使用自动检测")
        IMAGE_WIDTH = None
        IMAGE_HEIGHT = None

def check_dimension_compatibility(data: np.ndarray, target_width: int, target_height: int, 
                                 data_name: str = "data") -> bool:
    """
    检查数据尺寸是否与目标尺寸兼容
    
    Args:
        data: 要检查的数据数组
        target_width: 目标宽度
        target_height: 目标高度
        data_name: 数据名称（用于日志）
        
    Returns:
        True if compatible, False otherwise
    """
    if data is None:
        return False
    
    # 如果配置中禁用了尺寸检查，直接返回True
    if not CHECK_DIMENSIONS:
        print(f"  ⚠️  Dimension check disabled for {data_name}")
        return True
    
    data_height, data_width = data.shape[:2]
    
    if data_width == target_width and data_height == target_height:
        print(f"  ✅ {data_name} dimensions match: {data_width}x{data_height}")
        return True
    else:
        print(f"  ⚠️  {data_name} dimensions mismatch:")
        print(f"      Expected: {target_width}x{target_height}")
        print(f"      Actual: {data_width}x{data_height}")
        
        # 根据配置决定是否跳过
        if SKIP_ON_DIMENSION_MISMATCH:
            print(f"      Skipping {data_name} correction...")
            return False
        else:
            print(f"      Proceeding with {data_name} correction despite mismatch...")
            return True

def check_lens_shading_compatibility(raw_data: np.ndarray, lens_shading_params: Dict) -> bool:
    """
    检查镜头阴影参数是否与RAW图像尺寸兼容
    
    Args:
        raw_data: RAW图像数据
        lens_shading_params: 镜头阴影参数字典，包含R/G1/G2/B通道的小图
    
    Returns:
        bool: 如果兼容返回True，否则返回False
    """
    if lens_shading_params is None:
        print(f"  Warning: lens_shading_params is None")
        return False
    
    h, w = raw_data.shape[:2]
    expected_h, expected_w = h // 2, w // 2  # 期望的小图尺寸
    
    # 检查每个通道的尺寸
    channels = ['R', 'G1', 'G2', 'B']
    for channel in channels:
        if channel in lens_shading_params:
            channel_map = lens_shading_params[channel]
            if channel_map is not None:
                actual_h, actual_w = channel_map.shape[:2]
                # 直接判断尺寸是否完全匹配，不允许误差
                if actual_h != expected_h or actual_w != expected_w:
                    print(f"  Warning: lens shading {channel} channel dimension mismatch!")
                    print(f"    Expected: {expected_w}x{expected_h}")
                    print(f"    Actual: {actual_w}x{actual_h}")
                    return False
    
    return True

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

def lensshading_correction(raw_data: np.ndarray, correction_map: np.ndarray) -> np.ndarray:
    """
    Apply lens shading correction given a correction map array directly.

    - raw_data: RAW image, expected 0..4095 scale
    - correction_map: same HxW as raw_data; values multiply raw
    Returns float64 image clipped to [0,4095].
    """
    try:
        if correction_map is None:
            return raw_data
        if correction_map.shape[:2] != raw_data.shape[:2]:
            print("  Warning: correction_map shape mismatch; skipping lens shading correction")
            return raw_data
        data = raw_data.astype(np.float64)
        corrected = data * correction_map.astype(np.float64)
        # 只裁剪负值，保持精度
        corrected = np.clip(corrected, 0, None)
        print(f"  Lens shading correction applied directly: range {np.min(corrected)}-{np.max(corrected)}")
        return corrected
    except Exception as e:
        print(f"  Error in direct lens shading correction: {e}")
        return raw_data

def lensshading_correction_bayer(raw_data: np.ndarray, channel_maps: Dict[str, np.ndarray], bayer_pattern: str = 'rggb') -> np.ndarray:
    """
    Apply lens shading correction with per-CFA-channel correction maps on Bayer RAW.

    channel_maps keys expected: 'R', 'G1', 'G2', 'B'. Each map is (H/2, W/2).
    If sizes mismatch, maps will be resized to (H/2, W/2) with bilinear interpolation.
    Returns float64 image clipped to [0,4095].
    """
    try:
        h, w = raw_data.shape[:2]
        hh, ww = h // 2, w // 2

        def ensure_size(m: np.ndarray) -> np.ndarray:
            if m is None:
                return np.ones((hh, ww), dtype=np.float64)
            if m.shape != (hh, ww):
                # Resize to (ww, hh) as width-first for cv2
                m_resized = cv2.resize(m.astype(np.float64), (ww, hh), interpolation=cv2.INTER_LINEAR)
                return m_resized
            return m.astype(np.float64)

        Rm = ensure_size(channel_maps.get('R'))
        G1m = ensure_size(channel_maps.get('G1'))
        G2m = ensure_size(channel_maps.get('G2'))
        Bm = ensure_size(channel_maps.get('B'))

        data = raw_data.astype(np.float64)
        pat = (bayer_pattern or 'rggb').lower()

        if pat == 'rggb':
            data[0::2, 0::2] *= Rm
            data[0::2, 1::2] *= G1m
            data[1::2, 0::2] *= G2m
            data[1::2, 1::2] *= Bm
        else:
            # Fallback: treat as rggb indexing
            data[0::2, 0::2] *= Rm
            data[0::2, 1::2] *= G1m
            data[1::2, 0::2] *= G2m
            data[1::2, 1::2] *= Bm

        # 只裁剪负值，保持精度
        corrected = np.clip(data, 0, None)
        print(f"  Lens shading correction (per-channel) applied: range {np.min(corrected)}-{np.max(corrected)}")
        return corrected
    except Exception as e:
        print(f"  Error in per-channel lens shading correction: {e}")
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
    print("Demosaicing 16-bit RAW data (prefer colour_demosaicing Malvar2004)...")

    # Try colour_demosaicing first
    try:
        if _colour_demosaic is not None:
            pattern_map = {
                'rggb': 'RGGB',
                'bggr': 'BGGR',
                'grbg': 'GRBG',
                'gbrg': 'GBRG',
            }
            pat = pattern_map.get(bayer_pattern.lower())
            if pat is None:
                raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")

            raw_clip = np.clip(raw_data.astype(np.float64), 0.0, 4095.0)
            raw_norm = raw_clip / 4095.0
            rgb_float = _colour_demosaic(raw_norm, pattern=pat)
            # Convert RGB float [0,1] back to 16-bit (12-bit content)
            rgb_float = np.clip(rgb_float * 4095.0, 0, 4095)
            # Convert to BGR channel order for consistency with OpenCV usage downstream
            bgr_float = rgb_float[:, :, [2, 1, 0]]
            print(f"  Demosaicing (colour) completed: {rgb_float.shape}, range: {np.min(rgb_float)}-{np.max(rgb_float)}")
            return bgr_float
    except Exception as e:
        print(f"  colour_demosaicing failed, will fallback to OpenCV: {e}")

    # Fallback to OpenCV bilinear if colour_demosaicing is unavailable or failed
    try:
        bp = bayer_pattern.lower()
        raw_data_uint16 = raw_data.astype(np.uint16)
        if bp == 'rggb':
            demosaiced = cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerRG2BGR)
            corrected = demosaiced.copy()
            corrected[:, :, 0] = demosaiced[:, :, 2]
            corrected[:, :, 2] = demosaiced[:, :, 0]
            color_16bit = corrected
        elif bp == 'bggr':
            color_16bit = cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerBG2BGR)
        elif bp == 'grbg':
            color_16bit = cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerGR2BGR)
        elif bp == 'gbrg':
            color_16bit = cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerGB2BGR)
        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
        print(f"  Demosaicing (OpenCV) completed: {color_16bit.shape}, range: {np.min(color_16bit)}-{np.max(color_16bit)}")
        return color_16bit
    except Exception as e:
        print(f"  Error in fallback demosaicing: {e}")
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
        
        # 只裁剪负值，保持精度
        corrected = np.clip(corrected, 0, None)
        
        print(f"  White balance correction applied: {corrected.shape}, range: {np.min(corrected)}-{np.max(corrected)}")
        return corrected
        
    except Exception as e:
        print(f"  Error applying white balance correction: {e}")
        return color_image


def apply_white_balance_bayer(raw_data: np.ndarray, wb_params: Dict, bayer_pattern: str = 'rggb') -> np.ndarray:
    """
    Apply white balance directly on Bayer mosaic (16-bit container with 12-bit range).

    wb_params keys: 'r_gain', 'g_gain', 'b_gain'.
    """
    print("Applying white balance on Bayer mosaic...")
    try:
        r_gain = float(wb_params.get('r_gain', 1.0))
        g_gain = float(wb_params.get('g_gain', 1.0))
        b_gain = float(wb_params.get('b_gain', 1.0))

        pat = (bayer_pattern or 'rggb').lower()
        data = raw_data.astype(np.float64)

        if pat == 'rggb':
            # R at (0,0), G at (0,1) and (1,0), B at (1,1)
            data[0::2, 0::2] *= r_gain
            data[0::2, 1::2] *= g_gain
            data[1::2, 0::2] *= g_gain
            data[1::2, 1::2] *= b_gain
        else:
            # Fallback: treat as rggb
            data[0::2, 0::2] *= r_gain
            data[0::2, 1::2] *= g_gain
            data[1::2, 0::2] *= g_gain
            data[1::2, 1::2] *= b_gain

        # 只裁剪负值，保持精度
        data = np.clip(data, 0, None)
        print("  White balance applied on Bayer domain")
        return data
    except Exception as e:
        print(f"  Error applying Bayer white balance: {e}")
        return raw_data

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
        img_gamma = img_float ** (1.0 / gamma)  
        
        # Convert back to 16-bit
        img_corrected = img_gamma * 4095.0
        
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
        
        # 只裁剪负值，保持精度
        corrected = np.clip(corrected, 0, 4095)

        corrected = corrected[:, :, [2, 1, 0]]
        print(f"  CCM correction applied: {corrected.shape}, range: {np.min(corrected)}-{np.max(corrected)}")
        return corrected
        
    except Exception as e:
        print(f"  Error applying CCM correction: {e}")
        return color_image

def process_raw_array(raw_data: np.ndarray, dark_data: np.ndarray, lens_shading_params: Dict,
                      width: int, height: int, data_type: str, wb_params: Optional[Dict] = None,
                      dark_subtraction_enabled: bool = True, lens_shading_enabled: bool = True,
                      white_balance_enabled: bool = True, ccm_enabled: bool = True,
                      ccm_matrix_path: Optional[str] = None, ccm_matrix: Optional[np.ndarray] = None,
                      gamma_correction_enabled: bool = True, gamma_value: float = 2.2,
                      demosaic_output: bool = True) -> Dict:
    """
    Process a RAW ndarray through the complete ISP pipeline.

    Args mirror process_single_image, except raw_data is provided directly.
    """
    try:
        print(f"Processing ndarray RAW input...")
        if raw_data is None:
            return {'processing_success': False, 'error': 'raw_data is None'}
        if raw_data.shape != (height, width):
            print(f"  Warning: raw_data shape {raw_data.shape} != ({height}, {width}), attempting reshape if sizes match")
            if raw_data.size == width * height:
                raw_data = raw_data.reshape((height, width))
            else:
                return {'processing_success': False, 'error': f'size mismatch: {raw_data.size} vs {width*height}'}
        print(f"  RAW received: {raw_data.shape}, range: {np.min(raw_data)}-{np.max(raw_data)}")
        
        # 2. Dark current subtraction
        if dark_subtraction_enabled and dark_data is not None:
            print(f"  2. Applying dark current subtraction...")
            # 检查暗电流数据尺寸是否匹配
            if check_dimension_compatibility(dark_data, width, height, "dark reference"):
                dark_corrected = subtract_dark_current(raw_data, dark_data, clip_negative=True)
                print(f"  2. Dark current subtraction applied")
            else:
                dark_corrected = raw_data.copy()
                print(f"  2. Dark current subtraction skipped due to dimension mismatch")
        else:
            dark_corrected = raw_data.copy()
            print(f"  2. Dark current subtraction skipped")
        
        # 3. Lens shading correction
        if lens_shading_enabled and lens_shading_params is not None:
            print(f"  3. Applying lens shading correction (Bayer per-channel map)...")
            # 检查镜头阴影参数是否与图像尺寸匹配
            if check_lens_shading_compatibility(dark_corrected, lens_shading_params):
                # lens_shading_params 为一个包含 R/G1/G2/B 小图的字典
                lens_corrected = lensshading_correction_bayer(dark_corrected, lens_shading_params, 'rggb')
                print(f"  3. Lens shading correction applied")
            else:
                lens_corrected = dark_corrected.copy()
                print(f"  3. Lens shading correction skipped due to dimension mismatch")
        else:
            lens_corrected = dark_corrected.copy()
            print(f"  3. Lens shading correction skipped")
        
        # 4. White balance on Bayer, then demosaic in 16-bit domain
        if demosaic_output:
            # Apply WB on Bayer BEFORE demosaicing (if requested)
            wb_applied_pre = False
            raw_for_demosaic = lens_corrected
            if white_balance_enabled and wb_params is not None:
                print(f"  4. Applying white balance on Bayer before demosaicing...")
                raw_for_demosaic = apply_white_balance_bayer(lens_corrected.astype(np.uint16), wb_params, 'rggb')
                wb_applied_pre = True
            print(f"  4. Demosaicing in 16-bit domain...")
            
            color_img_16bit = demosaic_16bit(raw_for_demosaic, 'rggb')
            # print(f"mean:{np.mean(color_img_16bit)}")
            
            if color_img_16bit is not None:
                print(f"  4. 16-bit color image: {color_img_16bit.shape}, range: {np.min(color_img_16bit)}-{np.max(color_img_16bit)}")
                
                # 保存去马赛克后的图像用于对比
                demosaiced_8bit = np.round((color_img_16bit.astype(np.float32) / 4095.0) * 255.0).astype(np.uint8)
                
                # 5. Skip WB here if already applied on Bayer
                if wb_applied_pre:
                    print(f"  5. White balance already applied on Bayer, skipping post-demosaic WB")
                else:
                    if white_balance_enabled and wb_params is not None:
                        print(f"  5. Applying white balance correction in 16-bit domain...")
                        color_img_16bit = apply_white_balance_correction_16bit(color_img_16bit, wb_params)
                        print(f"  5. White balance correction applied to 16-bit image")
                    else:
                        print(f"  5. White balance correction skipped")
                
                # 保存白平衡后的图像用于对比
                wb_corrected_8bit = np.round((color_img_16bit.astype(np.float32) / 4095.0) * 255.0).astype(np.uint8)
                
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
                
                # 在CCM矫正后进行clip，然后进行gamma矫正
                color_img_16bit = np.clip(color_img_16bit, 0, 4095)
                print(f"mean:{np.mean(color_img_16bit)}")

                # 保存CCM矫正后的图像用于对比
                ccm_corrected_8bit = np.round((color_img_16bit.astype(np.float32) / 4095.0) * 255.0).astype(np.uint8)
                
                # 7. Gamma correction (convert to nonlinear domain)
                if gamma_correction_enabled:
                    print(f"  7. Applying gamma correction...")
                    color_img_16bit = apply_gamma_correction_16bit(color_img_16bit, gamma=gamma_value)
                    print(f"  7. Gamma correction applied to 16-bit image")
                else:
                    print(f"  7. Gamma correction skipped")
                
                # 保存伽马矫正后的图像用于对比
                gamma_corrected_8bit = np.round((color_img_16bit.astype(np.float32) / 4095.0) * 255.0).astype(np.uint8)
                
                # 8. Convert to 8-bit for display/saving
                print(f"  8. Converting to 8-bit for display...")
                # 
                max_val = np.max(color_img_16bit)
                if max_val > 0:
                    color_img_8bit = np.round((color_img_16bit.astype(np.float32) / 4095.0) * 255.0).astype(np.uint8)
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
        print(f"Error processing array: {e}")
        return {
            'processing_success': False,
            'error': str(e)
        }


def process_single_image(raw_file, dark_data=None, lens_shading_params=None, 
                        width: int = None, height: int = None, data_type: str = None, 
                        wb_params=None, dark_subtraction_enabled: bool = True, 
                        lens_shading_enabled: bool = True, white_balance_enabled: bool = True, 
                        ccm_enabled: bool = True, ccm_matrix_path: str = None, 
                        ccm_matrix: np.ndarray = None, gamma_correction_enabled: bool = True, 
                        gamma_value: float = 2.2, demosaic_output: bool = True) -> Dict:
    """
    Process single image with automatic parameter loading.
    Parameters can be paths (strings) or values (arrays/objects).
    If path, loads the parameter; if value, uses directly.
    
    Args:
        raw_file: Path to RAW file (str) or RAW data array (np.ndarray)
        dark_data: Path to dark reference (str) or dark data array (np.ndarray)
        lens_shading_params: Path to lens shading params (str) or params dict
        width: Image width (int)
        height: Image height (int)
        data_type: Data type (str)
        wb_params: Path to WB params (str) or params dict
        dark_subtraction_enabled: Enable dark subtraction (bool)
        lens_shading_enabled: Enable lens shading correction (bool)
        white_balance_enabled: Enable white balance (bool)
        ccm_enabled: Enable CCM correction (bool)
        ccm_matrix_path: Path to CCM matrix (str)
        ccm_matrix: CCM matrix array (np.ndarray)
        gamma_correction_enabled: Enable gamma correction (bool)
        gamma_value: Gamma value (float)
        demosaic_output: Enable demosaic output (bool)
    """
    try:
        # Determine if raw_file is a path or data array
        if isinstance(raw_file, str):
            print(f"Processing: {os.path.basename(raw_file)}")
            raw_data = None  # Will be loaded from file
        elif isinstance(raw_file, np.ndarray):
            print(f"Processing: Provided RAW data array {raw_file.shape}")
            raw_data = raw_file  # Use provided data directly
        else:
            return {'processing_success': False, 'error': 'raw_file must be a string path or numpy array'}
        
        # Load or use parameters
        print(f"  Loading parameters...")
        
        # Image dimensions and data type
        if width is None:
            width = IMAGE_WIDTH
        if height is None:
            height = IMAGE_HEIGHT
        if data_type is None:
            data_type = DATA_TYPE
            
        # Load dark data if path provided
        if isinstance(dark_data, str):
            print(f"  Loading dark reference from: {dark_data}")
            dark_data = load_dark_reference(dark_data, width, height, data_type)
            if dark_data is None:
                print("  Warning: Failed to load dark reference, continuing without dark subtraction")
                dark_subtraction_enabled = False
        elif dark_data is None and dark_subtraction_enabled:
            print(f"  Loading dark reference from: {DARK_RAW_PATH}")
            dark_data = load_dark_reference(DARK_RAW_PATH, width, height, data_type)
            if dark_data is None:
                print("  Warning: Failed to load dark reference, continuing without dark subtraction")
                dark_subtraction_enabled = False
        
        # Load lens shading parameters if path provided
        if isinstance(lens_shading_params, str):
            print(f"  Loading lens shading parameters from: {lens_shading_params}")
            lens_shading_params = load_correction_parameters(lens_shading_params)
            if lens_shading_params is None:
                print("  Warning: Failed to load lens shading parameters, continuing without lens shading correction")
                lens_shading_enabled = False
        elif lens_shading_params is None and lens_shading_enabled:
            print(f"  Loading lens shading parameters from: {LENS_SHADING_PARAMS_DIR}")
            lens_shading_params = load_correction_parameters(LENS_SHADING_PARAMS_DIR)
            if lens_shading_params is None:
                print("  Warning: Failed to load lens shading parameters, continuing without lens shading correction")
                lens_shading_enabled = False
        
        # Load white balance parameters if path provided
        if isinstance(wb_params, str):
            print(f"  Loading white balance parameters from: {wb_params}")
            wb_params = load_white_balance_parameters(wb_params)
            if wb_params is None:
                print("  Warning: Failed to load white balance parameters, continuing without white balance correction")
                white_balance_enabled = False
        elif wb_params is None and white_balance_enabled:
            print(f"  Loading white balance parameters from: {WB_PARAMS_PATH}")
            wb_params = load_white_balance_parameters(WB_PARAMS_PATH)
            if wb_params is None:
                print("  Warning: Failed to load white balance parameters, continuing without white balance correction")
                white_balance_enabled = False
        
        # Load CCM matrix if path provided, or use provided matrix
        if ccm_matrix is not None:
            print(f"  Using provided CCM matrix: {ccm_matrix.shape}")
        elif isinstance(ccm_matrix_path, str):
            print(f"  Loading CCM matrix from: {ccm_matrix_path}")
            ccm_result = load_ccm_matrix(ccm_matrix_path)
            if ccm_result is not None:
                ccm_matrix, _ = ccm_result
                print(f"  CCM matrix loaded: {ccm_matrix.shape}")
            else:
                print("  Warning: Failed to load CCM matrix, continuing without CCM correction")
                ccm_enabled = False
        elif ccm_matrix_path is None and ccm_enabled and ccm_matrix is None:
            # Check if global CCM_MATRIX has values
            if CCM_MATRIX and len(CCM_MATRIX) > 0:
                ccm_matrix = np.array(CCM_MATRIX)
                print(f"  Using global CCM matrix: {ccm_matrix.shape}")
            else:
                print("  No CCM matrix available, skipping CCM correction")
                ccm_enabled = False
        
        # Load RAW data if path provided, otherwise use provided data
        if isinstance(raw_file, str):
            print(f"  1. Loading RAW image from file...")
            raw_data = read_raw_image(raw_file, width, height, data_type)
            if raw_data is None:
                return {'processing_success': False, 'error': 'Failed to load RAW image'}
            print(f"  1. RAW loaded: {raw_data.shape}, range: {np.min(raw_data)}-{np.max(raw_data)}")
        else:
            print(f"  1. Using provided RAW data: {raw_data.shape}, range: {np.min(raw_data)}-{np.max(raw_data)}")

        return process_raw_array(
            raw_data=raw_data,
            dark_data=dark_data,
            lens_shading_params=lens_shading_params,
            width=width,
            height=height,
            data_type=data_type,
            wb_params=wb_params,
            dark_subtraction_enabled=dark_subtraction_enabled,
            lens_shading_enabled=lens_shading_enabled,
            white_balance_enabled=white_balance_enabled,
            ccm_enabled=ccm_enabled,
            ccm_matrix_path=ccm_matrix_path,
            ccm_matrix=ccm_matrix,
            gamma_correction_enabled=gamma_correction_enabled,
            gamma_value=gamma_value,
            demosaic_output=demosaic_output,
        )
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
    
    # Show lens shading difference (lens_corrected - dark_corrected)
    try:
        if (lens_corrected is not None) and (dark_corrected is not None):
            diff = lens_corrected.astype(np.float64) - dark_corrected.astype(np.float64)
            vmax = np.max(np.abs(diff))
            vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
            im = axes[2, 2].imshow(diff, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            axes[2, 2].set_title('Lens Shading Diff', fontsize=12, fontweight='bold')
            axes[2, 2].axis('off')
        else:
            axes[2, 2].axis('off')
    except Exception:
        axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    
    plt.show()

def main():
    """Main function - simplified version"""
    print("=" * 60)
    print("ISP (Image Signal Processing) Pipeline")
    print("=" * 60)
    
    # 设置分辨率
    set_resolution_config(RESOLUTION)
    
    print(f"Input path: {INPUT_PATH}")
    print(f"Dark reference: {DARK_RAW_PATH}")
    print(f"Lens shading params: {LENS_SHADING_PARAMS_DIR}")
    if IMAGE_WIDTH and IMAGE_HEIGHT:
        print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    else:
        print("Image size: Auto-detected from RAW file")
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
    
    # Process each file - parameters are loaded automatically in process_single_image
    for i, raw_file in enumerate(raw_files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(raw_files)}: {raw_file.name}")
        print(f"{'='*60}")
        
        result = process_single_image(
            raw_file=str(raw_file),
            dark_data=DARK_RAW_PATH,  # Pass path, will be loaded automatically
            lens_shading_params=LENS_SHADING_PARAMS_DIR,  # Pass path, will be loaded automatically
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            data_type=DATA_TYPE,
            wb_params=WB_PARAMS_PATH,  # Pass path, will be loaded automatically
            dark_subtraction_enabled=DARK_SUBTRACTION_ENABLED,
            lens_shading_enabled=LENS_SHADING_ENABLED,
            white_balance_enabled=WHITE_BALANCE_ENABLED,
            ccm_enabled=CCM_ENABLED,
            ccm_matrix_path=CCM_MATRIX_PATH,  # No path needed when using global CCM_MATRIX
            ccm_matrix=np.array(CCM_MATRIX) if CCM_MATRIX else None,
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
                    output_file = output_dir / f"{raw_file.stem}_processed.png"
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
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='ISP (Image Signal Processing) Pipeline')
    parser.add_argument('--input', '-i', help='Input RAW file or directory path')
    parser.add_argument('--resolution', '-r', choices=['1k', '4k', 'auto'], 
                       default=RESOLUTION, help='Resolution preset: 1k (1920x1080), 4k (3840x2160), or auto')
    parser.add_argument('--width', '-W', type=int, help='Image width (overrides resolution preset)')
    parser.add_argument('--height', '-H', type=int, help='Image height (overrides resolution preset)')
    parser.add_argument('--dark', '-d', help='Dark reference RAW file path')
    parser.add_argument('--lens-shading', '-l', help='Lens shading parameters directory path')
    parser.add_argument('--output', '-o', help='Output directory path')
    
    # 尺寸检查参数
    parser.add_argument('--no-check-dimensions', action='store_true', help='Disable dimension compatibility checking')
    parser.add_argument('--force-correction', action='store_true', help='Force correction even if dimensions mismatch')
    
    args = parser.parse_args()
    
    # 更新全局配置
    if args.input:
        INPUT_PATH = args.input
    if args.resolution:
        RESOLUTION = args.resolution
    if args.width and args.height:
        IMAGE_WIDTH = args.width
        IMAGE_HEIGHT = args.height
        print(f"使用指定尺寸: {args.width}x{args.height}")
    if args.dark:
        DARK_RAW_PATH = args.dark
    if args.lens_shading:
        LENS_SHADING_PARAMS_DIR = args.lens_shading
    if args.output:
        OUTPUT_DIRECTORY = args.output
    
    # 更新尺寸检查选项
    if args.no_check_dimensions:
        CHECK_DIMENSIONS = False
        print("尺寸检查已禁用")
    if args.force_correction:
        SKIP_ON_DIMENSION_MISMATCH = False
        print("强制进行校正，即使尺寸不匹配")
    
    main()
