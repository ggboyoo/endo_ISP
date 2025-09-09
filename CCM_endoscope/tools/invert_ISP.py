#!/usr/bin/env python3
"""
Inverse ISP (Image Signal Processing) Script
逆ISP处理程序 - 将sRGB图像逆向转换为RAW数据

处理流程：
1. 读入任意处理过的sRGB图片（jpg，png等），转化为12bit数据，以16uint类型储存
2. 逆gamma校正
3. 根据提供的CCM矩阵参数，逆CCM变换
4. 逆白平衡
5. 逆马赛克（Bayer插值）
6. 得到uint16的RAW图，并输出保存为.raw
"""

import numpy as np
import cv2
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import argparse

# 导入现有模块
try:
    from ccm_apply import load_matrix_from_json, apply_ccm
    from files_isp import imread_unicode, imwrite_unicode
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not be available.")

# ============================================================================
# 配置参数
# ============================================================================

# 默认配置
DEFAULT_CONFIG = {
    # 输入输出配置
    'INPUT_IMAGE_PATH': None,  # 输入sRGB图像路径
    'OUTPUT_RAW_PATH': None,   # 输出RAW文件路径
    'CCM_MATRIX_PATH': None,   # CCM矩阵文件路径
    'WB_PARAMS_PATH': None,    # 白平衡参数文件路径
    
    # 图像参数
    'IMAGE_WIDTH': 3840,       # 图像宽度
    'IMAGE_HEIGHT': 2160,      # 图像高度
    'BAYER_PATTERN': 'rggb',   # Bayer模式
    
    # 处理选项
    'GAMMA_VALUE': 2.2,        # 伽马值（用于逆伽马校正）
    'OUTPUT_BIT_DEPTH': 12,    # 输出位深（12bit数据存储在16bit容器中）
    'SAVE_INTERMEDIATE': True, # 是否保存中间结果
    'VERBOSE': True,           # 是否显示详细信息
}

# ============================================================================
# 核心函数
# ============================================================================

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """
    将sRGB值转换为线性RGB值
    
    Args:
        srgb: sRGB值，范围[0, 1]
        
    Returns:
        线性RGB值，范围[0, 1]
    """
    srgb = np.clip(srgb, 0.0, 1.0)
    threshold = 0.04045
    low = srgb <= threshold
    high = ~low
    out = np.zeros_like(srgb)
    out[low] = srgb[low] / 12.92
    out[high] = ((srgb[high] + 0.055) / 1.055) ** 2.4
    return out

def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """
    将线性RGB值转换为sRGB值
    
    Args:
        lin: 线性RGB值，范围[0, 1]
        
    Returns:
        sRGB值，范围[0, 1]
    """
    lin = np.clip(lin, 0.0, 1.0)
    threshold = 0.0031308
    low = lin <= threshold
    high = ~low
    out = np.zeros_like(lin)
    out[low] = lin[low] * 12.92
    out[high] = 1.055 * (lin[high] ** (1 / 2.4)) - 0.055
    return out

def load_image_as_12bit(image_path: str, target_width: int, target_height: int) -> np.ndarray:
    """
    加载图像并转换为12bit数据（存储在16bit容器中）
    
    Args:
        image_path: 图像文件路径
        target_width: 目标宽度
        target_height: 目标高度
        
    Returns:
        12bit数据（uint16格式，范围0-4095）
    """
    print(f"Loading image: {image_path}")
    
    # 读取图像
    if os.path.exists(image_path):
        try:
            # 使用OpenCV读取
            img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                # 尝试使用unicode读取
                img_bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
        except:
            img_bgr = None
    else:
        img_bgr = None
    
    if img_bgr is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    print(f"  Original image shape: {img_bgr.shape}")
    print(f"  Original image range: {np.min(img_bgr)}-{np.max(img_bgr)}")
    
    # 调整图像尺寸
    if img_bgr.shape[:2] != (target_height, target_width):
        print(f"  Resizing from {img_bgr.shape[:2]} to ({target_height}, {target_width})")
        img_bgr = cv2.resize(img_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # 转换为RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 归一化到[0, 1]
    img_normalized = img_rgb.astype(np.float64) / 255.0
    
    # 转换为12bit数据（存储在16bit容器中）
    img_12bit = (img_normalized * 4095.0).astype(np.uint16)
    
    print(f"  Converted to 12bit: {img_12bit.shape}, range: {np.min(img_12bit)}-{np.max(img_12bit)}")
    
    return img_12bit

def inverse_gamma_correction(img_12bit: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    逆伽马校正
    
    Args:
        img_12bit: 12bit图像数据（uint16格式，范围0-4095）
        gamma: 伽马值
        
    Returns:
        逆伽马校正后的12bit图像数据
    """
    print(f"Applying inverse gamma correction (gamma={gamma})...")
    
    # 转换为浮点数并归一化
    img_float = img_12bit.astype(np.float64) / 4095.0
    
    # 应用逆伽马校正
    img_inverse_gamma = np.power(img_float, gamma)
    
    # 转换回12bit数据
    img_corrected = (img_inverse_gamma * 4095.0).astype(np.uint16)
    
    print(f"  Inverse gamma correction applied: range {np.min(img_corrected)}-{np.max(img_corrected)}")
    
    return img_corrected

def load_ccm_matrix(ccm_path: str) -> Tuple[np.ndarray, str]:
    """
    加载CCM矩阵
    
    Args:
        ccm_path: CCM矩阵文件路径
        
    Returns:
        (CCM矩阵, 矩阵类型)
    """
    print(f"Loading CCM matrix from: {ccm_path}")
    
    try:
        # 查找JSON文件
        if os.path.isfile(ccm_path):
            json_file = ccm_path
        else:
            # 在目录中查找JSON文件
            ccm_dir = Path(ccm_path)
            json_files = list(ccm_dir.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in: {ccm_path}")
            json_file = str(json_files[0])
        
        with open(json_file, 'r', encoding='utf-8') as f:
            ccm_data = json.load(f)
        
        # 提取CCM矩阵和类型
        if 'ccm_matrix' in ccm_data:
            matrix = np.array(ccm_data['ccm_matrix'])
            matrix_type = ccm_data.get('ccm_type', 'linear3x3')
        elif 'matrix' in ccm_data:
            matrix = np.array(ccm_data['matrix'])
            matrix_type = ccm_data.get('type', 'linear3x3')
        else:
            raise ValueError(f"No CCM matrix found in JSON file: {json_file}")
        
        print(f"  CCM matrix loaded: {matrix.shape}, type: {matrix_type}")
        return matrix, matrix_type
        
    except Exception as e:
        print(f"Error loading CCM matrix: {e}")
        raise

def inverse_ccm_correction(img_12bit: np.ndarray, ccm_matrix: np.ndarray, matrix_type: str) -> np.ndarray:
    """
    逆CCM校正
    
    Args:
        img_12bit: 12bit图像数据
        ccm_matrix: CCM矩阵
        matrix_type: 矩阵类型（'linear3x3' 或 'affine3x4'）
        
    Returns:
        逆CCM校正后的12bit图像数据
    """
    print(f"Applying inverse CCM correction ({matrix_type})...")
    
    # 转换为浮点数
    img_float = img_12bit.astype(np.float64)
    
    # 重塑为2D数组用于矩阵运算
    h, w, c = img_float.shape
    img_flat = img_float.reshape(-1, 3)
    
    # 计算逆矩阵
    if matrix_type == 'linear3x3':
        # 3x3线性变换的逆
        try:
            inv_matrix = np.linalg.inv(ccm_matrix)
            corrected_flat = np.dot(img_flat, inv_matrix.T)
        except np.linalg.LinAlgError:
            print("  Warning: Matrix is singular, using pseudo-inverse")
            inv_matrix = np.linalg.pinv(ccm_matrix)
            corrected_flat = np.dot(img_flat, inv_matrix.T)
    elif matrix_type == 'affine3x4':
        # 3x4仿射变换的逆
        # 对于仿射变换 [R|t]，逆变换为 [R^-1|-R^-1*t]
        R = ccm_matrix[:, :3]
        t = ccm_matrix[:, 3]
        try:
            R_inv = np.linalg.inv(R)
            t_inv = -np.dot(R_inv, t)
            inv_matrix = np.column_stack([R_inv, t_inv])
            # 添加齐次坐标
            img_with_bias = np.column_stack([img_flat, np.ones(img_flat.shape[0])])
            corrected_flat = np.dot(img_with_bias, inv_matrix.T)
        except np.linalg.LinAlgError:
            print("  Warning: Matrix is singular, using pseudo-inverse")
            R_pinv = np.linalg.pinv(R)
            t_pinv = -np.dot(R_pinv, t)
            inv_matrix = np.column_stack([R_pinv, t_pinv])
            img_with_bias = np.column_stack([img_flat, np.ones(img_flat.shape[0])])
            corrected_flat = np.dot(img_with_bias, inv_matrix.T)
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")
    
    # 重塑回原始形状
    corrected = corrected_flat.reshape(h, w, 3)
    
    # 裁剪到有效范围并转换回12bit
    corrected = np.clip(corrected, 0, 4095).astype(np.uint16)
    
    print(f"  Inverse CCM correction applied: range {np.min(corrected)}-{np.max(corrected)}")
    
    return corrected

def load_white_balance_parameters(wb_path: str) -> Dict[str, float]:
    """
    加载白平衡参数
    
    Args:
        wb_path: 白平衡参数文件路径
        
    Returns:
        白平衡参数字典
    """
    print(f"Loading white balance parameters from: {wb_path}")
    
    try:
        # 查找JSON文件
        if os.path.isfile(wb_path):
            json_file = wb_path
        else:
            # 在目录中查找JSON文件
            wb_dir = Path(wb_path)
            json_files = list(wb_dir.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in: {wb_path}")
            json_file = str(json_files[0])
        
        with open(json_file, 'r', encoding='utf-8') as f:
            wb_data = json.load(f)
        
        # 提取白平衡增益
        if 'white_balance_gains' in wb_data:
            wb_params = wb_data['white_balance_gains']
        else:
            raise ValueError(f"No white_balance_gains found in JSON file: {json_file}")
        
        print(f"  White balance parameters loaded: {wb_params}")
        return wb_params
        
    except Exception as e:
        print(f"Error loading white balance parameters: {e}")
        raise

def inverse_white_balance_correction(img_12bit: np.ndarray, wb_params: Dict[str, float]) -> np.ndarray:
    """
    逆白平衡校正
    
    Args:
        img_12bit: 12bit图像数据
        wb_params: 白平衡参数字典
        
    Returns:
        逆白平衡校正后的12bit图像数据
    """
    print("Applying inverse white balance correction...")
    
    # 提取白平衡增益
    r_gain = wb_params.get('r_gain', 1.0)
    g_gain = wb_params.get('g_gain', 1.0)
    b_gain = wb_params.get('b_gain', 1.0)
    
    print(f"  White balance gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")
    
    # 应用逆白平衡增益
    corrected = img_12bit.copy().astype(np.float64)
    corrected[:, :, 0] /= b_gain  # B通道
    corrected[:, :, 1] /= g_gain  # G通道
    corrected[:, :, 2] /= r_gain  # R通道
    
    # 裁剪到有效范围并转换回12bit
    corrected = np.clip(corrected, 0, 4095).astype(np.uint16)
    
    print(f"  Inverse white balance correction applied: range {np.min(corrected)}-{np.max(corrected)}")
    
    return corrected

def inverse_demosaic(img_12bit: np.ndarray, bayer_pattern: str = 'rggb') -> np.ndarray:
    """
    逆马赛克（Bayer插值）- 将RGB图像转换为Bayer RAW格式
    
    Args:
        img_12bit: 12bit RGB图像数据
        bayer_pattern: Bayer模式
        
    Returns:
        Bayer RAW数据（uint16格式）
    """
    print(f"Applying inverse demosaicing (Bayer pattern: {bayer_pattern})...")
    
    h, w, c = img_12bit.shape
    
    # 创建Bayer RAW数组
    raw_data = np.zeros((h, w), dtype=np.uint16)
    
    if bayer_pattern.lower() == 'rggb':
        # RGGB模式
        raw_data[0::2, 0::2] = img_12bit[0::2, 0::2, 2]  # R
        raw_data[0::2, 1::2] = img_12bit[0::2, 1::2, 1]  # G
        raw_data[1::2, 0::2] = img_12bit[1::2, 0::2, 1]  # G
        raw_data[1::2, 1::2] = img_12bit[1::2, 1::2, 0]  # B
    elif bayer_pattern.lower() == 'bggr':
        # BGGR模式
        raw_data[0::2, 0::2] = img_12bit[0::2, 0::2, 0]  # B
        raw_data[0::2, 1::2] = img_12bit[0::2, 1::2, 1]  # G
        raw_data[1::2, 0::2] = img_12bit[1::2, 0::2, 1]  # G
        raw_data[1::2, 1::2] = img_12bit[1::2, 1::2, 2]  # R
    elif bayer_pattern.lower() == 'grbg':
        # GRBG模式
        raw_data[0::2, 0::2] = img_12bit[0::2, 0::2, 1]  # G
        raw_data[0::2, 1::2] = img_12bit[0::2, 1::2, 2]  # R
        raw_data[1::2, 0::2] = img_12bit[1::2, 0::2, 0]  # B
        raw_data[1::2, 1::2] = img_12bit[1::2, 1::2, 1]  # G
    elif bayer_pattern.lower() == 'gbrg':
        # GBRG模式
        raw_data[0::2, 0::2] = img_12bit[0::2, 0::2, 1]  # G
        raw_data[0::2, 1::2] = img_12bit[0::2, 1::2, 0]  # B
        raw_data[1::2, 0::2] = img_12bit[1::2, 0::2, 2]  # R
        raw_data[1::2, 1::2] = img_12bit[1::2, 1::2, 1]  # G
    else:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
    
    print(f"  Inverse demosaicing completed: {raw_data.shape}, range {np.min(raw_data)}-{np.max(raw_data)}")
    
    return raw_data

def save_raw_data(raw_data: np.ndarray, output_path: str) -> None:
    """
    保存RAW数据为.raw文件
    
    Args:
        raw_data: RAW数据数组
        output_path: 输出文件路径
    """
    print(f"Saving RAW data to: {output_path}")
    
    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为二进制文件
    raw_data.astype(np.uint16).tofile(output_path)
    
    print(f"  RAW data saved: {raw_data.shape}, range {np.min(raw_data)}-{np.max(raw_data)}")

def save_intermediate_image(img_data: np.ndarray, output_path: str, is_12bit: bool = True) -> None:
    """
    保存中间结果图像
    
    Args:
        img_data: 图像数据
        output_path: 输出文件路径
        is_12bit: 是否为12bit数据
    """
    if not is_12bit:
        # 8bit数据直接保存
        cv2.imwrite(output_path, img_data)
    else:
        # 12bit数据需要转换为8bit保存
        if len(img_data.shape) == 3:
            # 彩色图像
            img_8bit = (img_data.astype(np.float32) / 4095 * 255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img_bgr)
        else:
            # 灰度图像
            img_8bit = (img_data.astype(np.float32) / 4095 * 255.0).astype(np.uint8)
            cv2.imwrite(output_path, img_8bit)
    
    print(f"  Intermediate image saved: {output_path}")

def invert_isp_pipeline(image_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    完整的逆ISP处理流程
    
    Args:
        image_path: 输入图像路径
        config: 配置参数字典
        
    Returns:
        处理结果字典
    """
    print("=" * 60)
    print("Inverse ISP Processing Pipeline")
    print("=" * 60)
    print(f"Input image: {image_path}")
    print(f"Target size: {config['IMAGE_WIDTH']}x{config['IMAGE_HEIGHT']}")
    print(f"Bayer pattern: {config['BAYER_PATTERN']}")
    print("=" * 60)
    
    results = {}
    
    try:
        # 1. 加载图像并转换为12bit数据
        print("\n1. Loading and converting to 12bit data...")
        img_12bit = load_image_as_12bit(image_path, config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'])
        results['step1_12bit'] = img_12bit
        
        if config['SAVE_INTERMEDIATE']:
            output_dir = Path(config['OUTPUT_RAW_PATH']).parent
            save_intermediate_image(img_12bit, str(output_dir / "step1_12bit.png"), is_12bit=True)
        
        # 2. 逆伽马校正
        print("\n2. Applying inverse gamma correction...")
        img_inverse_gamma = inverse_gamma_correction(img_12bit, config['GAMMA_VALUE'])
        results['step2_inverse_gamma'] = img_inverse_gamma
        
        if config['SAVE_INTERMEDIATE']:
            output_dir = Path(config['OUTPUT_RAW_PATH']).parent
            save_intermediate_image(img_inverse_gamma, str(output_dir / "step2_inverse_gamma.png"), is_12bit=True)
        
        # 3. 逆CCM校正
        if config['CCM_MATRIX_PATH']:
            print("\n3. Applying inverse CCM correction...")
            ccm_matrix, matrix_type = load_ccm_matrix(config['CCM_MATRIX_PATH'])
            img_inverse_ccm = inverse_ccm_correction(img_inverse_gamma, ccm_matrix, matrix_type)
            results['step3_inverse_ccm'] = img_inverse_ccm
            
            if config['SAVE_INTERMEDIATE']:
                output_dir = Path(config['OUTPUT_RAW_PATH']).parent
                save_intermediate_image(img_inverse_ccm, str(output_dir / "step3_inverse_ccm.png"), is_12bit=True)
        else:
            print("\n3. Skipping inverse CCM correction (no matrix provided)")
            img_inverse_ccm = img_inverse_gamma
            results['step3_inverse_ccm'] = img_inverse_ccm
        
        # 4. 逆白平衡校正
        if config['WB_PARAMS_PATH']:
            print("\n4. Applying inverse white balance correction...")
            wb_params = load_white_balance_parameters(config['WB_PARAMS_PATH'])
            img_inverse_wb = inverse_white_balance_correction(img_inverse_ccm, wb_params)
            results['step4_inverse_wb'] = img_inverse_wb
            
            if config['SAVE_INTERMEDIATE']:
                output_dir = Path(config['OUTPUT_RAW_PATH']).parent
                save_intermediate_image(img_inverse_wb, str(output_dir / "step4_inverse_wb.png"), is_12bit=True)
        else:
            print("\n4. Skipping inverse white balance correction (no parameters provided)")
            img_inverse_wb = img_inverse_ccm
            results['step4_inverse_wb'] = img_inverse_wb
        
        # 5. 逆马赛克（Bayer插值）
        print("\n5. Applying inverse demosaicing...")
        raw_data = inverse_demosaic(img_inverse_wb, config['BAYER_PATTERN'])
        results['step5_raw'] = raw_data
        
        # 6. 保存RAW数据
        print("\n6. Saving RAW data...")
        save_raw_data(raw_data, config['OUTPUT_RAW_PATH'])
        results['output_path'] = config['OUTPUT_RAW_PATH']
        
        # 保存处理报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_image': image_path,
            'output_raw': config['OUTPUT_RAW_PATH'],
            'config': config,
            'processing_success': True,
            'image_info': {
                'original_shape': img_12bit.shape,
                'raw_shape': raw_data.shape,
                'raw_range': [int(np.min(raw_data)), int(np.max(raw_data))],
                'raw_dtype': str(raw_data.dtype)
            }
        }
        
        report_path = Path(config['OUTPUT_RAW_PATH']).parent / "inverse_isp_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessing report saved: {report_path}")
        print("\n" + "=" * 60)
        print("Inverse ISP Processing Complete!")
        print("=" * 60)
        
        results['processing_success'] = True
        results['report_path'] = str(report_path)
        
        return results
        
    except Exception as e:
        print(f"\nError in inverse ISP processing: {e}")
        results['processing_success'] = False
        results['error'] = str(e)
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Inverse ISP Processing - Convert sRGB to RAW')
    parser.add_argument('--input', '-i', required=True, help='Input sRGB image path')
    parser.add_argument('--output', '-o', required=True, help='Output RAW file path')
    parser.add_argument('--width', '-w', type=int, default=DEFAULT_CONFIG['IMAGE_WIDTH'], help='Image width')
    parser.add_argument('--height', '-h', type=int, default=DEFAULT_CONFIG['IMAGE_HEIGHT'], help='Image height')
    parser.add_argument('--bayer', '-b', choices=['rggb', 'bggr', 'grbg', 'gbrg'], 
                       default=DEFAULT_CONFIG['BAYER_PATTERN'], help='Bayer pattern')
    parser.add_argument('--ccm', help='CCM matrix file path')
    parser.add_argument('--wb', help='White balance parameters file path')
    parser.add_argument('--gamma', type=float, default=DEFAULT_CONFIG['GAMMA_VALUE'], help='Gamma value')
    parser.add_argument('--no-save-intermediate', action='store_true', help='Do not save intermediate results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # 构建配置
    config = DEFAULT_CONFIG.copy()
    config['INPUT_IMAGE_PATH'] = args.input
    config['OUTPUT_RAW_PATH'] = args.output
    config['IMAGE_WIDTH'] = args.width
    config['IMAGE_HEIGHT'] = args.height
    config['BAYER_PATTERN'] = args.bayer
    config['CCM_MATRIX_PATH'] = args.ccm
    config['WB_PARAMS_PATH'] = args.wb
    config['GAMMA_VALUE'] = args.gamma
    config['SAVE_INTERMEDIATE'] = not args.no_save_intermediate
    config['VERBOSE'] = args.verbose
    
    # 执行逆ISP处理
    result = invert_isp_pipeline(args.input, config)
    
    if result['processing_success']:
        print(f"\n✅ Inverse ISP processing completed successfully!")
        print(f"   Input: {args.input}")
        print(f"   Output: {args.output}")
        if 'report_path' in result:
            print(f"   Report: {result['report_path']}")
        return 0
    else:
        print(f"\n❌ Inverse ISP processing failed: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    exit(main())
