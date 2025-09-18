#!/usr/bin/env python3
"""
ISP和逆ISP对比测试脚本
实现：RAW → ISP → 逆ISP → RAW → ISP，并计算PSNR对比
"""

import numpy as np
import cv2
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入所需模块
try:
    from ISP import process_single_image, load_dark_reference, load_correction_parameters, load_white_balance_parameters, load_ccm_matrix
    from invert_ISP import invert_isp_pipeline, DEFAULT_CONFIG as INVERT_CONFIG
    from raw_reader import read_raw_image
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)

def detect_circular_roi(image: np.ndarray, threshold: float = 0.1) -> Tuple[int, int, int]:
    """
    检测内窥镜圆视场的有效区域（基于亮度检测）
    
    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        threshold: 亮度阈值，用于确定圆形边界
        
    Returns:
        (center_x, center_y, radius): 圆心坐标和半径
    """
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 归一化到0-1
    gray_norm = gray.astype(np.float32) / 255.0
    
    # 直接使用基于亮度的方法
    print("Using brightness-based method for circular ROI detection")
    return detect_circular_roi_brightness(gray_norm, threshold)

def detect_circular_roi_brightness(image: np.ndarray, threshold: float = 0.1) -> Tuple[int, int, int]:
    """
    基于亮度检测圆形ROI的主要方法
    
    Args:
        image: 归一化的灰度图像 (0-1)
        threshold: 亮度阈值
        
    Returns:
        (center_x, center_y, radius): 圆心坐标和半径
    """
    h, w = image.shape
    center_x, center_y = w // 2, h // 2
    
    # 确保圆心在图像范围内
    center_x = max(0, min(center_x, w - 1))
    center_y = max(0, min(center_y, h - 1))
    
    # 计算中心区域的亮度作为参考
    center_region_size = min(h, w) // 20  # 中心区域大小
    center_region = image[
        center_y - center_region_size//2:center_y + center_region_size//2,
        center_x - center_region_size//2:center_x + center_region_size//2
    ]
    center_brightness = np.mean(center_region)
    
    # 从中心向外搜索，找到亮度显著下降的位置
    max_radius = min(center_x, center_y, w - center_x, h - center_y) - 5
    max_radius = max(10, max_radius)  # 确保最小半径为10
    
    best_radius = 10
    best_score = 0
    
    for radius in range(10, max_radius, 2):  # 更细粒度的搜索
        # 检查圆形边界上的像素
        y, x = np.ogrid[:h, :w]
        boundary_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > (radius - 3) ** 2
        boundary_mask = boundary_mask & ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        
        if np.sum(boundary_mask) > 0:
            boundary_brightness = np.mean(image[boundary_mask])
            # 计算亮度下降的幅度
            brightness_drop = center_brightness - boundary_brightness
            # 计算得分：亮度下降越大，得分越高
            score = brightness_drop * radius  # 考虑半径，避免选择过小的圆
            
            if score > best_score:
                best_score = score
                best_radius = radius
    
    # 如果找到了合适的半径，使用它
    if best_score > threshold * center_brightness:
        print(f"Detected circular ROI (brightness): center=({center_x}, {center_y}), radius={best_radius}")
        print(f"  Center brightness: {center_brightness:.3f}, Boundary brightness: {boundary_brightness:.3f}")
        print(f"  Brightness drop: {best_score:.3f}")
        return center_x, center_y, best_radius
    
    # 如果没找到合适的半径，使用基于图像尺寸的默认值
    default_radius = min(center_x, center_y, w - center_x, h - center_y)
    radius = min(default_radius, min(h, w) // 3)
    radius = max(10, radius)  # 确保最小半径为10
    print(f"Using default circular ROI: center=({center_x}, {center_y}), radius={radius}")
    print(f"  Center brightness: {center_brightness:.3f}")
    return center_x, center_y, radius

def calculate_psnr_circular_roi(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0, 
                               threshold: float = 0.1, roi_enabled: bool = True) -> Tuple[float, Tuple[int, int, int]]:
    """
    计算两张图像在圆形ROI区域内的PSNR值
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        max_val: 图像的最大可能值
        threshold: 圆形ROI检测阈值
        roi_enabled: 是否启用ROI检测
        
    Returns:
        (psnr, roi_info): PSNR值（dB）和ROI信息 (center_x, center_y, radius)
    """
    # 确保两张图像尺寸相同
    if img1.shape != img2.shape:
        print(f"Warning: Image shapes don't match: {img1.shape} vs {img2.shape}")
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
    
    # 检测圆形ROI（如果启用）
    if roi_enabled:
        center_x, center_y, radius = detect_circular_roi(img1, threshold)
    else:
        # 如果禁用ROI，使用整个图像
        h, w = img1.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y, w - center_x, h - center_y)
        print(f"ROI detection disabled, using full image: center=({center_x}, {center_y}), radius={radius}")
    
    # 创建圆形掩码
    h, w = img1.shape[:2]
    
    # 再次确保ROI参数在图像范围内
    center_x = max(0, min(center_x, w - 1))
    center_y = max(0, min(center_y, h - 1))
    max_radius = min(center_x, center_y, w - center_x, h - center_y)
    radius = min(radius, max_radius)
    radius = max(1, radius)  # 确保半径至少为1
    
    y, x = np.ogrid[:h, :w]
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    
    # 应用掩码
    if len(img1.shape) == 3:
        # 彩色图像
        img1_roi = img1[mask]
        img2_roi = img2[mask]
    else:
        # 灰度图像
        img1_roi = img1[mask]
        img2_roi = img2[mask]
    
    if len(img1_roi) == 0:
        print("Warning: No valid pixels in circular ROI")
        return 0.0, (center_x, center_y, radius)
    
    # 计算MSE
    mse = np.mean((img1_roi.astype(np.float64) - img2_roi.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf'), (center_x, center_y, radius)  # 完全相同的图像
    
    # 计算PSNR
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    
    print(f"Circular ROI PSNR: {psnr:.2f} dB (ROI pixels: {len(img1_roi)})")
    return psnr, (center_x, center_y, radius)

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    计算两张图像之间的PSNR值（保持向后兼容）
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        max_val: 图像的最大可能值
        
    Returns:
        PSNR值（dB）
    """
    # 确保两张图像尺寸相同
    if img1.shape != img2.shape:
        print(f"Warning: Image shapes don't match: {img1.shape} vs {img2.shape}")
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
    
    # 计算MSE
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf')  # 完全相同的图像
    
    # 计算PSNR
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def calculate_psnr_16bit(img1: np.ndarray, img2: np.ndarray, max_val: float = 4095.0) -> float:
    """
    计算16bit图像之间的PSNR值
    
    Args:
        img1: 第一张16bit图像
        img2: 第二张16bit图像
        max_val: 图像的最大可能值（12bit数据在16bit容器中为4095）
        
    Returns:
        PSNR值（dB）
    """
    return calculate_psnr(img1, img2, max_val)

def save_comparison_images(original_isp: np.ndarray, reconstructed_isp: np.ndarray,
                          output_dir: Path, isp_roi_info: Tuple[int, int, int] = None) -> None:
    """
    保存对比图像，包括ROI可视化
    
    Args:
        original_isp: 原始ISP处理结果
        reconstructed_isp: 重建的ISP处理结果
        output_dir: 输出目录
        isp_roi_info: ISP图像ROI信息 (center_x, center_y, radius)
    """
    print("Saving comparison images...")
    
    # ISP图像对比
    if original_isp is not None:
        # 创建可写的副本
        original_isp_copy = original_isp.copy()
        # 添加ROI可视化到ISP图像
        if isp_roi_info is not None:
            center_x, center_y, radius = isp_roi_info
            cv2.circle(original_isp_copy, (center_x, center_y), radius, (0, 255, 0), 2)
            # 标记中心点
            cv2.circle(original_isp_copy, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.imwrite(str(output_dir / "original_isp.jpg"), original_isp_copy)
    
    if reconstructed_isp is not None:
        # 创建可写的副本
        reconstructed_isp_copy = reconstructed_isp.copy()
        # 添加ROI可视化到ISP图像
        if isp_roi_info is not None:
            center_x, center_y, radius = isp_roi_info
            cv2.circle(reconstructed_isp_copy, (center_x, center_y), radius, (0, 255, 0), 2)
            # 标记中心点
            cv2.circle(reconstructed_isp_copy, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.imwrite(str(output_dir / "reconstructed_isp.jpg"), reconstructed_isp_copy)
    
    # 创建对比图（只显示ISP图像）
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('ISP-Invert-ISP Comparison Results (with Circular ROI)', fontsize=16, fontweight='bold')
    
    # ISP对比
    if original_isp is not None:
        original_isp_rgb = cv2.cvtColor(original_isp, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_isp_rgb)
        axes[0].set_title('Original ISP Result', fontsize=12)
        # 添加ROI可视化到matplotlib图像
        if isp_roi_info is not None:
            center_x, center_y, radius = isp_roi_info
            circle = plt.Circle((center_x, center_y), radius, fill=False, color='green', linewidth=2)
            axes[0].add_patch(circle)
            axes[0].plot(center_x, center_y, 'ro', markersize=5)  # 中心点
    else:
        axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Original ISP Result', fontsize=12)
    axes[0].axis('off')
    
    if reconstructed_isp is not None:
        reconstructed_isp_rgb = cv2.cvtColor(reconstructed_isp, cv2.COLOR_BGR2RGB)
        axes[1].imshow(reconstructed_isp_rgb)
        axes[1].set_title('Reconstructed ISP Result', fontsize=12)
        # 添加ROI可视化到matplotlib图像
        if isp_roi_info is not None:
            center_x, center_y, radius = isp_roi_info
            circle = plt.Circle((center_x, center_y), radius, fill=False, color='green', linewidth=2)
            axes[1].add_patch(circle)
            axes[1].plot(center_x, center_y, 'ro', markersize=5)  # 中心点
    else:
        axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Reconstructed ISP Result', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    comparison_plot_path = output_dir / "comparison_results.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison images saved to: {output_dir}")

def isp_invert_isp_test(raw_file: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行完整的ISP-逆ISP-ISP测试流程
    
    Args:
        raw_file: 输入RAW文件路径
        config: 配置参数
        
    Returns:
        测试结果字典
    """
    print("=" * 80)
    print("ISP-Invert-ISP Comparison Test")
    print("=" * 80)
    print(f"Input RAW file: {raw_file}")
    print("=" * 80)
    
    results = {
        'test_success': False,
        'psnr_raw': None,
        'psnr_isp': None,
        'error': None,
        'original_raw_shape': None,
        'reconstructed_raw_shape': None
    }
    
    try:
        # 创建输出目录
        output_dir = Path(config['OUTPUT_DIRECTORY'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载原始RAW图像
        print("\n1. Loading original RAW image...")
        raw_data = read_raw_image(raw_file, config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'], config['DATA_TYPE'])
        if raw_data is None:
            raise ValueError(f"Failed to load RAW image: {raw_file}")
        
        print(f"  Original RAW: {raw_data.shape}, range: {np.min(raw_data)}-{np.max(raw_data)}")
        results['original_raw_shape'] = list(raw_data.shape)
        
        # 2. 第一次ISP处理
        print("\n2. First ISP processing...")
        isp_result = process_single_image(
            raw_file=raw_data,
            dark_data=config.get('dark_data'),
            lens_shading_params=config.get('lens_shading_params'),
            width=config['IMAGE_WIDTH'],
            height=config['IMAGE_HEIGHT'],
            data_type=config['DATA_TYPE'],
            wb_params=config.get('wb_params'),
            dark_subtraction_enabled=config.get('DARK_SUBTRACTION_ENABLED', True),
            lens_shading_enabled=config.get('LENS_SHADING_ENABLED', True),
            white_balance_enabled=config.get('WHITE_BALANCE_ENABLED', True),
            ccm_enabled=config.get('CCM_ENABLED', True),
            ccm_matrix_path=config.get('CCM_MATRIX_PATH'),
            ccm_matrix=config.get('ccm_matrix'),
            gamma_correction_enabled=config.get('GAMMA_CORRECTION_ENABLED', True),
            gamma_value=config.get('GAMMA_VALUE', 2.2),
            demosaic_output=config.get('DEMOSAIC_OUTPUT', True)
        )
        
        if not isp_result['processing_success']:
            raise ValueError(f"First ISP processing failed: {isp_result.get('error', 'Unknown error')}")
        
        print(f"  First ISP completed successfully")
        # 不存储大型结果对象，只存储关键信息
        results['first_isp_success'] = isp_result['processing_success']
        
        # 3. 逆ISP处理
        print("\n3. Inverse ISP processing...")
        invert_config = INVERT_CONFIG.copy()
        invert_config.update({
            'INPUT_IMAGE_PATH': str(output_dir / "temp_isp_result.png"),
            'OUTPUT_RAW_PATH': str(output_dir / "reconstructed.raw"),
            'IMAGE_WIDTH': config['IMAGE_WIDTH'],
            'IMAGE_HEIGHT': config['IMAGE_HEIGHT'],
            'DATA_TYPE': config['DATA_TYPE'],
            'BAYER_PATTERN': 'rggb',
            'SAVE_INTERMEDIATE': False,
            'DISPLAY_RAW_GRAYSCALE': False,
            'SAVE_RAW_GRAYSCALE': False,
            'CREATE_COMPARISON_PLOT': False,
            # 使用与ISP相同的处理开关
            'DARK_SUBTRACTION_ENABLED': config.get('DARK_SUBTRACTION_ENABLED', True),
            'LENS_SHADING_ENABLED': config.get('LENS_SHADING_ENABLED', True),
            'WHITE_BALANCE_ENABLED': config.get('WHITE_BALANCE_ENABLED', True),
            'CCM_ENABLED': config.get('CCM_ENABLED', True),
            'GAMMA_CORRECTION_ENABLED': config.get('GAMMA_CORRECTION_ENABLED', True),
            'GAMMA_VALUE': config.get('GAMMA_VALUE', 2.2),
            # 使用与ISP相同的参数路径
            'DARK_RAW_PATH': config.get('DARK_RAW_PATH'),
            'LENS_SHADING_PARAMS_DIR': config.get('LENS_SHADING_PARAMS_DIR'),
            'WB_PARAMS_PATH': config.get('WB_PARAMS_PATH'),
            'CCM_MATRIX_PATH': config.get('CCM_MATRIX_PATH'),
            # 使用与ISP相同的直接参数
            'ccm_matrix': config.get('ccm_matrix'),
            'wb_params': config.get('wb_params'),
            'dark_data': config.get('dark_data'),
            'lens_shading_params': config.get('lens_shading_params')
        })
        
        # 保存ISP结果为临时图像
        if isp_result['color_img'] is not None:
            cv2.imwrite(invert_config['INPUT_IMAGE_PATH'], isp_result['color_img'])
        
        
        invert_result = invert_isp_pipeline(invert_config['INPUT_IMAGE_PATH'], invert_config)
        if not invert_result['processing_success']:
            raise ValueError(f"Inverse ISP processing failed: {invert_result.get('error', 'Unknown error')}")
        
        print(f"  Inverse ISP completed successfully")
        results['invert_isp_success'] = invert_result['processing_success']
        
        # 4. 加载重建的RAW图像
        print("\n4. Loading reconstructed RAW image...")
        reconstructed_raw = read_raw_image(
            invert_config['OUTPUT_RAW_PATH'], 
            config['IMAGE_WIDTH'], 
            config['IMAGE_HEIGHT'], 
            config['DATA_TYPE']
        )
        
        if reconstructed_raw is None:
            raise ValueError("Failed to load reconstructed RAW image")
        
        print(f"  Reconstructed RAW: {reconstructed_raw.shape}, range: {np.min(reconstructed_raw)}-{np.max(reconstructed_raw)}")
        results['reconstructed_raw_shape'] = list(reconstructed_raw.shape)
        
        # 5. 跳过RAW图像PSNR计算
        print("\n5. Skipping RAW PSNR calculation as requested")
        
        # 6. 第二次ISP处理（使用重建的RAW）
        print("\n6. Second ISP processing (using reconstructed RAW)...")
        second_isp_result = process_single_image(
            raw_file=invert_config['OUTPUT_RAW_PATH'],
            dark_data=config.get('dark_data'),
            lens_shading_params=config.get('lens_shading_params'),
            width=config['IMAGE_WIDTH'],
            height=config['IMAGE_HEIGHT'],
            data_type=config['DATA_TYPE'],
            wb_params=config.get('wb_params'),
            dark_subtraction_enabled=config.get('DARK_SUBTRACTION_ENABLED', True),
            lens_shading_enabled=config.get('LENS_SHADING_ENABLED', True),
            white_balance_enabled=config.get('WHITE_BALANCE_ENABLED', True),
            ccm_enabled=config.get('CCM_ENABLED', True),
            ccm_matrix_path=config.get('CCM_MATRIX_PATH'),
            ccm_matrix=config.get('ccm_matrix'),
            gamma_correction_enabled=config.get('GAMMA_CORRECTION_ENABLED', True),
            gamma_value=config.get('GAMMA_VALUE', 2.2),
            demosaic_output=config.get('DEMOSAIC_OUTPUT', True)
        )
        
        if not second_isp_result['processing_success']:
            raise ValueError(f"Second ISP processing failed: {second_isp_result.get('error', 'Unknown error')}")
        
        print(f"  Second ISP completed successfully")
        results['second_isp_success'] = second_isp_result['processing_success']
        
        # 7. 计算ISP图像的PSNR（使用圆形ROI）
        print("\n7. Calculating ISP PSNR with circular ROI...")
        if isp_result['color_img'] is not None and second_isp_result['color_img'] is not None:
            roi_enabled = config.get('ROI_ENABLED', True)  # 从配置中获取ROI enable参数
            isp_psnr, isp_roi_info = calculate_psnr_circular_roi(isp_result['color_img'], second_isp_result['color_img'], 255.0, roi_enabled=roi_enabled)
            print(f"  ISP PSNR (circular ROI): {isp_psnr:.2f} dB")
            if roi_enabled and isp_roi_info is not None:
                print(f"  ROI info: center=({isp_roi_info[0]}, {isp_roi_info[1]}), radius={isp_roi_info[2]}")
            results['psnr_isp'] = isp_psnr
            # 当 ROI 未启用时，不在绘图中显示圆框
            results['isp_roi_info'] = isp_roi_info if roi_enabled else None
        else:
            print("  Warning: Cannot calculate ISP PSNR - one or both ISP results are None")
            results['psnr_isp'] = None
            results['isp_roi_info'] = None
        
        # 8. 保存对比图像
        print("\n8. Saving comparison images...")
        # 当 ROI 未启用时，不在plot中显示圆框（传入 None）
        save_comparison_images(
            isp_result['color_img'], second_isp_result['color_img'],
            output_dir, results.get('isp_roi_info')
        )
    except:
        print(f"Error in ISP-Invert-ISP test: {e}")
    
    return True

def main():
    """主函数"""
    # 配置参数
    config = {
        # 输入输出配置
        'INPUT_RAW_PATH':r"F:\ZJU\Picture\invert_isp\inverted_output.raw",  # 输入RAW文件路径
        'OUTPUT_DIRECTORY': r"F:\ZJU\Picture\isp_invert_isp_test",  # 输出目录
        
        # 图像参数 - 可选择 1K 或 4K 分辨率
        'RESOLUTION': '1K',  # 可选: '1K' (1920x1080) 或 '4K' (3840x2160)
        'DATA_TYPE': 'uint16',
        
        # ISP参数路径
        'DARK_RAW_PATH': r"F:\ZJU\Picture\dark\g3\average_dark.raw",
        'LENS_SHADING_PARAMS_DIR': r"F:\ZJU\Picture\lens shading\regress",  
        'WB_PARAMS_PATH': r"F:\ZJU\Picture\wb\wb_output",
        'CCM_MATRIX_PATH': r"F:\ZJU\Picture\ccm\ccm_2\ccm_output_20250905_162714",
        
        # 处理开关
        'ROI_ENABLED': False,  # 是否启用ROI检测
        'LENS_SHADING_ENABLED': False,
        'WHITE_BALANCE_ENABLED': True,
        'CCM_ENABLED': True,
        'GAMMA_CORRECTION_ENABLED': True,
        'GAMMA_VALUE': 2.2,
        'DEMOSAIC_OUTPUT': True,
        
        # 直接参数（优先使用）
        'ccm_matrix': np.array([
            [1.7801320111582375, -0.7844420268663381, 0.004310015708100662],
            [-0.24377094860030846, 2.4432181685707977, -1.1994472199704893],
            [-0.4715762768203783, -0.7105721829898775, 2.182148459810256]
        ]),
        'wb_params': {
            "white_balance_gains": {
                "b_gain": 2.168214315103357,
                "g_gain": 1.0,
                "r_gain": 1.3014453071420942
            }
        }
    }
    
    # 解析分辨率参数
    resolution = config['RESOLUTION'].upper()
    if resolution == '1K':
        config['IMAGE_WIDTH'] = 1920
        config['IMAGE_HEIGHT'] = 1080
        print("Using 1K resolution: 1920x1080")
    elif resolution == '4K':
        config['IMAGE_WIDTH'] = 3840
        config['IMAGE_HEIGHT'] = 2160
        print("Using 4K resolution: 3840x2160")
    else:
        raise ValueError(f"Unsupported resolution: {resolution}. Please use '1K' or '4K'")
    
    # 加载必要的参数
    print("Loading ISP parameters...")
    
    # 加载暗电流数据
    dark_data = load_dark_reference(
        config['DARK_RAW_PATH'], 
        config['IMAGE_WIDTH'], 
        config['IMAGE_HEIGHT'], 
        config['DATA_TYPE']
    )
    config['dark_data'] = dark_data
    
    # 加载镜头阴影参数
    lens_shading_params = load_correction_parameters(config['LENS_SHADING_PARAMS_DIR'])
    config['lens_shading_params'] = lens_shading_params
    
    # 加载白平衡参数
    wb_params = load_white_balance_parameters(config['WB_PARAMS_PATH'])
    config['wb_params'] = wb_params
    
    # 执行测试
    result = isp_invert_isp_test(config['INPUT_RAW_PATH'], config)
    

    print(f"\n✅ ISP-Invert-ISP test completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())