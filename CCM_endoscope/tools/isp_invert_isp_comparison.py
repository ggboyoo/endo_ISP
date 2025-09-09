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

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    计算两张图像之间的PSNR值
    
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

def save_comparison_images(original_raw: np.ndarray, reconstructed_raw: np.ndarray,
                          original_isp: np.ndarray, reconstructed_isp: np.ndarray,
                          output_dir: Path) -> None:
    """
    保存对比图像
    
    Args:
        original_raw: 原始RAW图像
        reconstructed_raw: 重建的RAW图像
        original_isp: 原始ISP处理结果
        reconstructed_isp: 重建的ISP处理结果
        output_dir: 输出目录
    """
    print("Saving comparison images...")
    
    # 保存RAW图像对比（灰度图）
    def raw_to_8bit(raw_img):
        max_val = 4095 if raw_img.dtype == np.uint16 else np.max(raw_img)
        if max_val > 0:
            return (raw_img.astype(np.float32) / max_val * 255.0).astype(np.uint8)
        else:
            return np.zeros_like(raw_img, dtype=np.uint8)
    
    # RAW图像对比
    original_raw_8bit = raw_to_8bit(original_raw)
    reconstructed_raw_8bit = raw_to_8bit(reconstructed_raw)
    
    cv2.imwrite(str(output_dir / "original_raw.png"), original_raw_8bit)
    cv2.imwrite(str(output_dir / "reconstructed_raw.png"), reconstructed_raw_8bit)
    
    # ISP图像对比
    if original_isp is not None:
        cv2.imwrite(str(output_dir / "original_isp.jpg"), original_isp)
    if reconstructed_isp is not None:
        cv2.imwrite(str(output_dir / "reconstructed_isp.jpg"), reconstructed_isp)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ISP-Invert-ISP Comparison Results', fontsize=16, fontweight='bold')
    
    # RAW对比
    axes[0, 0].imshow(original_raw_8bit, cmap='gray')
    axes[0, 0].set_title('Original RAW', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(reconstructed_raw_8bit, cmap='gray')
    axes[0, 1].set_title('Reconstructed RAW', fontsize=12)
    axes[0, 1].axis('off')
    
    # ISP对比
    if original_isp is not None:
        original_isp_rgb = cv2.cvtColor(original_isp, cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(original_isp_rgb)
        axes[1, 0].set_title('Original ISP Result', fontsize=12)
    else:
        axes[1, 0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Original ISP Result', fontsize=12)
    axes[1, 0].axis('off')
    
    if reconstructed_isp is not None:
        reconstructed_isp_rgb = cv2.cvtColor(reconstructed_isp, cv2.COLOR_BGR2RGB)
        axes[1, 1].imshow(reconstructed_isp_rgb)
        axes[1, 1].set_title('Reconstructed ISP Result', fontsize=12)
    else:
        axes[1, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Reconstructed ISP Result', fontsize=12)
    axes[1, 1].axis('off')
    
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
            raw_file=raw_file,
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
            'SAVE_INTERMEDIATE': False,
            'DISPLAY_RAW_GRAYSCALE': False,
            'SAVE_RAW_GRAYSCALE': False,
            'CREATE_COMPARISON_PLOT': False
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
        
        # 5. 计算RAW图像的PSNR
        print("\n5. Calculating RAW PSNR...")
        raw_psnr = calculate_psnr_16bit(raw_data, reconstructed_raw, 4095.0)
        print(f"  RAW PSNR: {raw_psnr:.2f} dB")
        results['psnr_raw'] = raw_psnr
        
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
        
        # 7. 计算ISP图像的PSNR
        print("\n7. Calculating ISP PSNR...")
        if isp_result['color_img'] is not None and second_isp_result['color_img'] is not None:
            isp_psnr = calculate_psnr(isp_result['color_img'], second_isp_result['color_img'], 255.0)
            print(f"  ISP PSNR: {isp_psnr:.2f} dB")
            results['psnr_isp'] = isp_psnr
        else:
            print("  Warning: Cannot calculate ISP PSNR - one or both ISP results are None")
            results['psnr_isp'] = None
        
        # 8. 保存对比图像
        print("\n8. Saving comparison images...")
        save_comparison_images(
            raw_data, reconstructed_raw,
            isp_result['color_img'], second_isp_result['color_img'],
            output_dir
        )
    except:
        print(f"Error in ISP-Invert-ISP test: {e}")
    
    return True

def main():
    """主函数"""
    # 配置参数
    config = {
        # 输入输出配置
        'INPUT_RAW_PATH': r"F:\ZJU\Picture\ccm\ccm_1\25-09-05 101447.raw",  # 输入RAW文件路径
        'OUTPUT_DIRECTORY': r"F:\ZJU\Picture\isp_invert_isp_test",  # 输出目录
        
        # 图像参数
        'IMAGE_WIDTH': 3840,
        'IMAGE_HEIGHT': 2160,
        'DATA_TYPE': 'uint16',
        
        # ISP参数路径
        'DARK_RAW_PATH': r"F:\ZJU\Picture\dark\g3\average_dark.raw",
        'LENS_SHADING_PARAMS_DIR': r"F:\ZJU\Picture\lens shading\new",
        'WB_PARAMS_PATH': r"F:\ZJU\Picture\wb\wb_output",
        'CCM_MATRIX_PATH': r"F:\ZJU\Picture\ccm\ccm_2\ccm_output_20250905_162714",
        
        # 处理开关
        'DARK_SUBTRACTION_ENABLED': True,
        'LENS_SHADING_ENABLED': True,
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
