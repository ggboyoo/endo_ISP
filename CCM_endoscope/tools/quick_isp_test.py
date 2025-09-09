#!/usr/bin/env python3
"""
快速ISP-逆ISP测试脚本
简化版本，用于快速验证功能
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """计算PSNR"""
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

def quick_test():
    """快速测试"""
    print("=" * 60)
    print("Quick ISP-Invert-ISP Test")
    print("=" * 60)
    
    # 测试参数
    raw_file = r"F:\ZJU\Picture\ccm\25-09-01 160527.raw"
    output_dir = Path(r"F:\ZJU\Picture\quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 导入模块
        from ISP import process_single_image, load_dark_reference, load_correction_parameters, load_white_balance_parameters
        from invert_ISP import invert_isp_pipeline, DEFAULT_CONFIG as INVERT_CONFIG
        from raw_reader import read_raw_image
        
        print("✅ Modules imported successfully")
        
        # 配置参数
        config = {
            'IMAGE_WIDTH': 3840,
            'IMAGE_HEIGHT': 2160,
            'DATA_TYPE': 'uint16',
            'DARK_RAW_PATH': r"F:\ZJU\Picture\dark\g3\average_dark.raw",
            'LENS_SHADING_PARAMS_DIR': r"F:\ZJU\Picture\lens shading\new",
            'WB_PARAMS_PATH': r"F:\ZJU\Picture\wb\wb_output",
            'CCM_MATRIX_PATH': r"F:\ZJU\Picture\ccm\ccm_2\ccm_output_20250905_162714",
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
        
        # 加载参数
        print("\n1. Loading parameters...")
        dark_data = load_dark_reference(config['DARK_RAW_PATH'], config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'], config['DATA_TYPE'])
        lens_shading_params = load_correction_parameters(config['LENS_SHADING_PARAMS_DIR'])
        wb_params = load_white_balance_parameters(config['WB_PARAMS_PATH'])
        
        print("✅ Parameters loaded successfully")
        
        # 第一次ISP处理
        print("\n2. First ISP processing...")
        isp_result = process_single_image(
            raw_file=raw_file,
            dark_data=dark_data,
            lens_shading_params=lens_shading_params,
            width=config['IMAGE_WIDTH'],
            height=config['IMAGE_HEIGHT'],
            data_type=config['DATA_TYPE'],
            wb_params=wb_params,
            ccm_matrix=config['ccm_matrix']
        )
        
        if not isp_result['processing_success']:
            raise ValueError(f"First ISP failed: {isp_result.get('error')}")
        
        print("✅ First ISP completed")
        
        # 保存ISP结果
        if isp_result['color_img'] is not None:
            cv2.imwrite(str(output_dir / "first_isp.jpg"), isp_result['color_img'])
        
        # 逆ISP处理
        print("\n3. Inverse ISP processing...")
        invert_config = INVERT_CONFIG.copy()
        invert_config.update({
            'INPUT_IMAGE_PATH': str(output_dir / "first_isp.jpg"),
            'OUTPUT_RAW_PATH': str(output_dir / "reconstructed.raw"),
            'IMAGE_WIDTH': config['IMAGE_WIDTH'],
            'IMAGE_HEIGHT': config['IMAGE_HEIGHT'],
            'DISPLAY_RAW_GRAYSCALE': False,
            'SAVE_RAW_GRAYSCALE': False,
            'CREATE_COMPARISON_PLOT': False
        })
        
        invert_result = invert_isp_pipeline(invert_config['INPUT_IMAGE_PATH'], invert_config)
        
        if not invert_result['processing_success']:
            raise ValueError(f"Inverse ISP failed: {invert_result.get('error')}")
        
        print("✅ Inverse ISP completed")
        
        # 第二次ISP处理
        print("\n4. Second ISP processing...")
        second_isp_result = process_single_image(
            raw_file=invert_config['OUTPUT_RAW_PATH'],
            dark_data=dark_data,
            lens_shading_params=lens_shading_params,
            width=config['IMAGE_WIDTH'],
            height=config['IMAGE_HEIGHT'],
            data_type=config['DATA_TYPE'],
            wb_params=wb_params,
            ccm_matrix=config['ccm_matrix']
        )
        
        if not second_isp_result['processing_success']:
            raise ValueError(f"Second ISP failed: {second_isp_result.get('error')}")
        
        print("✅ Second ISP completed")
        
        # 保存第二次ISP结果
        if second_isp_result['color_img'] is not None:
            cv2.imwrite(str(output_dir / "second_isp.jpg"), second_isp_result['color_img'])
        
        # 计算PSNR
        print("\n5. Calculating PSNR...")
        if isp_result['color_img'] is not None and second_isp_result['color_img'] is not None:
            psnr = calculate_psnr(isp_result['color_img'], second_isp_result['color_img'])
            print(f"✅ ISP PSNR: {psnr:.2f} dB")
        else:
            print("❌ Cannot calculate PSNR - missing images")
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Output directory: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
