#!/usr/bin/env python3
"""
集成简单去马赛克算法的ISP处理脚本
"""

import numpy as np
import cv2
import sys
from pathlib import Path
from typing import Optional, Dict

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from demosaic_easy import demosaic_easy
from raw_reader import read_raw_image

def process_single_image_easy(raw_file: str, 
                             dark_data: np.ndarray, 
                             lens_shading_params: Dict, 
                             width: int, 
                             height: int, 
                             data_type: str, 
                             wb_params: Optional[Dict] = None,
                             ccm_matrix: Optional[np.ndarray] = None,
                             bayer_pattern: str = 'rggb') -> Dict:
    """
    使用简单去马赛克算法处理单个RAW图像
    
    Args:
        raw_file: RAW文件路径
        dark_data: 暗电流数据
        lens_shading_params: 镜头阴影参数
        width: 图像宽度
        height: 图像高度
        data_type: 数据类型
        wb_params: 白平衡参数
        ccm_matrix: CCM矩阵
        bayer_pattern: Bayer模式
        
    Returns:
        处理结果字典
    """
    print("Processing single image with easy demosaicing...")
    
    try:
        # 1. 加载RAW数据
        print("1. Loading RAW data...")
        raw_data = read_raw_image(raw_file, width, height, data_type)
        if raw_data is None:
            raise ValueError(f"Failed to load RAW image: {raw_file}")
        
        print(f"  RAW data loaded: {raw_data.shape}, range: {np.min(raw_data)}-{np.max(raw_data)}")
        
        # 2. 暗电流减法
        print("2. Applying dark current subtraction...")
        dark_corrected = raw_data.astype(np.float32) - dark_data.astype(np.float32)
        dark_corrected = np.clip(dark_corrected, 0, None)
        print(f"  Dark correction applied, range: {np.min(dark_corrected)}-{np.max(dark_corrected)}")
        
        # 3. 镜头阴影矫正
        print("3. Applying lens shading correction...")
        if lens_shading_params and 'correction_matrix' in lens_shading_params:
            correction_matrix = lens_shading_params['correction_matrix']
            lens_corrected = dark_corrected * correction_matrix
            lens_corrected = np.clip(lens_corrected, 0, None)
            print(f"  Lens shading correction applied")
        else:
            lens_corrected = dark_corrected.copy()
            print(f"  Lens shading correction skipped")
        
        # 4. 简单去马赛克
        print("4. Applying easy demosaicing...")
        color_img = demosaic_easy(lens_corrected.astype(np.uint16), bayer_pattern)
        if color_img is None:
            raise ValueError("Easy demosaicing failed")
        
        print(f"  Easy demosaicing completed: {color_img.shape}, range: {np.min(color_img)}-{np.max(color_img)}")
        
        # 5. 白平衡矫正
        if wb_params and 'white_balance_gains' in wb_params:
            print("5. Applying white balance correction...")
            gains = wb_params['white_balance_gains']
            r_gain = gains.get('r_gain', 1.0)
            g_gain = gains.get('g_gain', 1.0)
            b_gain = gains.get('b_gain', 1.0)
            
            print(f"  White balance gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")
            
            wb_corrected = color_img.copy().astype(np.float32)
            wb_corrected[:, :, 2] *= r_gain  # Red channel
            wb_corrected[:, :, 1] *= g_gain  # Green channel
            wb_corrected[:, :, 0] *= b_gain  # Blue channel
            wb_corrected = np.clip(wb_corrected, 0, None)
            
            color_img = wb_corrected.astype(np.uint16)
            print(f"  White balance correction applied")
        else:
            print("5. White balance correction skipped")
        
        # 6. CCM矫正
        if ccm_matrix is not None:
            print("6. Applying CCM correction...")
            print(f"  CCM matrix shape: {ccm_matrix.shape}")
            
            # 将图像转换为float32进行矩阵运算
            img_float = color_img.astype(np.float32)
            
            # 重塑为 (N, 3) 进行矩阵乘法
            original_shape = img_float.shape
            img_reshaped = img_float.reshape(-1, 3)
            
            # 应用CCM矩阵
            ccm_corrected = np.dot(img_reshaped, ccm_matrix.T)
            ccm_corrected = np.clip(ccm_corrected, 0, None)
            
            # 重塑回原始形状
            color_img = ccm_corrected.reshape(original_shape).astype(np.uint16)
            print(f"  CCM correction applied")
        else:
            print("6. CCM correction skipped")
        
        # 7. 伽马变换
        print("7. Applying gamma correction...")
        gamma_value = 2.2
        img_float = color_img.astype(np.float32) / 4095.0  # 归一化到0-1
        img_gamma = np.power(img_float, 1.0 / gamma_value)
        img_gamma = np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8)
        
        print(f"  Gamma correction applied (γ={gamma_value})")
        print(f"  Final image: {img_gamma.shape}, range: {np.min(img_gamma)}-{np.max(img_gamma)}")
        
        return {
            'processing_success': True,
            'color_img': img_gamma,
            'raw_data': raw_data,
            'dark_corrected': dark_corrected,
            'lens_corrected': lens_corrected,
            'demosaiced': color_img,
            'final_image': img_gamma
        }
        
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return {
            'processing_success': False,
            'error': str(e)
        }

def main():
    """主函数"""
    print("ISP Processing with Easy Demosaicing")
    print("=" * 50)
    
    # 配置参数
    config = {
        'raw_file': r"F:\ZJU\Picture\ccm\ccm_1\25-09-05 101447.raw",
        'width': 3840,
        'height': 2160,
        'data_type': 'uint16',
        'bayer_pattern': 'rggb',
        'dark_file': r"F:\ZJU\Picture\dark\g3\average_dark.raw",
        'lens_shading_dir': r"F:\ZJU\Picture\lens shading\new",
        'wb_file': r"F:\ZJU\Picture\wb\wb_output",
        'ccm_matrix': np.array([
            [1.7801320111582375, -0.7844420268663381, 0.004310015708100662],
            [-0.24377094860030846, 2.4432181685707977, -1.1994472199704893],
            [-0.4715762768203783, -0.7105721829898775, 2.182148459810256]
        ]),
        'output_dir': Path(r"F:\ZJU\Picture\easy_demosaic_test")
    }
    
    # 创建输出目录
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载参数
        print("Loading parameters...")
        
        # 加载暗电流数据
        dark_data = read_raw_image(config['dark_file'], config['width'], config['height'], config['data_type'])
        if dark_data is None:
            raise ValueError(f"Failed to load dark data: {config['dark_file']}")
        
        # 加载镜头阴影参数
        lens_shading_params = None
        try:
            from lens_shading import load_correction_parameters
            lens_shading_params = load_correction_parameters(config['lens_shading_dir'])
        except Exception as e:
            print(f"Warning: Could not load lens shading parameters: {e}")
        
        # 加载白平衡参数
        wb_params = None
        try:
            from WB import load_white_balance_parameters
            wb_params = load_white_balance_parameters(config['wb_file'])
        except Exception as e:
            print(f"Warning: Could not load white balance parameters: {e}")
        
        # 处理图像
        result = process_single_image_easy(
            raw_file=config['raw_file'],
            dark_data=dark_data,
            lens_shading_params=lens_shading_params,
            width=config['width'],
            height=config['height'],
            data_type=config['data_type'],
            wb_params=wb_params,
            ccm_matrix=config['ccm_matrix'],
            bayer_pattern=config['bayer_pattern']
        )
        
        if result['processing_success']:
            print("\n✅ Processing completed successfully!")
            
            # 保存结果
            output_file = config['output_dir'] / "easy_demosaic_result.jpg"
            cv2.imwrite(str(output_file), result['color_img'])
            print(f"  Result saved: {output_file}")
            
            # 显示图像信息
            print(f"\nImage information:")
            print(f"  Final size: {result['color_img'].shape}")
            print(f"  Data range: {np.min(result['color_img'])}-{np.max(result['color_img'])}")
            print(f"  Data type: {result['color_img'].dtype}")
            
        else:
            print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
