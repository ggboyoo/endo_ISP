#!/usr/bin/env python3
"""
Batch RAW Image Processing Script
批量RAW图像处理程序
调用ISP.py处理指定路径下的所有RAW图像并保存为JPG格式
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

# Import ISP functions
try:
    from ISP import process_single_image, load_dark_reference, load_correction_parameters
except ImportError:
    print("Error: ISP.py not found in the same directory!")
    print("Please ensure ISP.py is in the same directory as this script.")
    sys.exit(1)

# 配置参数 - 直接在这里修改，无需命令行输入
CONFIG = {
    # 输入输出路径
    'INPUT_DIR': r"F:\ZJU\Picture\ccm\ccm_2",  # 输入目录包含RAW文件
    'OUTPUT_DIR': r"F:\ZJU\Picture\ccm\ccm_2\processed",  # 输出目录用于处理后的JPG文件
    'DARK_FILE': r"F:\ZJU\Picture\dark\g6\average_dark.raw",  # 暗电流参考文件路径
    'LENS_SHADING_DIR': r"F:\ZJU\Picture\lens shading\new",  # 镜头阴影矫正参数目录
    'WB_PARAMS_PATH': r"F:\ZJU\Picture\wb\wb_output",  # 白平衡参数文件或目录
    'CCM_MATRIX_PATH': r"F:\ZJU\Picture\ccm\ccm_output",  # CCM矩阵文件或目录
    
    # 图像参数
    'IMAGE_WIDTH': 3840,
    'IMAGE_HEIGHT': 2160,
    'DATA_TYPE': 'uint16',
    'BAYER_PATTERN': 'rggb',
    
    # 处理选项
    'DARK_SUBTRACTION_ENABLED': True,
    'LENS_SHADING_ENABLED': True,
    'WHITE_BALANCE_ENABLED': True,  # 默认不应用白平衡，因为我们要计算白平衡参数
    'CCM_ENABLED': False,  # 默认不应用CCM矫正
    
    # 输出配置
    'OUTPUT_QUALITY': 95,  # JPG质量 (1-100)
    'SAVE_16BIT': False,  # 是否保存16位图像
    'OVERWRITE_EXISTING': False,  # 是否覆盖已存在的文件
}

def find_raw_files(input_dir: str, extensions: List[str] = None) -> List[Path]:
    """
    在指定目录中查找RAW文件
    
    Args:
        input_dir: 输入目录路径
        extensions: 文件扩展名列表，默认为['.raw', '.RAW']
        
    Returns:
        RAW文件路径列表
    """
    if extensions is None:
        extensions = ['.raw', '.RAW']
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return []
    
    raw_files = []
    for ext in extensions:
        raw_files.extend(input_path.glob(f"*{ext}"))
    
    # 按文件名排序
    raw_files.sort()
    
    print(f"Found {len(raw_files)} RAW files in {input_dir}")
    return raw_files

def process_single_raw_file(raw_file: Path, output_dir: Path, config: Dict, 
                          dark_data: Optional[np.ndarray] = None,
                          lens_shading_params: Optional[Dict] = None) -> Dict:
    """
    处理单个RAW文件
    
    Args:
        raw_file: RAW文件路径
        output_dir: 输出目录
        config: 配置参数
        dark_data: 暗电流数据
        lens_shading_params: 镜头阴影矫正参数
        
    Returns:
        处理结果字典
    """
    print(f"\nProcessing: {raw_file.name}")
    
    try:
        # 加载白平衡参数
        wb_params = None
        if config['WHITE_BALANCE_ENABLED'] and 'WB_PARAMS_PATH' in config:
            from ISP import load_white_balance_parameters
            wb_params = load_white_balance_parameters(config['WB_PARAMS_PATH'])
        
        # 调用ISP处理
        isp_result = process_single_image(
            raw_file=str(raw_file),
            dark_data=dark_data,
            lens_shading_params=lens_shading_params,
            width=config['IMAGE_WIDTH'],
            height=config['IMAGE_HEIGHT'],
            data_type=config['DATA_TYPE'],
            wb_params=wb_params,
            dark_subtraction_enabled=config['DARK_SUBTRACTION_ENABLED'],
            lens_shading_enabled=config['LENS_SHADING_ENABLED'],
            white_balance_enabled=config['WHITE_BALANCE_ENABLED'],
            ccm_enabled=config['CCM_ENABLED'],
            ccm_matrix_path=config.get('CCM_MATRIX_PATH'),
            demosaic_output=True
        )
        
        if not isp_result['processing_success']:
            return {
                'success': False,
                'error': f"ISP processing failed: {isp_result.get('error', 'Unknown error')}",
                'file': str(raw_file)
            }
        
        # 生成输出文件名
        output_filename = raw_file.stem + '_processed.jpg'
        output_path = output_dir / output_filename
        
        # 检查文件是否已存在
        if output_path.exists() and not config['OVERWRITE_EXISTING']:
            print(f"  Skipping (file exists): {output_filename}")
            return {
                'success': True,
                'skipped': True,
                'file': str(raw_file),
                'output': str(output_path)
            }
        
        # 保存8位图像
        if 'color_img' in isp_result and isp_result['color_img'] is not None:
            # 使用OpenCV保存JPG
            import cv2
            success = cv2.imwrite(str(output_path), isp_result['color_img'], 
                                [cv2.IMWRITE_JPEG_QUALITY, config['OUTPUT_QUALITY']])
            
            if success:
                print(f"  Saved: {output_filename}")
                
                # 保存16位图像（如果启用）
                if config['SAVE_16BIT'] and 'color_img_16bit' in isp_result and isp_result['color_img_16bit'] is not None:
                    output_16bit_path = output_dir / (raw_file.stem + '_processed_16bit.png')
                    cv2.imwrite(str(output_16bit_path), isp_result['color_img_16bit'])
                    print(f"  Saved 16-bit: {output_16bit_path.name}")
                
                return {
                    'success': True,
                    'file': str(raw_file),
                    'output': str(output_path),
                    'output_16bit': str(output_16bit_path) if config['SAVE_16BIT'] else None
                }
            else:
                return {
                    'success': False,
                    'error': f"Failed to save image: {output_filename}",
                    'file': str(raw_file)
                }
        else:
            return {
                'success': False,
                'error': "No 8-bit image data available",
                'file': str(raw_file)
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f"Exception during processing: {str(e)}",
            'file': str(raw_file)
        }

def batch_process_raw_files(config: Dict) -> Dict:
    """
    批量处理RAW文件
    
    Args:
        config: 配置参数字典
        
    Returns:
        批量处理结果
    """
    # 从配置中获取路径
    input_dir = config['INPUT_DIR']
    output_dir = config['OUTPUT_DIR']
    dark_file = config.get('DARK_FILE')
    lens_shading_dir = config.get('LENS_SHADING_DIR')
    
    print("=" * 60)
    print("Batch RAW Image Processing")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {config['IMAGE_WIDTH']}x{config['IMAGE_HEIGHT']}")
    print(f"Data type: {config['DATA_TYPE']}")
    print(f"Bayer pattern: {config['BAYER_PATTERN']}")
    print(f"Dark subtraction: {'Enabled' if config['DARK_SUBTRACTION_ENABLED'] else 'Disabled'}")
    print(f"Lens shading: {'Enabled' if config['LENS_SHADING_ENABLED'] else 'Disabled'}")
    print(f"White balance: {'Enabled' if config['WHITE_BALANCE_ENABLED'] else 'Disabled'}")
    print(f"CCM correction: {'Enabled' if config['CCM_ENABLED'] else 'Disabled'}")
    print("=" * 60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找RAW文件
    raw_files = find_raw_files(input_dir)
    if not raw_files:
        return {
            'success': False,
            'error': 'No RAW files found',
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
    
    # 加载暗电流数据
    dark_data = None
    if config['DARK_SUBTRACTION_ENABLED'] and dark_file:
        print(f"\nLoading dark reference: {dark_file}")
        dark_data = load_dark_reference(dark_file, config['IMAGE_WIDTH'], 
                                      config['IMAGE_HEIGHT'], config['DATA_TYPE'])
        if dark_data is None:
            print("Warning: Failed to load dark reference, continuing without dark subtraction")
            config['DARK_SUBTRACTION_ENABLED'] = False
    
    # 加载镜头阴影矫正参数
    lens_shading_params = None
    if config['LENS_SHADING_ENABLED'] and lens_shading_dir:
        print(f"\nLoading lens shading parameters: {lens_shading_dir}")
        lens_shading_params = load_correction_parameters(lens_shading_dir)
        if lens_shading_params is None:
            print("Warning: Failed to load lens shading parameters, continuing without lens shading correction")
            config['LENS_SHADING_ENABLED'] = False
    
    # 处理每个文件
    results = []
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    
    print(f"\nProcessing {len(raw_files)} files...")
    print("-" * 60)
    
    for i, raw_file in enumerate(raw_files, 1):
        print(f"[{i}/{len(raw_files)}] ", end="")
        
        result = process_single_raw_file(
            raw_file=raw_file,
            output_dir=output_path,
            config=config,
            dark_data=dark_data,
            lens_shading_params=lens_shading_params
        )
        
        results.append(result)
        
        if result['success']:
            if result.get('skipped', False):
                skipped_count += 1
            else:
                processed_count += 1
        else:
            failed_count += 1
            print(f"  Error: {result['error']}")
    
    # 保存处理报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': input_dir,
        'output_directory': output_dir,
        'config': config,
        'summary': {
            'total_files': len(raw_files),
            'processed': processed_count,
            'failed': failed_count,
            'skipped': skipped_count
        },
        'results': results
    }
    
    report_path = output_path / 'processing_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Total files: {len(raw_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Processing report saved: {report_path}")
    print("=" * 60)
    
    return {
        'success': failed_count == 0,
        'processed': processed_count,
        'failed': failed_count,
        'skipped': skipped_count,
        'report_path': str(report_path)
    }

def main():
    """主函数"""
    print("=" * 60)
    print("Files ISP - 批量RAW图像处理程序")
    print("=" * 60)
    print("配置参数:")
    print(f"  输入目录: {CONFIG['INPUT_DIR']}")
    print(f"  输出目录: {CONFIG['OUTPUT_DIR']}")
    print(f"  暗电流文件: {CONFIG.get('DARK_FILE', 'None')}")
    print(f"  镜头阴影目录: {CONFIG.get('LENS_SHADING_DIR', 'None')}")
    print(f"  图像尺寸: {CONFIG['IMAGE_WIDTH']}x{CONFIG['IMAGE_HEIGHT']}")
    print(f"  数据类型: {CONFIG['DATA_TYPE']}")
    print(f"  Bayer模式: {CONFIG['BAYER_PATTERN']}")
    print("=" * 60)
    
    # 执行批量处理
    result = batch_process_raw_files(CONFIG)
    
    # 退出状态
    if result['success']:
        print(f"\n✅ 批量处理完成!")
        print(f"  处理成功: {result['processed']} 个文件")
        print(f"  处理失败: {result['failed']} 个文件")
        print(f"  跳过文件: {result['skipped']} 个文件")
        if 'report_path' in result:
            print(f"  处理报告: {result['report_path']}")
        sys.exit(0)
    else:
        print(f"\n❌ 批量处理完成，但有错误!")
        print(f"  处理成功: {result['processed']} 个文件")
        print(f"  处理失败: {result['failed']} 个文件")
        print(f"  跳过文件: {result['skipped']} 个文件")
        sys.exit(1)

if __name__ == "__main__":
    main()
