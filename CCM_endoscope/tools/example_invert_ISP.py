#!/usr/bin/env python3
"""
Example usage of invert_ISP.py
逆ISP处理脚本使用示例
"""

import os
import sys
from pathlib import Path
from invert_ISP import invert_isp_pipeline, DEFAULT_CONFIG

def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("Example 1: Basic Inverse ISP Processing")
    print("=" * 60)
    
    # 假设有一个处理过的图像
    input_image = "path/to/your/processed_image.jpg"
    output_raw = "path/to/output/reconstructed.raw"
    
    # 基本配置
    config = DEFAULT_CONFIG.copy()
    config['INPUT_IMAGE_PATH'] = input_image
    config['OUTPUT_RAW_PATH'] = output_raw
    config['IMAGE_WIDTH'] = 3840
    config['IMAGE_HEIGHT'] = 2160
    config['BAYER_PATTERN'] = 'rggb'
    config['SAVE_INTERMEDIATE'] = True
    config['VERBOSE'] = True
    
    # 执行逆ISP处理
    result = invert_isp_pipeline(input_image, config)
    
    if result['processing_success']:
        print("✅ Basic inverse ISP processing completed successfully!")
        print(f"   Output RAW file: {result['output_path']}")
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

def example_with_parameters():
    """带参数的使用示例"""
    print("=" * 60)
    print("Example 2: Inverse ISP with CCM and White Balance")
    print("=" * 60)
    
    # 文件路径
    input_image = "path/to/your/processed_image.jpg"
    output_raw = "path/to/output/reconstructed.raw"
    ccm_file = "path/to/ccm_matrix.json"
    wb_file = "path/to/wb_parameters.json"
    
    # 完整配置
    config = DEFAULT_CONFIG.copy()
    config['INPUT_IMAGE_PATH'] = input_image
    config['OUTPUT_RAW_PATH'] = output_raw
    config['IMAGE_WIDTH'] = 3840
    config['IMAGE_HEIGHT'] = 2160
    config['BAYER_PATTERN'] = 'rggb'
    config['CCM_MATRIX_PATH'] = ccm_file
    config['WB_PARAMS_PATH'] = wb_file
    config['GAMMA_VALUE'] = 2.2
    config['SAVE_INTERMEDIATE'] = True
    config['VERBOSE'] = True
    
    # 执行逆ISP处理
    result = invert_isp_pipeline(input_image, config)
    
    if result['processing_success']:
        print("✅ Full inverse ISP processing completed successfully!")
        print(f"   Output RAW file: {result['output_path']}")
        if 'report_path' in result:
            print(f"   Processing report: {result['report_path']}")
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

def example_batch_processing():
    """批量处理示例"""
    print("=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    # 输入和输出目录
    input_dir = Path("path/to/input/images")
    output_dir = Path("path/to/output/raw_files")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} image files to process")
    
    # 处理每个图像
    success_count = 0
    for i, image_file in enumerate(image_files, 1):
        print(f"\nProcessing {i}/{len(image_files)}: {image_file.name}")
        
        # 配置参数
        config = DEFAULT_CONFIG.copy()
        config['INPUT_IMAGE_PATH'] = str(image_file)
        config['OUTPUT_RAW_PATH'] = str(output_dir / f"{image_file.stem}.raw")
        config['IMAGE_WIDTH'] = 3840
        config['IMAGE_HEIGHT'] = 2160
        config['BAYER_PATTERN'] = 'rggb'
        config['SAVE_INTERMEDIATE'] = False
        config['VERBOSE'] = False
        
        # 执行逆ISP处理
        result = invert_isp_pipeline(str(image_file), config)
        
        if result['processing_success']:
            print(f"  ✅ Success: {result['output_path']}")
            success_count += 1
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\nBatch processing completed: {success_count}/{len(image_files)} files processed successfully")

def example_different_sizes():
    """不同尺寸处理示例"""
    print("=" * 60)
    print("Example 4: Processing Different Image Sizes")
    print("=" * 60)
    
    # 不同尺寸的配置
    size_configs = [
        {"width": 1920, "height": 1080, "name": "HD"},
        {"width": 3840, "height": 2160, "name": "4K"},
        {"width": 640, "height": 480, "name": "VGA"},
        {"width": 1280, "height": 720, "name": "HD_720"}
    ]
    
    input_image = "path/to/your/processed_image.jpg"
    
    for size_config in size_configs:
        print(f"\nProcessing {size_config['name']} ({size_config['width']}x{size_config['height']})")
        
        # 配置参数
        config = DEFAULT_CONFIG.copy()
        config['INPUT_IMAGE_PATH'] = input_image
        config['OUTPUT_RAW_PATH'] = f"path/to/output/reconstructed_{size_config['name'].lower()}.raw"
        config['IMAGE_WIDTH'] = size_config['width']
        config['IMAGE_HEIGHT'] = size_config['height']
        config['BAYER_PATTERN'] = 'rggb'
        config['SAVE_INTERMEDIATE'] = False
        config['VERBOSE'] = False
        
        # 执行逆ISP处理
        result = invert_isp_pipeline(input_image, config)
        
        if result['processing_success']:
            print(f"  ✅ {size_config['name']} processing completed")
        else:
            print(f"  ❌ {size_config['name']} processing failed: {result.get('error', 'Unknown error')}")

def example_command_line_usage():
    """命令行使用示例"""
    print("=" * 60)
    print("Example 5: Command Line Usage")
    print("=" * 60)
    
    print("Basic usage:")
    print("python invert_ISP.py --input image.jpg --output image.raw")
    
    print("\nWith all parameters:")
    print("python invert_ISP.py \\")
    print("    --input processed_image.jpg \\")
    print("    --output reconstructed.raw \\")
    print("    --width 3840 \\")
    print("    --height 2160 \\")
    print("    --bayer rggb \\")
    print("    --ccm ccm_matrix.json \\")
    print("    --wb wb_parameters.json \\")
    print("    --gamma 2.2")
    
    print("\nBatch processing with shell:")
    print("for img in *.jpg; do")
    print("    python invert_ISP.py --input \"$img\" --output \"${img%.jpg}.raw\"")
    print("done")

def main():
    """主函数"""
    print("Inverse ISP Processing Examples")
    print("=" * 60)
    
    print("\nThis script demonstrates various usage patterns for invert_ISP.py")
    print("Note: These are examples with placeholder paths.")
    print("Replace the paths with your actual file paths before running.")
    
    # 运行示例
    example_basic_usage()
    example_with_parameters()
    example_batch_processing()
    example_different_sizes()
    example_command_line_usage()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
