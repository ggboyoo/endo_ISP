#!/usr/bin/env python3
"""
invert_ISP.py 使用示例
展示如何使用增强后的逆ISP处理功能
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from invert_ISP import invert_isp_pipeline, DEFAULT_CONFIG

def example_usage():
    """展示invert_ISP.py的使用方法"""
    print("=" * 60)
    print("invert_ISP.py 使用示例")
    print("=" * 60)
    
    print("\n1. 使用默认配置运行（无需命令行参数）:")
    print("   python invert_ISP.py")
    print("   - 使用DEFAULT_CONFIG中设置的默认路径")
    print("   - 自动显示RAW图为灰度图")
    print("   - 保存中间结果和最终结果")
    
    print("\n2. 指定输入输出路径:")
    print("   python invert_ISP.py -i input.jpg -o output.raw")
    
    print("\n3. 禁用某些处理步骤:")
    print("   python invert_ISP.py --no-gamma --no-lens-shading")
    
    print("\n4. 禁用显示功能:")
    print("   python invert_ISP.py --no-display --no-save-grayscale")
    
    print("\n5. 完整参数示例:")
    print("   python invert_ISP.py \\")
    print("     --input test_image.jpg \\")
    print("     --output result.raw \\")
    print("     --bayer rggb \\")
    print("     --gamma 2.2 \\")
    print("     --ccm ccm_matrix.json \\")
    print("     --wb wb_params.json \\")
    print("     --lens-shading lens_params_dir \\")
    print("     --dark dark_reference.raw")
    print("   # 注意：图像尺寸会自动从输入图像检测，无需指定 --width 和 --height")
    
    print("\n" + "=" * 60)
    print("程序化使用示例:")
    print("=" * 60)
    
    # 创建自定义配置
    custom_config = DEFAULT_CONFIG.copy()
    custom_config.update({
        'INPUT_IMAGE_PATH': 'test_input.jpg',
        'OUTPUT_RAW_PATH': 'test_output.raw',
        'DISPLAY_RAW_GRAYSCALE': True,
        'SAVE_RAW_GRAYSCALE': True,
        'CREATE_COMPARISON_PLOT': True,
        'SAVE_INTERMEDIATE': True
    })
    
    print(f"自定义配置:")
    print(f"  输入图像: {custom_config['INPUT_IMAGE_PATH']}")
    print(f"  输出RAW: {custom_config['OUTPUT_RAW_PATH']}")
    print(f"  显示灰度图: {custom_config['DISPLAY_RAW_GRAYSCALE']}")
    print(f"  保存灰度图: {custom_config['SAVE_RAW_GRAYSCALE']}")
    print(f"  创建对比图: {custom_config['CREATE_COMPARISON_PLOT']}")
    
    print(f"\nCCM矩阵:")
    for i, row in enumerate(custom_config['CCM_MATRIX']):
        print(f"  [{i}]: {row}")
    
    print(f"\n白平衡参数:")
    for key, value in custom_config['WB_PARAMS']['white_balance_gains'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("功能特性:")
    print("=" * 60)
    print("✅ 完整的逆ISP处理流程")
    print("✅ 支持所有ISP.py中的参数")
    print("✅ 8bit灰度图显示RAW数据")
    print("✅ 可选的输入输出路径")
    print("✅ 自动检测图像尺寸（无需手动指定宽高）")
    print("✅ 中间结果保存")
    print("✅ 对比图生成")
    print("✅ 命令行和程序化使用")
    print("✅ 灵活的处理开关")
    print("=" * 60)

if __name__ == "__main__":
    example_usage()
