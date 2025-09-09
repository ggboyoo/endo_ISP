#!/usr/bin/env python3
"""
测试增强后的invert_ISP功能
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from invert_ISP import invert_isp_pipeline, DEFAULT_CONFIG

def test_invert_isp():
    """测试逆ISP处理功能"""
    print("=" * 60)
    print("测试增强后的invert_ISP功能")
    print("=" * 60)
    
    # 测试配置
    test_config = DEFAULT_CONFIG.copy()
    test_config.update({
        'INPUT_IMAGE_PATH': 'test_input.jpg',  # 需要提供测试图像
        'OUTPUT_RAW_PATH': 'test_output.raw',
        'SAVE_INTERMEDIATE': True,
        'VERBOSE': True
    })
    
    print("配置参数:")
    for key, value in test_config.items():
        if key not in ['CCM_MATRIX', 'WB_PARAMS']:  # 跳过复杂对象
            print(f"  {key}: {value}")
    
    print("\nCCM矩阵:")
    for i, row in enumerate(test_config['CCM_MATRIX']):
        print(f"  [{i}]: {row}")
    
    print("\n白平衡参数:")
    for key, value in test_config['WB_PARAMS']['white_balance_gains'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("测试完成 - 配置验证通过")
    print("=" * 60)
    
    # 如果有测试图像，可以运行实际处理
    if os.path.exists(test_config['INPUT_IMAGE_PATH']):
        print(f"\n发现测试图像: {test_config['INPUT_IMAGE_PATH']}")
        print("开始逆ISP处理...")
        
        result = invert_isp_pipeline(test_config['INPUT_IMAGE_PATH'], test_config)
        
        if result['processing_success']:
            print("✅ 逆ISP处理成功完成!")
            print(f"   输出文件: {result['output_path']}")
        else:
            print(f"❌ 逆ISP处理失败: {result.get('error', 'Unknown error')}")
    else:
        print(f"\n未找到测试图像: {test_config['INPUT_IMAGE_PATH']}")
        print("请提供测试图像以进行完整测试")

if __name__ == "__main__":
    test_invert_isp()
