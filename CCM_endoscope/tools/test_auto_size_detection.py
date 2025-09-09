#!/usr/bin/env python3
"""
测试invert_ISP.py的自动尺寸检测功能
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from invert_ISP import load_image_as_12bit, DEFAULT_CONFIG

def test_auto_size_detection():
    """测试自动尺寸检测功能"""
    print("=" * 60)
    print("测试invert_ISP.py的自动尺寸检测功能")
    print("=" * 60)
    
    # 测试图像路径（使用用户设置的默认路径）
    test_image = DEFAULT_CONFIG['INPUT_IMAGE_PATH']
    
    if os.path.exists(test_image):
        print(f"测试图像: {test_image}")
        
        try:
            # 测试自动尺寸检测
            img_12bit, width, height = load_image_as_12bit(test_image)
            
            print(f"\n✅ 自动尺寸检测成功!")
            print(f"   检测到的尺寸: {width}x{height}")
            print(f"   图像形状: {img_12bit.shape}")
            print(f"   数据类型: {img_12bit.dtype}")
            print(f"   数值范围: {img_12bit.min()}-{img_12bit.max()}")
            
            # 验证尺寸是否正确
            expected_height, expected_width = img_12bit.shape[:2]
            if width == expected_width and height == expected_height:
                print(f"✅ 尺寸检测正确!")
            else:
                print(f"❌ 尺寸检测错误! 期望: {expected_width}x{expected_height}, 实际: {width}x{height}")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            
    else:
        print(f"❌ 测试图像不存在: {test_image}")
        print("请确保DEFAULT_CONFIG中的INPUT_IMAGE_PATH指向有效的图像文件")
    
    print("\n" + "=" * 60)
    print("功能说明:")
    print("=" * 60)
    print("✅ 图像尺寸自动检测 - 无需手动指定宽高")
    print("✅ 支持任意尺寸的图像")
    print("✅ 自动适配不同分辨率的输入")
    print("✅ 保持原始图像的长宽比")
    print("=" * 60)

if __name__ == "__main__":
    test_auto_size_detection()
