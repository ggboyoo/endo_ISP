#!/usr/bin/env python3
"""
测试简单去马赛克函数
"""

import numpy as np
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from demosaic_easy import demosaic_easy

def test_easy_demosaic():
    """测试简单去马赛克函数"""
    print("Testing Easy Demosaicing")
    print("=" * 40)
    
    # 创建测试数据
    height, width = 8, 8
    raw_data = np.zeros((height, width), dtype=np.uint16)
    
    # 填充测试数据 (RGGB模式)
    # R G R G
    # G B G B
    # R G R G
    # G B G B
    raw_data[0, 0] = 1000  # R
    raw_data[0, 1] = 2000  # G
    raw_data[0, 2] = 1000  # R
    raw_data[0, 3] = 2000  # G
    raw_data[1, 0] = 2000  # G
    raw_data[1, 1] = 3000  # B
    raw_data[1, 2] = 2000  # G
    raw_data[1, 3] = 3000  # B
    raw_data[2, 0] = 1000  # R
    raw_data[2, 1] = 2000  # G
    raw_data[2, 2] = 1000  # R
    raw_data[2, 3] = 2000  # G
    raw_data[3, 0] = 2000  # G
    raw_data[3, 1] = 3000  # B
    raw_data[3, 2] = 2000  # G
    raw_data[3, 3] = 3000  # B
    
    print(f"Test RAW data:")
    print(raw_data)
    
    # 测试去马赛克
    result = demosaic_easy(raw_data, 'rggb')
    
    if result is not None:
        print(f"\nDemosaiced result:")
        print(f"Shape: {result.shape}")
        print(f"Range: {np.min(result)}-{np.max(result)}")
        
        # 显示第一个2x2块的结果
        print(f"\nFirst 2x2 block RGB values:")
        for y in range(2):
            for x in range(2):
                rgb = result[y, x]
                print(f"  Pixel ({y},{x}): R={rgb[2]}, G={rgb[1]}, B={rgb[0]}")
        
        # 验证所有像素的RGB值是否相同
        first_rgb = result[0, 0]
        all_same = np.all(result == first_rgb)
        print(f"\nAll pixels have same RGB values: {all_same}")
        
        if all_same:
            print(f"RGB values: R={first_rgb[2]}, G={first_rgb[1]}, B={first_rgb[0]}")
            print(f"Expected: R=1000, G=2000, B=3000")
    else:
        print("Demosaicing failed!")

if __name__ == "__main__":
    test_easy_demosaic()
