#!/usr/bin/env python3
"""
调试照明调整功能
"""

import numpy as np
import cv2
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_illumination_correction():
    """调试照明调整功能"""
    print("=== 调试照明调整功能 ===")
    
    # 创建测试配置
    config = {
        'ILLUMINATION_GRID_SIZE': 32,
        'ILLUMINATION_REFERENCE_METHOD': 'center',
        'ILLUMINATION_ADJUSTMENT_STRENGTH': 1.0,
        'ILLUMINATION_SMOOTHING': True
    }
    
    # 测试不同的图像尺寸
    test_sizes = [
        (2160, 3840),  # 正常尺寸
        (1080, 1920),  # 较小尺寸
        (100, 100),    # 很小尺寸
        (50, 50),      # 极小尺寸
    ]
    
    for height, width in test_sizes:
        print(f"\n--- 测试图像尺寸: {width}x{height} ---")
        
        # 创建测试图像
        illumination = np.random.randint(100, 255, (height, width), dtype=np.uint8)
        print(f"图像范围: {np.min(illumination)} - {np.max(illumination)}")
        
        # 计算网格数量
        grid_size = config['ILLUMINATION_GRID_SIZE']
        grid_h = height // grid_size
        grid_w = width // grid_size
        
        print(f"网格大小: {grid_size}")
        print(f"计算网格数量: {grid_w}x{grid_h}")
        
        if grid_h <= 0 or grid_w <= 0:
            print(f"❌ 网格数量无效! 图像太小")
            continue
        
        # 创建网格矩阵
        correction_grid = np.ones((grid_h, grid_w), dtype=np.float64)
        print(f"✅ 网格矩阵形状: {correction_grid.shape}")
        
        # 测试网格计算
        processed_cells = 0
        for i in range(grid_h):
            for j in range(grid_w):
                start_h = i * grid_size
                end_h = min((i + 1) * grid_size, height)
                start_w = j * grid_size
                end_w = min((j + 1) * grid_size, width)
                
                grid_region = illumination[start_h:end_h, start_w:end_w]
                grid_brightness = np.mean(grid_region)
                
                processed_cells += 1
                
                if i == 0 and j == 0:  # 只打印第一个网格的详细信息
                    print(f"第一个网格区域: ({start_h}:{end_h}, {start_w}:{end_w})")
                    print(f"第一个网格亮度: {grid_brightness:.2f}")
        
        print(f"✅ 处理了 {processed_cells} 个网格")

if __name__ == "__main__":
    debug_illumination_correction()
