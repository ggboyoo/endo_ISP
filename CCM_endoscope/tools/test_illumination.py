#!/usr/bin/env python3
"""
测试照明调整功能
"""

import numpy as np
import cv2
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ccm_endoscope import create_illumination_correction_grid

def test_illumination_correction():
    """测试照明调整功能"""
    print("=== 测试照明调整功能 ===")
    
    # 创建测试配置
    config = {
        'ILLUMINATION_GRID_SIZE': 32,
        'ILLUMINATION_REFERENCE_METHOD': 'center',
        'ILLUMINATION_ADJUSTMENT_STRENGTH': 1.0,
        'ILLUMINATION_SMOOTHING': True
    }
    
    # 创建测试图像 - 模拟照明不均匀
    height, width = 2160, 3840
    print(f"创建测试图像: {width}x{height}")
    
    # 创建渐变照明图（中心亮，边缘暗）
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # 创建渐变：中心255，边缘100
    illumination = 100 + (255 - 100) * (1 - distance / max_distance)
    illumination = np.clip(illumination, 0, 255).astype(np.uint8)
    
    print(f"测试图像范围: {np.min(illumination)} - {np.max(illumination)}")
    
    # 测试2D图像
    print("\n--- 测试2D灰度图像 ---")
    try:
        grid_2d = create_illumination_correction_grid(illumination, config)
        print(f"2D图像网格形状: {grid_2d.shape}")
        print(f"2D图像网格范围: {np.min(grid_2d):.3f} - {np.max(grid_2d):.3f}")
    except Exception as e:
        print(f"2D图像测试失败: {e}")
    
    # 测试3D图像
    print("\n--- 测试3D彩色图像 ---")
    try:
        # 将灰度图转换为3D彩色图
        illumination_3d = cv2.cvtColor(illumination, cv2.COLOR_GRAY2BGR)
        print(f"3D图像形状: {illumination_3d.shape}")
        
        grid_3d = create_illumination_correction_grid(illumination_3d, config)
        print(f"3D图像网格形状: {grid_3d.shape}")
        print(f"3D图像网格范围: {np.min(grid_3d):.3f} - {np.max(grid_3d):.3f}")
    except Exception as e:
        print(f"3D图像测试失败: {e}")
    
    # 测试小图像
    print("\n--- 测试小图像 ---")
    try:
        small_image = illumination[::10, ::10]  # 缩小10倍
        print(f"小图像形状: {small_image.shape}")
        
        grid_small = create_illumination_correction_grid(small_image, config)
        print(f"小图像网格形状: {grid_small.shape}")
        print(f"小图像网格范围: {np.min(grid_small):.3f} - {np.max(grid_small):.3f}")
    except Exception as e:
        print(f"小图像测试失败: {e}")

if __name__ == "__main__":
    test_illumination_correction()
