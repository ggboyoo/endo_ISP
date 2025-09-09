#!/usr/bin/env python3
"""
Test script for histogram feature in apply_shading_correction.py
测试直方图功能的脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the histogram functions
try:
    from apply_shading_correction import (
        create_histogram_comparison, 
        create_statistics_summary, 
        print_statistics_summary
    )
    print("Successfully imported histogram functions")
except ImportError as e:
    print(f"Error importing functions: {e}")
    exit(1)

def create_test_images():
    """创建测试图像数据"""
    print("Creating test images...")
    
    # 创建模拟的原始图像（带镜头阴影）
    height, width = 1080, 1920
    
    # 创建中心亮、边缘暗的镜头阴影效果
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    
    # 计算到中心的距离
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # 创建镜头阴影（中心为1，边缘为0.3）
    shading_factor = 0.3 + 0.7 * (1 - distance / max_distance)
    
    # 添加噪声和变化
    base_image = np.random.normal(1000, 200, (height, width))
    original_image = (base_image * shading_factor).astype(np.uint16)
    
    # 创建暗电流矫正后的图像
    dark_current = np.random.normal(50, 10, (height, width))
    dark_corrected_image = np.clip(original_image - dark_current, 0, 65535).astype(np.uint16)
    
    # 创建完全矫正后的图像（模拟镜头阴影矫正）
    correction_matrix = 1.0 / shading_factor
    corrected_image = np.clip(dark_corrected_image * correction_matrix, 0, 65535).astype(np.uint16)
    
    print(f"Test images created:")
    print(f"  Original: {original_image.shape}, range: {np.min(original_image)}-{np.max(original_image)}")
    print(f"  Dark corrected: {dark_corrected_image.shape}, range: {np.min(dark_corrected_image)}-{np.max(dark_corrected_image)}")
    print(f"  Fully corrected: {corrected_image.shape}, range: {np.min(corrected_image)}-{np.max(corrected_image)}")
    
    return original_image, dark_corrected_image, corrected_image

def test_histogram_functions():
    """测试直方图功能"""
    print("\n=== Testing Histogram Functions ===")
    
    # 创建测试图像
    original, dark_corrected, corrected = create_test_images()
    
    # 测试统计信息摘要
    print("\n1. Testing statistics summary...")
    stats = create_statistics_summary(original, corrected, dark_corrected)
    print_statistics_summary(stats)
    
    # 测试直方图对比
    print("\n2. Testing histogram comparison...")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    create_histogram_comparison(
        original_image=original,
        corrected_image=corrected,
        dark_corrected_image=dark_corrected,
        output_dir=str(output_dir),
        filename="test_histogram"
    )
    
    print(f"\nTest completed! Check {output_dir} for output files.")

def test_without_dark_correction():
    """测试不包含暗电流矫正的情况"""
    print("\n=== Testing Without Dark Correction ===")
    
    # 创建测试图像
    original, _, corrected = create_test_images()
    
    # 测试统计信息摘要（不包含暗电流矫正）
    print("\n1. Testing statistics summary (no dark correction)...")
    stats = create_statistics_summary(original, corrected)
    print_statistics_summary(stats)
    
    # 测试直方图对比（不包含暗电流矫正）
    print("\n2. Testing histogram comparison (no dark correction)...")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    create_histogram_comparison(
        original_image=original,
        corrected_image=corrected,
        output_dir=str(output_dir),
        filename="test_histogram_no_dark"
    )
    
    print(f"\nTest completed! Check {output_dir} for output files.")

if __name__ == "__main__":
    print("=== Histogram Feature Test ===")
    
    try:
        # 测试完整功能
        test_histogram_functions()
        
        # 测试不包含暗电流矫正的情况
        test_without_dark_correction()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()




