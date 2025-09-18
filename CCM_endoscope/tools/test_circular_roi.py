#!/usr/bin/env python3
"""
测试圆形ROI检测和PSNR计算功能
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_circular_image(width: int = 1920, height: int = 1080, 
                              center_x: int = None, center_y: int = None, 
                              radius: int = None) -> np.ndarray:
    """创建测试用的圆形图像"""
    if center_x is None:
        center_x = width // 2
    if center_y is None:
        center_y = height // 2
    if radius is None:
        radius = min(width, height) // 4
    
    # 创建黑色背景
    image = np.zeros((height, width), dtype=np.uint8)
    
    # 创建圆形区域
    y, x = np.ogrid[:height, :width]
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    
    # 在圆形区域内添加渐变
    for i in range(radius):
        inner_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= (radius - i) ** 2
        intensity = int(255 * (1 - i / radius))
        image[inner_mask] = intensity
    
    return image

def test_circular_roi_detection():
    """测试圆形ROI检测功能"""
    print("=== 测试圆形ROI检测功能 ===")
    
    try:
        from example_invert_ISP import detect_circular_roi, calculate_psnr_circular_roi
        print("✓ 成功导入ROI检测函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 测试1: 1K分辨率
    print("\n--- 测试1: 1K分辨率 (1920x1080) ---")
    test_img_1k = create_test_circular_image(1920, 1080, 960, 540, 300)
    
    center_x, center_y, radius = detect_circular_roi(test_img_1k)
    print(f"检测结果: center=({center_x}, {center_y}), radius={radius}")
    print(f"预期结果: center=(960, 540), radius=300")
    
    # 检查检测精度
    center_error = np.sqrt((center_x - 960)**2 + (center_y - 540)**2)
    radius_error = abs(radius - 300)
    
    if center_error < 50 and radius_error < 50:
        print("✓ 1K分辨率检测通过")
    else:
        print(f"✗ 1K分辨率检测失败: center_error={center_error:.1f}, radius_error={radius_error}")
        return False
    
    # 测试2: 4K分辨率
    print("\n--- 测试2: 4K分辨率 (3840x2160) ---")
    test_img_4k = create_test_circular_image(3840, 2160, 1920, 1080, 600)
    
    center_x, center_y, radius = detect_circular_roi(test_img_4k)
    print(f"检测结果: center=({center_x}, {center_y}), radius={radius}")
    print(f"预期结果: center=(1920, 1080), radius=600")
    
    # 检查检测精度
    center_error = np.sqrt((center_x - 1920)**2 + (center_y - 1080)**2)
    radius_error = abs(radius - 600)
    
    if center_error < 100 and radius_error < 100:
        print("✓ 4K分辨率检测通过")
    else:
        print(f"✗ 4K分辨率检测失败: center_error={center_error:.1f}, radius_error={radius_error}")
        return False
    
    return True

def test_circular_roi_psnr():
    """测试圆形ROI PSNR计算功能"""
    print("\n=== 测试圆形ROI PSNR计算功能 ===")
    
    try:
        from example_invert_ISP import calculate_psnr_circular_roi
        print("✓ 成功导入PSNR计算函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 创建测试图像
    print("\n--- 测试PSNR计算 ---")
    img1 = create_test_circular_image(1920, 1080, 960, 540, 300)
    img2 = img1.copy()  # 完全相同的图像
    
    # 添加一些噪声到img2
    noise = np.random.normal(0, 10, img2.shape).astype(np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 计算PSNR
    psnr, roi_info = calculate_psnr_circular_roi(img1, img2, 255.0)
    print(f"PSNR结果: {psnr:.2f} dB")
    print(f"ROI信息: center=({roi_info[0]}, {roi_info[1]}), radius={roi_info[2]}")
    
    if psnr > 20:  # 期望PSNR应该比较高
        print("✓ PSNR计算通过")
        return True
    else:
        print("✗ PSNR计算失败")
        return False

def test_visualization():
    """测试ROI可视化功能"""
    print("\n=== 测试ROI可视化功能 ===")
    
    try:
        from example_invert_ISP import detect_circular_roi
        print("✓ 成功导入可视化函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 创建测试图像
    test_img = create_test_circular_image(1920, 1080, 960, 540, 300)
    
    # 检测ROI
    center_x, center_y, radius = detect_circular_roi(test_img)
    
    # 创建可视化图像
    vis_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    cv2.circle(vis_img, (center_x, center_y), radius, (0, 255, 0), 2)
    cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)  # 中心点
    
    # 保存可视化结果
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "roi_detection_test.png"), vis_img)
    print(f"✓ 可视化结果已保存到: {output_dir / 'roi_detection_test.png'}")
    
    return True

if __name__ == "__main__":
    print("圆形ROI检测和PSNR计算功能测试")
    print("=" * 50)
    
    success1 = test_circular_roi_detection()
    success2 = test_circular_roi_psnr()
    success3 = test_visualization()
    
    print("\n" + "=" * 50)
    if success1 and success2 and success3:
        print("✓ 所有测试通过！")
        sys.exit(0)
    else:
        print("✗ 部分测试失败！")
        sys.exit(1)
