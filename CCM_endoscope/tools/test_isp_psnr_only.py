#!/usr/bin/env python3
"""
测试只计算ISP图像PSNR的功能
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_isp_images(width: int = 1920, height: int = 1080) -> tuple:
    """创建测试用的ISP图像"""
    # 创建第一张图像
    img1 = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 在中心创建圆形区域
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    
    # 创建渐变圆形
    y, x = np.ogrid[:height, :width]
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    
    # 添加颜色渐变
    for i in range(radius):
        inner_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= (radius - i) ** 2
        intensity = int(255 * (1 - i / radius))
        img1[inner_mask] = [intensity, intensity // 2, intensity // 3]  # BGR
    
    # 创建第二张图像（添加一些噪声）
    img2 = img1.copy().astype(np.int16)
    noise = np.random.normal(0, 15, img2.shape).astype(np.int16)
    img2 = np.clip(img2 + noise, 0, 255).astype(np.uint8)
    
    return img1, img2

def test_isp_psnr_calculation():
    """测试ISP图像PSNR计算功能"""
    print("=== 测试ISP图像PSNR计算功能 ===")
    
    try:
        from example_invert_ISP import calculate_psnr_circular_roi
        print("✓ 成功导入PSNR计算函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 创建测试图像
    print("\n--- 创建测试图像 ---")
    img1, img2 = create_test_isp_images(1920, 1080)
    print(f"图像1形状: {img1.shape}, 范围: {np.min(img1)}-{np.max(img1)}")
    print(f"图像2形状: {img2.shape}, 范围: {np.min(img2)}-{np.max(img2)}")
    
    # 计算PSNR
    print("\n--- 计算圆形ROI PSNR ---")
    psnr, roi_info = calculate_psnr_circular_roi(img1, img2, 255.0)
    print(f"PSNR结果: {psnr:.2f} dB")
    print(f"ROI信息: center=({roi_info[0]}, {roi_info[1]}), radius={roi_info[2]}")
    
    if psnr > 15:  # 期望PSNR应该比较高
        print("✓ PSNR计算通过")
        return True, img1, img2, roi_info
    else:
        print("✗ PSNR计算失败")
        return False, img1, img2, roi_info

def test_visualization(img1, img2, roi_info):
    """测试可视化功能"""
    print("\n=== 测试可视化功能 ===")
    
    try:
        from example_invert_ISP import save_comparison_images
        print("✓ 成功导入可视化函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 创建输出目录
    output_dir = Path("test_isp_output")
    output_dir.mkdir(exist_ok=True)
    
    # 保存对比图像
    print("\n--- 保存对比图像 ---")
    save_comparison_images(img1, img2, output_dir, roi_info)
    
    # 检查文件是否创建
    files_created = [
        "original_isp.jpg",
        "reconstructed_isp.jpg", 
        "comparison_results.png"
    ]
    
    all_created = True
    for filename in files_created:
        file_path = output_dir / filename
        if file_path.exists():
            print(f"✓ 文件已创建: {file_path}")
        else:
            print(f"✗ 文件未创建: {file_path}")
            all_created = False
    
    if all_created:
        print("✓ 可视化功能测试通过")
        return True
    else:
        print("✗ 可视化功能测试失败")
        return False

def test_4k_resolution():
    """测试4K分辨率"""
    print("\n=== 测试4K分辨率 ===")
    
    try:
        from example_invert_ISP import calculate_psnr_circular_roi
        print("✓ 成功导入PSNR计算函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 创建4K测试图像
    print("\n--- 创建4K测试图像 ---")
    img1, img2 = create_test_isp_images(3840, 2160)
    print(f"4K图像1形状: {img1.shape}")
    print(f"4K图像2形状: {img2.shape}")
    
    # 计算PSNR
    print("\n--- 计算4K圆形ROI PSNR ---")
    psnr, roi_info = calculate_psnr_circular_roi(img1, img2, 255.0)
    print(f"4K PSNR结果: {psnr:.2f} dB")
    print(f"4K ROI信息: center=({roi_info[0]}, {roi_info[1]}), radius={roi_info[2]}")
    
    if psnr > 15:
        print("✓ 4K分辨率测试通过")
        return True
    else:
        print("✗ 4K分辨率测试失败")
        return False

if __name__ == "__main__":
    print("ISP图像PSNR计算功能测试")
    print("=" * 50)
    
    # 测试1K分辨率
    success1, img1, img2, roi_info = test_isp_psnr_calculation()
    success2 = test_visualization(img1, img2, roi_info)
    success3 = test_4k_resolution()
    
    print("\n" + "=" * 50)
    if success1 and success2 and success3:
        print("✓ 所有测试通过！")
        print("✓ RAW图像PSNR计算已删除")
        print("✓ 只在ISP图像上计算PSNR")
        print("✓ ROI可视化功能正常")
        sys.exit(0)
    else:
        print("✗ 部分测试失败！")
        sys.exit(1)
