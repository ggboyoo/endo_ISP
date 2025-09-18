#!/usr/bin/env python3
"""
测试ROI区域边界检查功能
"""

import numpy as np
import cv2
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_images():
    """创建测试图像"""
    # 创建不同尺寸的测试图像
    test_images = {}
    
    # 1K图像 (1920x1080)
    img_1k = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    # 添加圆形区域
    center_x, center_y = 960, 540
    radius = 400
    y, x = np.ogrid[:1080, :1920]
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    img_1k[mask] = [255, 255, 255]  # 白色圆形
    test_images['1K'] = img_1k
    
    # 4K图像 (3840x2160)
    img_4k = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    # 添加圆形区域
    center_x, center_y = 1920, 1080
    radius = 800
    y, x = np.ogrid[:2160, :3840]
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    img_4k[mask] = [255, 255, 255]  # 白色圆形
    test_images['4K'] = img_4k
    
    # 小图像 (640x480)
    img_small = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # 添加圆形区域
    center_x, center_y = 320, 240
    radius = 150
    y, x = np.ogrid[:480, :640]
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    img_small[mask] = [255, 255, 255]  # 白色圆形
    test_images['Small'] = img_small
    
    return test_images

def test_roi_detection():
    """测试ROI检测功能"""
    print("=== 测试ROI检测功能 ===")
    
    try:
        from example_invert_ISP import detect_circular_roi, detect_circular_roi_brightness
        print("✓ 成功导入ROI检测函数（基于亮度检测）")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    test_images = create_test_images()
    
    for name, img in test_images.items():
        print(f"\n--- 测试 {name} 图像 ({img.shape[1]}x{img.shape[0]}) ---")
        
        # 测试亮度检测
        try:
            center_x, center_y, radius = detect_circular_roi(img)
            print(f"亮度检测结果: center=({center_x}, {center_y}), radius={radius}")
            
            # 检查边界
            h, w = img.shape[:2]
            if (0 <= center_x < w and 0 <= center_y < h and 
                0 < radius <= min(center_x, center_y, w - center_x, h - center_y)):
                print("✓ 亮度检测边界检查通过")
            else:
                print("✗ 亮度检测边界检查失败")
                return False
                
        except Exception as e:
            print(f"✗ 亮度检测失败: {e}")
            return False
        
        # 测试直接亮度检测
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_norm = gray.astype(np.float32) / 255.0
            center_x, center_y, radius = detect_circular_roi_brightness(gray_norm)
            print(f"直接亮度检测结果: center=({center_x}, {center_y}), radius={radius}")
            
            # 检查边界
            if (0 <= center_x < w and 0 <= center_y < h and 
                0 < radius <= min(center_x, center_y, w - center_x, h - center_y)):
                print("✓ 直接亮度检测边界检查通过")
            else:
                print("✗ 直接亮度检测边界检查失败")
                return False
                
        except Exception as e:
            print(f"✗ 直接亮度检测失败: {e}")
            return False
    
    return True

def test_psnr_calculation():
    """测试PSNR计算功能"""
    print("\n=== 测试PSNR计算功能 ===")
    
    try:
        from example_invert_ISP import calculate_psnr_circular_roi
        print("✓ 成功导入PSNR计算函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    test_images = create_test_images()
    
    for name, img in test_images.items():
        print(f"\n--- 测试 {name} 图像 PSNR计算 ---")
        
        # 创建轻微不同的图像
        img2 = img.copy().astype(np.float32)
        noise = np.random.normal(0, 10, img2.shape).astype(np.float32)
        img2 = np.clip(img2 + noise, 0, 255).astype(np.uint8)
        
        try:
            psnr, roi_info = calculate_psnr_circular_roi(img, img2, 255.0)
            center_x, center_y, radius = roi_info
            print(f"PSNR结果: {psnr:.2f} dB")
            print(f"ROI信息: center=({center_x}, {center_y}), radius={radius}")
            
            # 检查边界
            h, w = img.shape[:2]
            if (0 <= center_x < w and 0 <= center_y < h and 
                0 < radius <= min(center_x, center_y, w - center_x, h - center_y)):
                print("✓ PSNR计算边界检查通过")
            else:
                print("✗ PSNR计算边界检查失败")
                return False
                
        except Exception as e:
            print(f"✗ PSNR计算失败: {e}")
            return False
    
    return True

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    try:
        from example_invert_ISP import detect_circular_roi, calculate_psnr_circular_roi
        print("✓ 成功导入函数（基于亮度检测）")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 测试极小图像
    print("\n--- 测试极小图像 (10x10) ---")
    tiny_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    try:
        center_x, center_y, radius = detect_circular_roi(tiny_img)
        print(f"极小图像检测结果: center=({center_x}, {center_y}), radius={radius}")
        
        h, w = tiny_img.shape[:2]
        if (0 <= center_x < w and 0 <= center_y < h and 
            0 < radius <= min(center_x, center_y, w - center_x, h - center_y)):
            print("✓ 极小图像边界检查通过")
        else:
            print("✗ 极小图像边界检查失败")
            return False
    except Exception as e:
        print(f"✗ 极小图像检测失败: {e}")
        return False
    
    # 测试极长图像
    print("\n--- 测试极长图像 (100x2000) ---")
    long_img = np.random.randint(0, 255, (100, 2000, 3), dtype=np.uint8)
    try:
        center_x, center_y, radius = detect_circular_roi(long_img)
        print(f"极长图像检测结果: center=({center_x}, {center_y}), radius={radius}")
        
        h, w = long_img.shape[:2]
        if (0 <= center_x < w and 0 <= center_y < h and 
            0 < radius <= min(center_x, center_y, w - center_x, h - center_y)):
            print("✓ 极长图像边界检查通过")
        else:
            print("✗ 极长图像边界检查失败")
            return False
    except Exception as e:
        print(f"✗ 极长图像检测失败: {e}")
        return False
    
    return True

def test_visualization():
    """测试可视化功能"""
    print("\n=== 测试可视化功能 ===")
    
    try:
        from example_invert_ISP import save_comparison_images
        print("✓ 成功导入可视化函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 创建测试图像
    test_images = create_test_images()
    img1 = test_images['1K']
    img2 = test_images['1K'].copy()
    
    # 添加一些噪声
    noise = np.random.normal(0, 10, img2.shape).astype(np.float32)
    img2 = np.clip(img2.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 创建输出目录
    output_dir = Path("test_roi_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 测试PSNR计算和可视化
        from example_invert_ISP import calculate_psnr_circular_roi
        psnr, roi_info = calculate_psnr_circular_roi(img1, img2, 255.0)
        
        # 测试保存功能
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
            
    except Exception as e:
        print(f"✗ 可视化测试失败: {e}")
        return False

if __name__ == "__main__":
    print("ROI区域边界检查功能测试")
    print("=" * 50)
    
    # 运行所有测试
    test1 = test_roi_detection()
    test2 = test_psnr_calculation()
    test3 = test_edge_cases()
    test4 = test_visualization()
    
    print("\n" + "=" * 50)
    if test1 and test2 and test3 and test4:
        print("✓ 所有测试通过！")
        print("✓ ROI区域边界检查功能正常")
        print("✓ 不会出现超出图像范围的错误")
        sys.exit(0)
    else:
        print("✗ 部分测试失败！")
        sys.exit(1)
