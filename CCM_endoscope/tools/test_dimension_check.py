#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试invert_ISP.py的尺寸检查功能
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="test_dimension_")
    print(f"临时目录: {temp_dir}")
    
    # 创建不同尺寸的测试图像
    test_images = {
        "1920x1080.jpg": (1920, 1080),
        "3840x2160.jpg": (3840, 2160),
        "1280x720.jpg": (1280, 720)
    }
    
    for filename, (width, height) in test_images.items():
        # 创建随机图像
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 保存图像
        img_path = os.path.join(temp_dir, filename)
        import cv2
        cv2.imwrite(img_path, img)
        print(f"创建测试图像: {filename} ({width}x{height})")
    
    # 创建不同尺寸的暗电流数据
    dark_data = {
        "dark_1920x1080.raw": (1920, 1080),
        "dark_3840x2160.raw": (3840, 2160),
        "dark_1280x720.raw": (1280, 720)
    }
    
    for filename, (width, height) in dark_data.items():
        # 创建随机暗电流数据
        dark = np.random.randint(0, 100, (height, width), dtype=np.uint16)
        
        # 保存RAW数据
        dark_path = os.path.join(temp_dir, filename)
        dark.astype(np.uint16).tofile(dark_path)
        print(f"创建暗电流数据: {filename} ({width}x{height})")
    
    return temp_dir, test_images, dark_data

def test_dimension_checking():
    """测试尺寸检查功能"""
    print("=" * 60)
    print("测试 invert_ISP.py 的尺寸检查功能")
    print("=" * 60)
    
    try:
        # 创建测试数据
        temp_dir, test_images, dark_data = create_test_data()
        
        # 测试不同的尺寸组合
        test_cases = [
            {
                "name": "尺寸匹配",
                "image": "1920x1080.jpg",
                "dark": "dark_1920x1080.raw",
                "expected": "应该正常处理"
            },
            {
                "name": "图像尺寸不匹配",
                "image": "1920x1080.jpg", 
                "dark": "dark_3840x2160.raw",
                "expected": "应该跳过暗电流校正"
            },
            {
                "name": "图像尺寸不匹配（相反）",
                "image": "3840x2160.jpg",
                "dark": "dark_1920x1080.raw", 
                "expected": "应该跳过暗电流校正"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}: {test_case['name']}")
            print(f"图像: {test_case['image']}")
            print(f"暗电流: {test_case['dark']}")
            print(f"预期: {test_case['expected']}")
            print("-" * 40)
            
            # 构建命令
            image_path = os.path.join(temp_dir, test_case['image'])
            dark_path = os.path.join(temp_dir, test_case['dark'])
            output_path = os.path.join(temp_dir, f"output_{i}.raw")
            
            cmd = [
                "python", "invert_ISP.py",
                "--input", image_path,
                "--output", output_path,
                "--dark", dark_path,
                "--no-display",
                "--no-save-grayscale",
                "--no-comparison"
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            
            # 执行命令
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            print("输出:")
            print(result.stdout)
            if result.stderr:
                print("错误:")
                print(result.stderr)
            
            print(f"返回码: {result.returncode}")
            print()
        
        print("=" * 60)
        print("测试完成")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
            print(f"清理临时目录: {temp_dir}")
        except:
            pass

if __name__ == "__main__":
    success = test_dimension_checking()
    sys.exit(0 if success else 1)
