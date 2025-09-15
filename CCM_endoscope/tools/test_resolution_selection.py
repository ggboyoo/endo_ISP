#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试invert_ISP.py的分辨率选择功能
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import cv2

def create_test_images():
    """创建不同分辨率的测试图像"""
    print("创建测试图像...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="test_resolution_")
    print(f"临时目录: {temp_dir}")
    
    # 创建不同分辨率的测试图像
    resolutions = {
        "test_1k.jpg": (1920, 1080),
        "test_4k.jpg": (3840, 2160),
        "test_720p.jpg": (1280, 720)
    }
    
    for filename, (width, height) in resolutions.items():
        # 创建彩色测试图像
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 添加一些图案
        cv2.rectangle(img, (0, 0), (width-1, height-1), (255, 255, 255), 2)
        cv2.putText(img, f"{width}x{height}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # 保存图像
        img_path = os.path.join(temp_dir, filename)
        cv2.imwrite(img_path, img)
        print(f"创建测试图像: {filename} ({width}x{height})")
    
    return temp_dir, resolutions

def test_resolution_selection():
    """测试分辨率选择功能"""
    print("=" * 60)
    print("测试 invert_ISP.py 的分辨率选择功能")
    print("=" * 60)
    
    try:
        # 创建测试图像
        temp_dir, resolutions = create_test_images()
        
        # 测试不同的分辨率选择
        test_cases = [
            {
                "name": "1K分辨率预设",
                "image": "test_1k.jpg",
                "resolution": "1k",
                "expected_width": 1920,
                "expected_height": 1080
            },
            {
                "name": "4K分辨率预设", 
                "image": "test_4k.jpg",
                "resolution": "4k",
                "expected_width": 3840,
                "expected_height": 2160
            },
            {
                "name": "自动检测分辨率",
                "image": "test_720p.jpg",
                "resolution": "auto",
                "expected_width": 1280,
                "expected_height": 720
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}: {test_case['name']}")
            print(f"图像: {test_case['image']}")
            print(f"分辨率设置: {test_case['resolution']}")
            print(f"预期尺寸: {test_case['expected_width']}x{test_case['expected_height']}")
            print("-" * 40)
            
            # 构建命令
            image_path = os.path.join(temp_dir, test_case['image'])
            output_path = os.path.join(temp_dir, f"output_{i}.raw")
            
            cmd = [
                "python", "invert_ISP.py",
                "--input", image_path,
                "--output", output_path,
                "--resolution", test_case['resolution'],
                "--no-display",
                "--no-save-grayscale", 
                "--no-comparison",
                "--no-save-intermediate"
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
            
            # 检查输出文件
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                expected_size = test_case['expected_width'] * test_case['expected_height'] * 2  # uint16 = 2 bytes
                print(f"输出文件大小: {file_size} bytes")
                print(f"预期文件大小: {expected_size} bytes")
                if file_size == expected_size:
                    print("✅ 文件大小匹配")
                else:
                    print("❌ 文件大小不匹配")
            else:
                print("❌ 输出文件未生成")
            
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
    success = test_resolution_selection()
    sys.exit(0 if success else 1)
