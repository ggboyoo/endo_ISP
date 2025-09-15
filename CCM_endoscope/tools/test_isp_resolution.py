#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ISP.py的分辨率选择功能
"""

import os
import sys
import tempfile
import shutil
import numpy as np

def create_test_raw_files():
    """创建不同分辨率的测试RAW文件"""
    print("创建测试RAW文件...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="test_isp_resolution_")
    print(f"临时目录: {temp_dir}")
    
    # 创建不同分辨率的测试RAW文件
    resolutions = {
        "test_1k.raw": (1920, 1080),
        "test_4k.raw": (3840, 2160),
        "test_720p.raw": (1280, 720)
    }
    
    for filename, (width, height) in resolutions.items():
        # 创建随机RAW数据
        raw_data = np.random.randint(0, 4095, (height, width), dtype=np.uint16)
        
        # 保存RAW文件
        raw_path = os.path.join(temp_dir, filename)
        raw_data.astype(np.uint16).tofile(raw_path)
        print(f"创建测试RAW文件: {filename} ({width}x{height})")
    
    return temp_dir, resolutions

def test_resolution_selection():
    """测试分辨率选择功能"""
    print("=" * 60)
    print("测试 ISP.py 的分辨率选择功能")
    print("=" * 60)
    
    try:
        # 创建测试RAW文件
        temp_dir, resolutions = create_test_raw_files()
        
        # 测试不同的分辨率选择
        test_cases = [
            {
                "name": "1K分辨率预设",
                "raw_file": "test_1k.raw",
                "resolution": "1k",
                "expected_width": 1920,
                "expected_height": 1080
            },
            {
                "name": "4K分辨率预设", 
                "raw_file": "test_4k.raw",
                "resolution": "4k",
                "expected_width": 3840,
                "expected_height": 2160
            },
            {
                "name": "自动检测分辨率",
                "raw_file": "test_720p.raw",
                "resolution": "auto",
                "expected_width": 1280,
                "expected_height": 720
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}: {test_case['name']}")
            print(f"RAW文件: {test_case['raw_file']}")
            print(f"分辨率设置: {test_case['resolution']}")
            print(f"预期尺寸: {test_case['expected_width']}x{test_case['expected_height']}")
            print("-" * 40)
            
            # 构建命令
            raw_path = os.path.join(temp_dir, test_case['raw_file'])
            output_dir = os.path.join(temp_dir, f"output_{i}")
            
            cmd = [
                "python", "ISP.py",
                "--input", raw_path,
                "--resolution", test_case['resolution'],
                "--output", output_dir
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
            
            # 检查输出目录
            if os.path.exists(output_dir):
                output_files = os.listdir(output_dir)
                print(f"输出文件: {output_files}")
                print("✅ 处理完成")
            else:
                print("❌ 输出目录未生成")
            
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
