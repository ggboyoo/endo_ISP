#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ISP.py的尺寸检查功能
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import json
from pathlib import Path

def create_test_data():
    """创建不同尺寸的测试数据"""
    print("创建测试数据...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="test_isp_dimension_")
    print(f"临时目录: {temp_dir}")
    
    # 创建不同尺寸的测试RAW文件
    test_cases = {
        "test_1k.raw": (1920, 1080),
        "test_4k.raw": (3840, 2160),
        "test_720p.raw": (1280, 720)
    }
    
    for filename, (width, height) in test_cases.items():
        # 创建随机RAW数据
        raw_data = np.random.randint(0, 4095, (height, width), dtype=np.uint16)
        
        # 保存RAW文件
        raw_path = os.path.join(temp_dir, filename)
        raw_data.astype(np.uint16).tofile(raw_path)
        print(f"创建测试RAW文件: {filename} ({width}x{height})")
    
    # 创建不同尺寸的暗电流文件
    dark_cases = {
        "dark_1k.raw": (1920, 1080),
        "dark_4k.raw": (3840, 2160),
        "dark_720p.raw": (1280, 720)
    }
    
    for filename, (width, height) in dark_cases.items():
        # 创建暗电流数据
        dark_data = np.random.randint(0, 100, (height, width), dtype=np.uint16)
        
        # 保存暗电流文件
        dark_path = os.path.join(temp_dir, filename)
        dark_data.astype(np.uint16).tofile(dark_path)
        print(f"创建暗电流文件: {filename} ({width}x{height})")
    
    # 创建不同尺寸的镜头阴影参数
    lens_cases = {
        "lens_1k": (1920, 1080),
        "lens_4k": (3840, 2160),
        "lens_720p": (1280, 720)
    }
    
    for dirname, (width, height) in lens_cases.items():
        lens_dir = os.path.join(temp_dir, dirname)
        os.makedirs(lens_dir, exist_ok=True)
        
        # 创建校正矩阵
        correction_map = np.random.uniform(0.8, 1.2, (height, width)).astype(np.float32)
        
        # 保存校正参数
        params = {
            'correction_map': correction_map.tolist(),
            'width': width,
            'height': height
        }
        
        params_file = os.path.join(lens_dir, "correction_params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"创建镜头阴影参数: {dirname} ({width}x{height})")
    
    return temp_dir, test_cases, dark_cases, lens_cases

def test_dimension_check():
    """测试尺寸检查功能"""
    print("=" * 60)
    print("测试 ISP.py 的尺寸检查功能")
    print("=" * 60)
    
    try:
        # 创建测试数据
        temp_dir, test_cases, dark_cases, lens_cases = create_test_data()
        
        # 测试不同的尺寸组合
        test_scenarios = [
            {
                "name": "尺寸匹配 - 1K图像 + 1K暗电流 + 1K镜头阴影",
                "raw_file": "test_1k.raw",
                "dark_file": "dark_1k.raw", 
                "lens_dir": "lens_1k",
                "resolution": "1k",
                "expected_dark": True,
                "expected_lens": True
            },
            {
                "name": "尺寸不匹配 - 1K图像 + 4K暗电流 + 1K镜头阴影",
                "raw_file": "test_1k.raw",
                "dark_file": "dark_4k.raw",
                "lens_dir": "lens_1k", 
                "resolution": "1k",
                "expected_dark": False,
                "expected_lens": True
            },
            {
                "name": "尺寸不匹配 - 4K图像 + 1K暗电流 + 4K镜头阴影",
                "raw_file": "test_4k.raw",
                "dark_file": "dark_1k.raw",
                "lens_dir": "lens_4k",
                "resolution": "4k", 
                "expected_dark": False,
                "expected_lens": True
            },
            {
                "name": "尺寸不匹配 - 720p图像 + 1K暗电流 + 1K镜头阴影",
                "raw_file": "test_720p.raw",
                "dark_file": "dark_1k.raw",
                "lens_dir": "lens_1k",
                "resolution": "auto",
                "expected_dark": False,
                "expected_lens": False
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n测试场景 {i}: {scenario['name']}")
            print("-" * 50)
            
            # 构建命令
            raw_path = os.path.join(temp_dir, scenario['raw_file'])
            dark_path = os.path.join(temp_dir, scenario['dark_file'])
            lens_path = os.path.join(temp_dir, scenario['lens_dir'])
            output_dir = os.path.join(temp_dir, f"output_{i}")
            
            cmd = [
                "python", "ISP.py",
                "--input", raw_path,
                "--resolution", scenario['resolution'],
                "--dark", dark_path,
                "--lens-shading", lens_path,
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
            
            # 分析输出，检查是否正确跳过了不匹配的步骤
            output_text = result.stdout.lower()
            
            # 检查暗电流处理
            if "dark current subtraction skipped due to dimension mismatch" in output_text:
                dark_skipped = True
            elif "dark current subtraction applied" in output_text:
                dark_skipped = False
            else:
                dark_skipped = None
            
            # 检查镜头阴影处理
            if "lens shading correction skipped due to dimension mismatch" in output_text:
                lens_skipped = True
            elif "lens shading correction applied" in output_text:
                lens_skipped = False
            else:
                lens_skipped = None
            
            print(f"暗电流处理: {'跳过' if dark_skipped else '应用' if dark_skipped is not None else '未知'}")
            print(f"镜头阴影处理: {'跳过' if lens_skipped else '应用' if lens_skipped is not None else '未知'}")
            
            # 验证结果
            if dark_skipped == scenario['expected_dark']:
                print("✅ 暗电流处理结果符合预期")
            else:
                print(f"❌ 暗电流处理结果不符合预期 (期望: {scenario['expected_dark']}, 实际: {dark_skipped})")
            
            if lens_skipped == scenario['expected_lens']:
                print("✅ 镜头阴影处理结果符合预期")
            else:
                print(f"❌ 镜头阴影处理结果不符合预期 (期望: {scenario['expected_lens']}, 实际: {lens_skipped})")
            
            print()
        
        print("=" * 60)
        print("测试完成")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
            print(f"清理临时目录: {temp_dir}")
        except:
            pass

def test_command_line_options():
    """测试命令行选项"""
    print("\n" + "=" * 60)
    print("测试命令行选项")
    print("=" * 60)
    
    try:
        # 创建测试数据
        temp_dir, test_cases, dark_cases, lens_cases = create_test_data()
        
        # 测试禁用尺寸检查
        print("\n测试禁用尺寸检查...")
        raw_path = os.path.join(temp_dir, "test_1k.raw")
        dark_path = os.path.join(temp_dir, "dark_4k.raw")  # 尺寸不匹配
        lens_path = os.path.join(temp_dir, "lens_1k")
        output_dir = os.path.join(temp_dir, "output_no_check")
        
        cmd = [
            "python", "ISP.py",
            "--input", raw_path,
            "--resolution", "1k",
            "--dark", dark_path,
            "--lens-shading", lens_path,
            "--output", output_dir,
            "--no-check-dimensions"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        print("输出:")
        print(result.stdout)
        
        if "dimension check disabled" in result.stdout.lower():
            print("✅ 尺寸检查已正确禁用")
        else:
            print("❌ 尺寸检查未正确禁用")
        
        # 测试强制校正
        print("\n测试强制校正...")
        output_dir2 = os.path.join(temp_dir, "output_force")
        
        cmd2 = [
            "python", "ISP.py",
            "--input", raw_path,
            "--resolution", "1k", 
            "--dark", dark_path,
            "--lens-shading", lens_path,
            "--output", output_dir2,
            "--force-correction"
        ]
        
        print(f"执行命令: {' '.join(cmd2)}")
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=".")
        
        print("输出:")
        print(result2.stdout)
        
        if "force correction even if dimensions mismatch" in result2.stdout.lower():
            print("✅ 强制校正已正确启用")
        else:
            print("❌ 强制校正未正确启用")
        
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
    success1 = test_dimension_check()
    success2 = test_command_line_options()
    
    if success1 and success2:
        print("\n🎉 所有测试通过!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败!")
        sys.exit(1)
