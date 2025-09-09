#!/usr/bin/env python3
"""
Test script for invert_ISP.py
测试逆ISP处理脚本
"""

import numpy as np
import cv2
import os
import json
from pathlib import Path
from invert_ISP import invert_isp_pipeline, DEFAULT_CONFIG

def create_test_image(width=3840, height=2160):
    """创建测试图像"""
    print("Creating test image...")
    
    # 创建彩色测试图像
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加一些彩色区域
    # 红色区域
    img[height//4:height//2, width//4:width//2] = [255, 0, 0]
    
    # 绿色区域
    img[height//4:height//2, width//2:3*width//4] = [0, 255, 0]
    
    # 蓝色区域
    img[height//2:3*height//4, width//4:width//2] = [0, 0, 255]
    
    # 白色区域
    img[height//2:3*height//4, width//2:3*width//4] = [255, 255, 255]
    
    # 添加渐变
    for i in range(width):
        img[3*height//4:, i] = [i * 255 // width, 128, 255 - i * 255 // width]
    
    return img

def create_test_ccm_matrix():
    """创建测试CCM矩阵"""
    print("Creating test CCM matrix...")
    
    # 简单的3x3线性变换矩阵
    ccm_matrix = np.array([
        [1.2, -0.1, 0.05],
        [-0.05, 1.1, -0.02],
        [0.01, -0.08, 1.15]
    ])
    
    ccm_data = {
        "ccm_matrix": ccm_matrix.tolist(),
        "ccm_type": "linear3x3",
        "description": "Test CCM matrix for inverse ISP"
    }
    
    return ccm_data

def create_test_wb_parameters():
    """创建测试白平衡参数"""
    print("Creating test white balance parameters...")
    
    wb_data = {
        "white_balance_gains": {
            "r_gain": 1.2,
            "g_gain": 1.0,
            "b_gain": 0.9
        },
        "method": "test",
        "description": "Test white balance parameters for inverse ISP"
    }
    
    return wb_data

def test_basic_inverse_isp():
    """测试基本逆ISP功能"""
    print("=" * 60)
    print("Testing Basic Inverse ISP")
    print("=" * 60)
    
    # 创建测试目录
    test_dir = Path("test_inverse_isp")
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试图像
    test_img = create_test_image(1920, 1080)  # 使用较小的尺寸进行测试
    test_img_path = test_dir / "test_image.jpg"
    cv2.imwrite(str(test_img_path), test_img)
    print(f"Test image saved: {test_img_path}")
    
    # 配置参数
    config = DEFAULT_CONFIG.copy()
    config['INPUT_IMAGE_PATH'] = str(test_img_path)
    config['OUTPUT_RAW_PATH'] = str(test_dir / "test_output.raw")
    config['IMAGE_WIDTH'] = 1920
    config['IMAGE_HEIGHT'] = 1080
    config['BAYER_PATTERN'] = 'rggb'
    config['SAVE_INTERMEDIATE'] = True
    config['VERBOSE'] = True
    
    # 执行逆ISP处理
    result = invert_isp_pipeline(str(test_img_path), config)
    
    if result['processing_success']:
        print("✅ Basic inverse ISP test passed!")
        return True
    else:
        print(f"❌ Basic inverse ISP test failed: {result.get('error', 'Unknown error')}")
        return False

def test_with_ccm_and_wb():
    """测试带CCM和白平衡的逆ISP功能"""
    print("=" * 60)
    print("Testing Inverse ISP with CCM and White Balance")
    print("=" * 60)
    
    # 创建测试目录
    test_dir = Path("test_inverse_isp_full")
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试图像
    test_img = create_test_image(1920, 1080)
    test_img_path = test_dir / "test_image.jpg"
    cv2.imwrite(str(test_img_path), test_img)
    print(f"Test image saved: {test_img_path}")
    
    # 创建CCM矩阵文件
    ccm_data = create_test_ccm_matrix()
    ccm_path = test_dir / "test_ccm.json"
    with open(ccm_path, 'w') as f:
        json.dump(ccm_data, f, indent=2)
    print(f"Test CCM matrix saved: {ccm_path}")
    
    # 创建白平衡参数文件
    wb_data = create_test_wb_parameters()
    wb_path = test_dir / "test_wb.json"
    with open(wb_path, 'w') as f:
        json.dump(wb_data, f, indent=2)
    print(f"Test white balance parameters saved: {wb_path}")
    
    # 配置参数
    config = DEFAULT_CONFIG.copy()
    config['INPUT_IMAGE_PATH'] = str(test_img_path)
    config['OUTPUT_RAW_PATH'] = str(test_dir / "test_output_full.raw")
    config['IMAGE_WIDTH'] = 1920
    config['IMAGE_HEIGHT'] = 1080
    config['BAYER_PATTERN'] = 'rggb'
    config['CCM_MATRIX_PATH'] = str(ccm_path)
    config['WB_PARAMS_PATH'] = str(wb_path)
    config['SAVE_INTERMEDIATE'] = True
    config['VERBOSE'] = True
    
    # 执行逆ISP处理
    result = invert_isp_pipeline(str(test_img_path), config)
    
    if result['processing_success']:
        print("✅ Full inverse ISP test passed!")
        return True
    else:
        print(f"❌ Full inverse ISP test failed: {result.get('error', 'Unknown error')}")
        return False

def test_different_bayer_patterns():
    """测试不同Bayer模式"""
    print("=" * 60)
    print("Testing Different Bayer Patterns")
    print("=" * 60)
    
    bayer_patterns = ['rggb', 'bggr', 'grbg', 'gbrg']
    test_results = {}
    
    for pattern in bayer_patterns:
        print(f"\nTesting Bayer pattern: {pattern}")
        
        # 创建测试目录
        test_dir = Path(f"test_bayer_{pattern}")
        test_dir.mkdir(exist_ok=True)
        
        # 创建测试图像
        test_img = create_test_image(640, 480)  # 使用更小的尺寸
        test_img_path = test_dir / "test_image.jpg"
        cv2.imwrite(str(test_img_path), test_img)
        
        # 配置参数
        config = DEFAULT_CONFIG.copy()
        config['INPUT_IMAGE_PATH'] = str(test_img_path)
        config['OUTPUT_RAW_PATH'] = str(test_dir / f"test_output_{pattern}.raw")
        config['IMAGE_WIDTH'] = 640
        config['IMAGE_HEIGHT'] = 480
        config['BAYER_PATTERN'] = pattern
        config['SAVE_INTERMEDIATE'] = False
        config['VERBOSE'] = False
        
        # 执行逆ISP处理
        result = invert_isp_pipeline(str(test_img_path), config)
        test_results[pattern] = result['processing_success']
        
        if result['processing_success']:
            print(f"  ✅ {pattern} pattern test passed!")
        else:
            print(f"  ❌ {pattern} pattern test failed: {result.get('error', 'Unknown error')}")
    
    return test_results

def main():
    """主测试函数"""
    print("=" * 60)
    print("Inverse ISP Test Suite")
    print("=" * 60)
    
    # 运行测试
    test_results = []
    
    # 基本功能测试
    print("\n1. Basic Inverse ISP Test")
    basic_test = test_basic_inverse_isp()
    test_results.append(("Basic Inverse ISP", basic_test))
    
    # 完整功能测试
    print("\n2. Full Inverse ISP Test (with CCM and WB)")
    full_test = test_with_ccm_and_wb()
    test_results.append(("Full Inverse ISP", full_test))
    
    # Bayer模式测试
    print("\n3. Bayer Pattern Tests")
    bayer_tests = test_different_bayer_patterns()
    for pattern, result in bayer_tests.items():
        test_results.append((f"Bayer {pattern}", result))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
