#!/usr/bin/env python3
"""
测试CCM保存功能
"""

import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

def test_save_functionality():
    """测试保存功能"""
    print("=== Testing CCM Save Functionality ===")
    
    # 创建测试输出目录
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"Test image shape: {test_image.shape}")
    
    # 测试保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_path = output_dir / f"test_image_{timestamp}.png"
    
    print(f"Attempting to save test image to: {test_path}")
    success = cv2.imwrite(str(test_path), test_image)
    
    if success:
        print(f"✓ Test image saved successfully: {test_path}")
    else:
        print(f"✗ Failed to save test image: {test_path}")
    
    # 验证文件存在
    if test_path.exists():
        print(f"✓ Test file exists: {test_path}")
        file_size = test_path.stat().st_size
        print(f"  File size: {file_size} bytes")
    else:
        print(f"✗ Test file does not exist: {test_path}")
    
    # 测试读取
    try:
        loaded_image = cv2.imread(str(test_path))
        if loaded_image is not None:
            print(f"✓ Test image loaded successfully: {loaded_image.shape}")
        else:
            print(f"✗ Failed to load test image")
    except Exception as e:
        print(f"✗ Error loading test image: {e}")
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    test_save_functionality()



