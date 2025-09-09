#!/usr/bin/env python3
"""
Test script to verify ISP image saving functionality
"""

import numpy as np
import cv2
from pathlib import Path

def test_image_save():
    """Test basic image saving functionality"""
    print("=== Testing Image Save Functionality ===")
    
    # Create a test output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create test images
    print("Creating test images...")
    
    # 8-bit test image
    test_8bit = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    test_8bit[:, :, 0] = 255  # Red channel
    test_8bit[:, :, 1] = 128  # Green channel  
    test_8bit[:, :, 2] = 64   # Blue channel
    
    # 16-bit test image
    test_16bit = np.random.randint(0, 4096, (100, 100, 3), dtype=np.uint16)
    test_16bit[:, :, 0] = 4095  # Red channel
    test_16bit[:, :, 1] = 2048  # Green channel
    test_16bit[:, :, 2] = 1024  # Blue channel
    
    # Test saving 8-bit image
    print("Testing 8-bit image save...")
    try:
        img_8bit_path = output_dir / "test_8bit.png"
        success = cv2.imwrite(str(img_8bit_path), test_8bit)
        if success:
            print(f"  ✓ 8-bit image saved: {img_8bit_path}")
        else:
            print(f"  ✗ Failed to save 8-bit image")
    except Exception as e:
        print(f"  ✗ Error saving 8-bit image: {e}")
    
    # Test saving 16-bit image
    print("Testing 16-bit image save...")
    try:
        img_16bit_path = output_dir / "test_16bit.png"
        success = cv2.imwrite(str(img_16bit_path), test_16bit)
        if success:
            print(f"  ✓ 16-bit image saved: {img_16bit_path}")
        else:
            print(f"  ✗ Failed to save 16-bit image")
    except Exception as e:
        print(f"  ✗ Error saving 16-bit image: {e}")
    
    # Test saving RAW data
    print("Testing RAW data save...")
    try:
        raw_path = output_dir / "test_raw.raw"
        with open(raw_path, 'wb') as f:
            test_16bit.tofile(f)
        print(f"  ✓ RAW data saved: {raw_path}")
    except Exception as e:
        print(f"  ✗ Error saving RAW data: {e}")
    
    # Check if files exist
    print("\nChecking saved files...")
    files_to_check = [
        output_dir / "test_8bit.png",
        output_dir / "test_16bit.png", 
        output_dir / "test_raw.raw"
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✓ {file_path.name}: {size} bytes")
        else:
            print(f"  ✗ {file_path.name}: Not found")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_image_save()



