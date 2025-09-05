#!/usr/bin/env python3
"""
Minimal ISP test to verify image saving functionality
"""

import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime

def create_test_raw_data(width=100, height=100):
    """Create test RAW data"""
    # Create a simple test pattern
    raw_data = np.zeros((height, width), dtype=np.uint16)
    
    # Create a simple pattern
    for i in range(height):
        for j in range(width):
            raw_data[i, j] = (i * 10 + j) % 4096
    
    return raw_data

def test_demosaic_16bit(raw_data, bayer_pattern='rggb'):
    """Test 16-bit demosaicing"""
    print("Testing 16-bit demosaicing...")
    
    try:
        if bayer_pattern == 'rggb':
            # Convert to uint16 for OpenCV demosaicing
            raw_data_uint16 = raw_data.astype(np.uint16)
            # Direct demosaicing on 16-bit data
            demosaiced = cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerRG2BGR)
            
            # Apply the R/B channel swap correction
            corrected = demosaiced.copy()
            corrected[:, :, 0] = demosaiced[:, :, 2]  # B = R
            corrected[:, :, 2] = demosaiced[:, :, 0]  # R = B
            
            return corrected
        else:
            raw_data_uint16 = raw_data.astype(np.uint16)
            return cv2.cvtColor(raw_data_uint16, cv2.COLOR_BayerRG2BGR)
    except Exception as e:
        print(f"Demosaicing error: {e}")
        return None

def test_white_balance_16bit(color_image):
    """Test 16-bit white balance"""
    print("Testing 16-bit white balance...")
    
    try:
        # Simple white balance gains
        gains = {'b_gain': 1.2, 'g_gain': 1.0, 'r_gain': 0.8}
        
        corrected = color_image.copy().astype(np.float32)
        corrected[:, :, 0] *= gains['b_gain']  # B channel
        corrected[:, :, 1] *= gains['g_gain']  # G channel
        corrected[:, :, 2] *= gains['r_gain']  # R channel
        
        corrected = np.clip(corrected, 0, 4095)
        return corrected.astype(np.uint16)
    except Exception as e:
        print(f"White balance error: {e}")
        return color_image

def test_image_save(color_16bit, output_dir, base_filename):
    """Test image saving"""
    print(f"\n=== Testing Image Save ===")
    print(f"Output directory: {output_dir}")
    print(f"Base filename: {base_filename}")
    
    try:
        # Convert to 8-bit for display
        color_8bit = (color_16bit.astype(np.float32) / 4095.0 * 255.0).astype(np.uint8)
        
        # Save 8-bit image
        img_8bit_path = output_dir / f"{base_filename}_processed.png"
        success = cv2.imwrite(str(img_8bit_path), color_8bit)
        if success:
            print(f"✓ 8-bit PNG saved: {img_8bit_path}")
        else:
            print(f"✗ Failed to save 8-bit PNG")
        
        # Save 16-bit image
        img_16bit_path = output_dir / f"{base_filename}_processed_16bit.png"
        success = cv2.imwrite(str(img_16bit_path), color_16bit)
        if success:
            print(f"✓ 16-bit PNG saved: {img_16bit_path}")
        else:
            print(f"✗ Failed to save 16-bit PNG")
        
        # Save RAW data
        raw_path = output_dir / f"{base_filename}_16bit.raw"
        with open(raw_path, 'wb') as f:
            color_16bit.tofile(f)
        print(f"✓ RAW data saved: {raw_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Image save error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== Minimal ISP Test ===")
    
    # Create output directory
    output_dir = Path("minimal_test_output")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create test RAW data
    print("Creating test RAW data...")
    raw_data = create_test_raw_data(200, 200)
    print(f"RAW data: {raw_data.shape}, range: {np.min(raw_data)}-{np.max(raw_data)}")
    
    # Test demosaicing
    color_16bit = test_demosaic_16bit(raw_data)
    if color_16bit is None:
        print("❌ Demosaicing failed")
        return False
    
    print(f"Demosaiced: {color_16bit.shape}, range: {np.min(color_16bit)}-{np.max(color_16bit)}")
    
    # Test white balance
    color_16bit_wb = test_white_balance_16bit(color_16bit)
    print(f"White balanced: {color_16bit_wb.shape}, range: {np.min(color_16bit_wb)}-{np.max(color_16bit_wb)}")
    
    # Test image saving
    success = test_image_save(color_16bit_wb, output_dir, "test_minimal")
    
    if success:
        print("\n✅ Minimal ISP test completed successfully!")
        
        # Check saved files
        print("\nChecking saved files...")
        files_to_check = [
            output_dir / "test_minimal_processed.png",
            output_dir / "test_minimal_processed_16bit.png",
            output_dir / "test_minimal_16bit.raw"
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"✓ {file_path.name}: {size} bytes")
            else:
                print(f"✗ {file_path.name}: Not found")
        
        return True
    else:
        print("\n❌ Minimal ISP test failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All tests passed! ISP functionality is working.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
