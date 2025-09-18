#!/usr/bin/env python3
"""
Test script to verify process_single_image handles both file paths and data arrays
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_raw_file_types():
    """Test process_single_image with different raw_file types"""
    print("=== Testing process_single_image with different raw_file types ===")
    
    try:
        from ISP import process_single_image
        print("✓ ISP module imported successfully")
    except ImportError as e:
        print(f"✗ ISP import failed: {e}")
        return False
    
    # Create a dummy RAW data array (simulating 12-bit data in 16-bit container)
    dummy_raw_data = np.random.randint(0, 4096, size=(2160, 3840), dtype=np.uint16)
    print(f"Created dummy RAW data: {dummy_raw_data.shape}, range: {np.min(dummy_raw_data)}-{np.max(dummy_raw_data)}")
    
    # Test 1: Using data array directly
    print("\n--- Test 1: Using RAW data array directly ---")
    try:
        result1 = process_single_image(
            raw_file=dummy_raw_data,  # Pass array directly
            width=3840,
            height=2160,
            data_type='uint16',
            dark_subtraction_enabled=False,
            lens_shading_enabled=False,
            white_balance_enabled=False,
            ccm_enabled=False,
            gamma_correction_enabled=False,
            demosaic_output=True
        )
        
        if result1['processing_success']:
            print("✓ Test 1 passed: Array input processed successfully")
            print(f"  Output shape: {result1.get('color_img_16bit', 'N/A').shape if 'color_img_16bit' in result1 else 'N/A'}")
        else:
            print(f"✗ Test 1 failed: {result1.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Test 1 failed with exception: {e}")
        return False
    
    # Test 2: Using file path (if a test file exists)
    print("\n--- Test 2: Using file path ---")
    test_raw_file = "test_image.raw"
    
    # Create a temporary test file
    try:
        dummy_raw_data.tofile(test_raw_file)
        print(f"Created temporary test file: {test_raw_file}")
        
        result2 = process_single_image(
            raw_file=test_raw_file,  # Pass file path
            width=3840,
            height=2160,
            data_type='uint16',
            dark_subtraction_enabled=False,
            lens_shading_enabled=False,
            white_balance_enabled=False,
            ccm_enabled=False,
            gamma_correction_enabled=False,
            demosaic_output=True
        )
        
        if result2['processing_success']:
            print("✓ Test 2 passed: File path input processed successfully")
            print(f"  Output shape: {result2.get('color_img_16bit', 'N/A').shape if 'color_img_16bit' in result2 else 'N/A'}")
        else:
            print(f"✗ Test 2 failed: {result2.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Test 2 failed with exception: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_raw_file):
            os.remove(test_raw_file)
            print(f"Cleaned up test file: {test_raw_file}")
    
    # Test 3: Invalid input type
    print("\n--- Test 3: Invalid input type ---")
    try:
        result3 = process_single_image(
            raw_file=123,  # Invalid type
            width=3840,
            height=2160,
            data_type='uint16'
        )
        
        if not result3['processing_success'] and 'raw_file must be a string path or numpy array' in result3.get('error', ''):
            print("✓ Test 3 passed: Invalid input type correctly rejected")
        else:
            print(f"✗ Test 3 failed: Expected error for invalid input type")
            return False
            
    except Exception as e:
        print(f"✗ Test 3 failed with exception: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    return True

if __name__ == "__main__":
    success = test_raw_file_types()
    sys.exit(0 if success else 1)
