#!/usr/bin/env python3
"""
Test script for CCM functionality in ISP.py
"""

import numpy as np
import json
from pathlib import Path

def create_test_ccm_matrix():
    """Create a test CCM matrix for testing"""
    # Create a simple 3x3 identity matrix with slight adjustments
    ccm_matrix = np.array([
        [1.2, -0.1, 0.0],
        [-0.05, 1.1, -0.05],
        [0.0, -0.1, 1.3]
    ], dtype=np.float64)
    
    return ccm_matrix

def save_test_ccm_json():
    """Save test CCM matrix as JSON file"""
    ccm_matrix = create_test_ccm_matrix()
    
    # Create test CCM data
    ccm_data = {
        "ccm_matrix": ccm_matrix.tolist(),
        "ccm_type": "linear3x3",
        "description": "Test CCM matrix for ISP.py",
        "created_by": "test_ccm_isp.py"
    }
    
    # Save to JSON file
    output_path = Path("test_ccm_matrix.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ccm_data, f, indent=2, ensure_ascii=False)
    
    print(f"Test CCM matrix saved to: {output_path}")
    print(f"CCM matrix:")
    for i, row in enumerate(ccm_matrix):
        print(f"  Row {i}: {row}")
    
    return str(output_path)

def save_test_ccm_csv():
    """Save test CCM matrix as CSV file"""
    ccm_matrix = create_test_ccm_matrix()
    
    # Save to CSV file
    output_path = Path("test_ccm_matrix.csv")
    np.savetxt(output_path, ccm_matrix, delimiter=',', fmt='%.6f')
    
    print(f"Test CCM matrix saved to: {output_path}")
    print(f"CCM matrix:")
    for i, row in enumerate(ccm_matrix):
        print(f"  Row {i}: {row}")
    
    return str(output_path)

def test_ccm_loading():
    """Test CCM matrix loading functionality"""
    print("=== Testing CCM Matrix Loading ===")
    
    # Import ISP functions
    try:
        from ISP import load_ccm_matrix
    except ImportError:
        print("Error: Could not import load_ccm_matrix from ISP.py")
        return False
    
    # Test JSON loading
    print("\n1. Testing JSON loading...")
    json_path = save_test_ccm_json()
    result = load_ccm_matrix(json_path)
    
    if result is not None:
        matrix, matrix_type = result
        print(f"  ✓ JSON loading successful")
        print(f"  Matrix shape: {matrix.shape}")
        print(f"  Matrix type: {matrix_type}")
    else:
        print(f"  ✗ JSON loading failed")
        return False
    
    # Test CSV loading
    print("\n2. Testing CSV loading...")
    csv_path = save_test_ccm_csv()
    result = load_ccm_matrix(csv_path)
    
    if result is not None:
        matrix, matrix_type = result
        print(f"  ✓ CSV loading successful")
        print(f"  Matrix shape: {matrix.shape}")
        print(f"  Matrix type: {matrix_type}")
    else:
        print(f"  ✗ CSV loading failed")
        return False
    
    return True

def test_ccm_application():
    """Test CCM application functionality"""
    print("\n=== Testing CCM Application ===")
    
    # Import ISP functions
    try:
        from ISP import apply_ccm_16bit
    except ImportError:
        print("Error: Could not import apply_ccm_16bit from ISP.py")
        return False
    
    # Create test 16-bit image
    test_image = np.random.randint(0, 4096, (100, 100, 3), dtype=np.uint16)
    print(f"Test image shape: {test_image.shape}, dtype: {test_image.dtype}")
    print(f"Test image range: {np.min(test_image)} - {np.max(test_image)}")
    
    # Create test CCM matrix
    ccm_matrix = create_test_ccm_matrix()
    
    # Apply CCM
    print("\nApplying CCM correction...")
    corrected_image = apply_ccm_16bit(test_image, ccm_matrix, 'linear3x3')
    
    if corrected_image is not None:
        print(f"  ✓ CCM application successful")
        print(f"  Corrected image shape: {corrected_image.shape}, dtype: {corrected_image.dtype}")
        print(f"  Corrected image range: {np.min(corrected_image)} - {np.max(corrected_image)}")
        return True
    else:
        print(f"  ✗ CCM application failed")
        return False

def main():
    """Main test function"""
    print("=== CCM ISP Test Script ===")
    
    # Test CCM loading
    if not test_ccm_loading():
        print("CCM loading tests failed!")
        return 1
    
    # Test CCM application
    if not test_ccm_application():
        print("CCM application tests failed!")
        return 1
    
    print("\n=== All Tests Passed! ===")
    print("CCM functionality in ISP.py is working correctly.")
    print("\nTo use CCM in ISP.py:")
    print("1. Set CCM_ENABLED = True in ISP.py")
    print("2. Set CCM_MATRIX_PATH to your CCM matrix file path")
    print("3. Set CCM_MATRIX_TYPE to 'linear3x3' or 'affine3x4'")
    print("4. Run ISP.py to process your RAW images with CCM correction")
    
    return 0

if __name__ == "__main__":
    exit(main())







