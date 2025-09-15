#!/usr/bin/env python3
"""
Simple test to verify basic functionality
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if required modules can be imported"""
    print("=== Testing Module Imports ===")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ opencv imported successfully")
    except ImportError as e:
        print(f"✗ opencv import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    return True

def test_basic_operations():
    """Test basic operations"""
    print("\n=== Testing Basic Operations ===")
    
    try:
        import numpy as np
        import cv2
        
        # Test numpy operations
        test_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        print(f"✓ Created test array: {test_array.shape}, dtype: {test_array.dtype}")
        
        # Test OpenCV operations
        success = cv2.imwrite("test_simple.png", test_array)
        if success:
            print("✓ OpenCV image save successful")
            
            # Check if file exists
            if Path("test_simple.png").exists():
                print("✓ Test image file created")
                # Clean up
                Path("test_simple.png").unlink()
                print("✓ Test file cleaned up")
            else:
                print("✗ Test image file not found")
                return False
        else:
            print("✗ OpenCV image save failed")
            return False
            
    except Exception as e:
        print(f"✗ Basic operations failed: {e}")
        return False
    
    return True

def test_directory_operations():
    """Test directory operations"""
    print("\n=== Testing Directory Operations ===")
    
    try:
        # Test creating directory
        test_dir = Path("test_dir")
        test_dir.mkdir(exist_ok=True)
        print("✓ Directory creation successful")
        
        # Test file operations
        test_file = test_dir / "test.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        print("✓ File creation successful")
        
        # Check file exists
        if test_file.exists():
            print("✓ File exists check successful")
        
        # Clean up
        test_file.unlink()
        test_dir.rmdir()
        print("✓ Cleanup successful")
        
    except Exception as e:
        print(f"✗ Directory operations failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("Simple Functionality Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed - please install required modules")
        print("Try: pip install numpy opencv-python matplotlib")
        return False
    
    # Test basic operations
    if not test_basic_operations():
        print("\n❌ Basic operations test failed")
        return False
    
    # Test directory operations
    if not test_directory_operations():
        print("\n❌ Directory operations test failed")
        return False
    
    print("\n✅ All tests passed!")
    print("Basic functionality is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






