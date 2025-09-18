#!/usr/bin/env python3
"""
Example demonstrating process_single_image with different raw_file types
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_with_data_array():
    """Example using RAW data array directly"""
    print("=== Example 1: Using RAW data array directly ===")
    
    try:
        from ISP import process_single_image
        
        # Create a dummy RAW data array (simulating 12-bit data in 16-bit container)
        dummy_raw_data = np.random.randint(0, 4096, size=(2160, 3840), dtype=np.uint16)
        print(f"Created dummy RAW data: {dummy_raw_data.shape}, range: {np.min(dummy_raw_data)}-{np.max(dummy_raw_data)}")
        
        # Process using the data array directly
        result = process_single_image(
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
        
        if result['processing_success']:
            print("✓ Processing successful with data array input")
            if 'color_img_16bit' in result:
                print(f"  Output shape: {result['color_img_16bit'].shape}")
        else:
            print(f"✗ Processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Exception occurred: {e}")

def example_with_file_path():
    """Example using file path"""
    print("\n=== Example 2: Using file path ===")
    
    try:
        from ISP import process_single_image
        
        # This would work with an actual RAW file
        raw_file_path = "path/to/your/image.raw"
        
        print(f"Would process file: {raw_file_path}")
        print("(This example shows the interface - replace with actual file path)")
        
        # Example call (commented out since file doesn't exist)
        # result = process_single_image(
        #     raw_file=raw_file_path,  # Pass file path
        #     width=3840,
        #     height=2160,
        #     data_type='uint16',
        #     dark_subtraction_enabled=False,
        #     lens_shading_enabled=False,
        #     white_balance_enabled=False,
        #     ccm_enabled=False,
        #     gamma_correction_enabled=False,
        #     demosaic_output=True
        # )
        
    except Exception as e:
        print(f"✗ Exception occurred: {e}")

def example_with_mixed_parameters():
    """Example with mixed parameter types"""
    print("\n=== Example 3: Mixed parameter types ===")
    
    try:
        from ISP import process_single_image
        
        # Create dummy data
        dummy_raw_data = np.random.randint(0, 4096, size=(2160, 3840), dtype=np.uint16)
        dummy_dark_data = np.random.randint(0, 100, size=(2160, 3840), dtype=np.uint16)
        
        # Example with mixed parameter types
        result = process_single_image(
            raw_file=dummy_raw_data,  # Data array
            dark_data=dummy_dark_data,  # Data array
            lens_shading_params="path/to/lens_params",  # Path string
            width=3840,
            height=2160,
            data_type='uint16',
            wb_params="path/to/wb_params",  # Path string
            ccm_matrix=np.array([[1.5, -0.3, 0.1], [-0.1, 1.2, -0.2], [0.05, -0.1, 1.1]]),  # Data array
            dark_subtraction_enabled=True,
            lens_shading_enabled=False,  # Disabled since we're passing a path
            white_balance_enabled=False,  # Disabled since we're passing a path
            ccm_enabled=True,
            gamma_correction_enabled=False,
            demosaic_output=True
        )
        
        print("✓ Mixed parameter types example completed")
        print("  - raw_file: data array")
        print("  - dark_data: data array") 
        print("  - lens_shading_params: path string")
        print("  - wb_params: path string")
        print("  - ccm_matrix: data array")
        
    except Exception as e:
        print(f"✗ Exception occurred: {e}")

if __name__ == "__main__":
    print("Examples of process_single_image with different raw_file types")
    print("=" * 60)
    
    example_with_data_array()
    example_with_file_path()
    example_with_mixed_parameters()
    
    print("\n" + "=" * 60)
    print("Key points:")
    print("1. raw_file can be a string path or numpy array")
    print("2. Other parameters can also be paths or values")
    print("3. If path is provided, parameter is loaded automatically")
    print("4. If value is provided, it's used directly")
    print("5. This provides maximum flexibility for different use cases")
