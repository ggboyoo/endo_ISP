#!/usr/bin/env python3
"""
Test script to verify Bayer pattern fix
Compares different demosaicing methods to find the correct one
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from raw_reader import (
    read_raw_image, 
    normalize_to_8bit, 
    demosaic_image_corrected,
    demosaic_image_corrected_fixed,
    demosaic_image_with_channel_test,
    save_image
)


def test_bayer_patterns(raw_file: str, width: int = 3840, height: int = 2160):
    """
    Test different Bayer pattern interpretations on a single RAW file
    """
    print(f"Testing Bayer patterns on: {raw_file}")
    
    # Read and normalize RAW image
    raw_img = read_raw_image(raw_file, width, height, 'uint16')
    img_8bit = normalize_to_8bit(raw_img, 4095)
    
    # Test different demosaicing methods
    methods = {
        'Original': demosaic_image_corrected,
        'Fixed_RGGB': demosaic_image_corrected_fixed,
        'Channel_Test': demosaic_image_with_channel_test
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n--- Testing {method_name} ---")
        try:
            if method_name == 'Channel_Test':
                result = method_func(img_8bit, 'rggb')
            else:
                result = method_func(img_8bit, 'rggb')
            
            if result is not None:
                results[method_name] = result
                print(f"✓ {method_name}: Success")
                
                # Save result for comparison
                output_file = f"test_{method_name.lower()}.png"
                save_image(result, output_file)
                print(f"  Saved: {output_file}")
            else:
                print(f"✗ {method_name}: Failed - No result")
                
        except Exception as e:
            print(f"✗ {method_name}: Failed - {e}")
    
    return results


def compare_results(results: dict):
    """
    Display comparison of different demosaicing results
    """
    if not results:
        print("No results to compare!")
        return
    
    # Create a figure with subplots
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for i, (method_name, img) in enumerate(results.items()):
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"{method_name}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main test function"""
    # Test file path
    test_file = r"F:\ZJU\Picture\25-08-25 142238.raw"
    
    if not Path(test_file).exists():
        print(f"Test file not found: {test_file}")
        print("Please update the test_file path in the script.")
        return
    
    print("=== Bayer Pattern Fix Test ===")
    print(f"Test file: {test_file}")
    
    # Test different Bayer patterns
    results = test_bayer_patterns(test_file)
    
    # Compare results
    if results:
        print(f"\n=== Comparison Results ===")
        print(f"Generated {len(results)} different demosaicing results")
        print("Check the saved PNG files to see which one looks correct.")
        print("The 'Fixed_RGGB' method should have corrected R/G channel swap.")
        
        # Display comparison
        compare_results(results)
    else:
        print("No successful results to compare!")


if __name__ == "__main__":
    main()


