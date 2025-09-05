#!/usr/bin/env python3
"""
Test script for noise calibration program
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_test_gray_chart():
    """Create a test gray chart image for testing"""
    print("Creating test gray chart image...")
    
    # Create a test image with 24 gray patches
    width, height = 3840, 2160
    rows, cols = 4, 6
    
    # Create base image
    image = np.zeros((height, width), dtype=np.uint16)
    
    # Define gray levels (0-4095 for 12-bit data)
    gray_levels = np.linspace(0, 4095, 24, dtype=np.uint16)
    
    # Calculate patch dimensions
    patch_h = height // rows
    patch_w = width // cols
    
    # Fill patches with different gray levels
    for i in range(rows):
        for j in range(cols):
            patch_id = i * cols + j
            if patch_id < len(gray_levels):
                # Calculate patch boundaries
                start_y = i * patch_h
                end_y = (i + 1) * patch_h
                start_x = j * patch_w
                end_x = (j + 1) * patch_w
                
                # Add some noise to make it realistic
                base_value = gray_levels[patch_id]
                noise = np.random.normal(0, base_value * 0.02, (patch_h, patch_w))  # 2% noise
                patch_values = np.clip(base_value + noise, 0, 4095).astype(np.uint16)
                
                image[start_y:end_y, start_x:end_x] = patch_values
                
                print(f"  Patch {patch_id + 1}: Gray level {base_value}, position ({start_x}, {start_y})")
    
    return image

def save_test_raw_file():
    """Save test gray chart as RAW file"""
    print("Saving test gray chart as RAW file...")
    
    # Create test image
    test_image = create_test_gray_chart()
    
    # Save as RAW file
    output_path = Path("test_gray_chart.raw")
    with open(output_path, 'wb') as f:
        test_image.tofile(f)
    
    print(f"Test gray chart saved to: {output_path}")
    print(f"Image shape: {test_image.shape}, dtype: {test_image.dtype}")
    print(f"Image range: {np.min(test_image)} - {np.max(test_image)}")
    
    return str(output_path)

def test_noise_analysis():
    """Test noise analysis functionality"""
    print("\n=== Testing Whole Image Noise Analysis ===")
    
    # Import noise calibration functions
    try:
        from noise_cali import analyze_whole_image_noise
    except ImportError:
        print("Error: Could not import functions from noise_cali.py")
        return False
    
    # Create test image
    test_image = create_test_gray_chart()
    
    # Analyze whole image noise
    print("\n1. Testing whole image noise analysis...")
    image_stats = analyze_whole_image_noise(test_image)
    
    if 'error' not in image_stats:
        print(f"  ✓ Whole image analysis successful")
        print(f"    Mean: {image_stats['mean']:.1f}")
        print(f"    Std: {image_stats['std']:.1f}")
        print(f"    SNR: {image_stats['snr']:.1f}")
        print(f"    CV: {image_stats['cv']:.4f}")
        print(f"    Total pixels: {image_stats['total_pixels']:,}")
    else:
        print(f"  ✗ Whole image analysis failed: {image_stats['error']}")
        return False
    
    return True

def test_histogram_creation():
    """Test histogram creation functionality"""
    print("\n=== Testing Whole Image Histogram Creation ===")
    
    # Import noise calibration functions
    try:
        from noise_cali import create_whole_image_histogram, analyze_whole_image_noise
    except ImportError:
        print("Error: Could not import functions from noise_cali.py")
        return False
    
    # Create test data
    test_image = create_test_gray_chart()
    image_stats = analyze_whole_image_noise(test_image)
    
    if 'error' in image_stats:
        print(f"  ✗ Image analysis failed: {image_stats['error']}")
        return False
    
    # Create output directory
    output_dir = Path("test_noise_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test histogram creation
    print("\nCreating test histograms...")
    try:
        create_whole_image_histogram(test_image, image_stats, output_dir)
        print("  ✓ Whole image histogram creation successful")
        return True
    except Exception as e:
        print(f"  ✗ Histogram creation failed: {e}")
        return False

def test_pixel_statistics():
    """Test pixel statistics functionality"""
    print("\n=== Testing Pixel Statistics Analysis ===")
    
    # Import noise calibration functions
    try:
        from noise_cali import calculate_pixel_statistics, create_variance_mean_plot
    except ImportError:
        print("Error: Could not import functions from noise_cali.py")
        return False
    
    # Create multiple test images
    print("\n1. Creating multiple test images...")
    test_images = []
    for i in range(5):  # Create 5 test images
        test_image = create_test_gray_chart()
        # Add some variation to each image
        noise = np.random.normal(0, 10, test_image.shape)
        test_image = np.clip(test_image.astype(np.float32) + noise, 0, 4095).astype(np.uint16)
        test_images.append(test_image)
        print(f"  Created test image {i+1}")
    
    # Save test images as RAW files
    test_files = []
    for i, test_image in enumerate(test_images):
        test_file = f"test_image_{i+1}.raw"
        with open(test_file, 'wb') as f:
            test_image.tofile(f)
        test_files.append(test_file)
        print(f"  Saved test file: {test_file}")
    
    # Test pixel statistics calculation
    print("\n2. Testing pixel statistics calculation...")
    try:
        pixel_stats = calculate_pixel_statistics(test_files, None, 3840, 2160, 'uint16')
        
        if 'error' not in pixel_stats:
            print(f"  ✓ Pixel statistics calculation successful")
            print(f"    Total pixels: {pixel_stats['total_pixels']:,}")
            print(f"    Valid files: {pixel_stats['valid_files']}")
            print(f"    Overall mean: {pixel_stats['overall_mean']:.2f}")
            print(f"    Overall variance: {pixel_stats['overall_var']:.2f}")
        else:
            print(f"  ✗ Pixel statistics calculation failed: {pixel_stats['error']}")
            return False
    except Exception as e:
        print(f"  ✗ Pixel statistics calculation failed: {e}")
        return False
    
    # Test variance vs mean plot creation
    print("\n3. Testing variance vs mean plot creation...")
    try:
        output_dir = Path("test_pixel_stats_output")
        output_dir.mkdir(exist_ok=True)
        
        create_variance_mean_plot(pixel_stats, output_dir, "test_pixel_stats")
        print("  ✓ Variance vs mean plot creation successful")
    except Exception as e:
        print(f"  ✗ Variance vs mean plot creation failed: {e}")
        return False
    
    # Clean up test files
    print("\n4. Cleaning up test files...")
    for test_file in test_files:
        try:
            Path(test_file).unlink()
            print(f"  Removed: {test_file}")
        except:
            pass
    
    return True

def main():
    """Main test function"""
    print("=== Noise Calibration Test Script ===")
    
    # Test noise analysis
    if not test_noise_analysis():
        print("Noise analysis tests failed!")
        return 1
    
    # Test histogram creation
    if not test_histogram_creation():
        print("Histogram creation tests failed!")
        return 1
    
    # Test pixel statistics
    if not test_pixel_statistics():
        print("Pixel statistics tests failed!")
        return 1
    
    print("\n=== All Tests Passed! ===")
    print("Noise calibration program is working correctly.")
    print("\nTo use noise_cali.py:")
    print("1. Set INPUT_PATH to your RAW folder path (for multiple images) or file path (for single image)")
    print("2. Set IMAGE_WIDTH and IMAGE_HEIGHT to match your image dimensions")
    print("3. Set DARK_SUBTRACTION_ENABLED if you want to apply dark current correction")
    print("4. Run noise_cali.py to analyze noise:")
    print("   - For multiple images: calculates pixel-wise statistics and variance vs mean plots")
    print("   - For single image: generates whole image histogram")
    print("5. The program will generate plots and save analysis data")
    
    return 0

if __name__ == "__main__":
    exit(main())
