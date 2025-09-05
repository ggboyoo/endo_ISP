#!/usr/bin/env python3
"""
RAW Image Reader and Processor
Replicates MATLAB functionality for reading and processing RAW images
"""

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def read_raw_image(filename: str, col: int = 3840, row: int = 2160, 
                   data_type: str = 'uint16', endian: str = 'little') -> np.ndarray:
    """
    Read RAW image file
    
    Args:
        filename: Path to RAW file
        col: Image width (columns)
        row: Image height (rows)
        data_type: Data type ('uint16', 'uint8', etc.)
        endian: Byte order ('little' or 'big')
    
    Returns:
        Image array with shape (row, col)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Read raw data
    with open(filename, 'rb') as f:
        raw_data = f.read()
    
    # Convert to numpy array
    dtype_map = {
        'uint8': np.uint8,
        'uint16': np.uint16,
        'uint32': np.uint32,
        'float32': np.float32,
        'float64': np.float64
    }
    
    if data_type not in dtype_map:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    dtype = dtype_map[data_type]
    data = np.frombuffer(raw_data, dtype=dtype)
    
    # Reshape to image dimensions
    if len(data) != col * row:
        print(f"Warning: Expected {col * row} pixels, got {len(data)}")
        # Try to infer dimensions
        if len(data) % col == 0:
            row = len(data) // col
            print(f"Inferred row count: {row}")
        elif len(data) % row == 0:
            col = len(data) // row
            print(f"Inferred column count: {col}")
        else:
            raise ValueError(f"Cannot reshape {len(data)} pixels to {col}x{row}")
    
    # Reshape and transpose to match MATLAB behavior
    img = data.reshape((row, col))
    
    return img


def swap_pixels_every_other(img: np.ndarray) -> np.ndarray:
    """
    Swap pixels every other column (replicates MATLAB loop)
    
    Args:
        img: Input image array
    
    Returns:
        Image with swapped pixels
    """
    result = img.copy()
    row, col = img.shape
    
    for i in range(row):
        for j in range(0, col, 2):
            if j + 1 < col:
                p1 = result[i, j]
                p2 = result[i, j + 1]
                result[i, j] = p2
                result[i, j + 1] = p1
    
    return result


def normalize_to_8bit(img: np.ndarray, max_val: int = 4095) -> np.ndarray:
    """
    Normalize image to 8-bit range
    
    Args:
        img: Input image array
        max_val: Maximum value in input image
    
    Returns:
        Normalized 8-bit image
    """
    normalized = np.clip(img.astype(np.float32) / max_val * 255, 0, 255)
    return normalized.astype(np.uint8)


def demosaic_image(img: np.ndarray, pattern: str = 'rggb') -> np.ndarray:
    """
    Demosaic RAW image to color image
    
    Args:
        img: Input RAW image
        pattern: Bayer pattern ('rggb', 'bggr', 'grbg', 'gbrg')
    
    Returns:
        Demosaiced color image
    """
    # OpenCV expects BGR order, so we need to handle different patterns
    if pattern == 'rggb':
        # OpenCV default is BGR, so RGGB becomes BGR
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
    elif pattern == 'bggr':
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    elif pattern == 'grbg':
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    elif pattern == 'gbrg':
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    
    return demosaiced


def detect_bayer_pattern(img: np.ndarray) -> str:
    """
    Detect Bayer pattern by analyzing pixel values
    This helps determine the correct demosaicing approach
    """
    # Sample center region to avoid edge effects
    h, w = img.shape
    center_h, center_w = h // 4, w // 4
    center_img = img[center_h:3*center_h, center_w:3*center_w]
    
    # Calculate average values for different pixel positions
    # Assuming 2x2 Bayer pattern
    even_even = center_img[::2, ::2].mean()
    even_odd = center_img[::2, 1::2].mean()
    odd_even = center_img[1::2, ::2].mean()
    odd_odd = center_img[1::2, 1::2].mean()
    
    print(f"Bayer pattern analysis:")
    print(f"  Even-Even (0,0): {even_even:.1f}")
    print(f"  Even-Odd  (0,1): {even_odd:.1f}")
    print(f"  Odd-Even  (1,0): {odd_even:.1f}")
    print(f"  Odd-Odd   (1,1): {odd_odd:.1f}")
    
    # Determine pattern based on relative values
    # Red pixels are usually brightest, blue darkest
    values = [even_even, even_odd, odd_even, odd_odd]
    max_idx = np.argmax(values)
    min_idx = np.argmin(values)
    
    # Common patterns and their 2x2 arrangements:
    # RGGB: R(0,0) G(0,1) G(1,0) B(1,1)
    # BGGR: B(0,0) G(0,1) G(1,0) R(1,1)
    # GRBG: G(0,0) R(0,1) B(1,0) G(1,1)
    # GBRG: G(0,0) B(0,1) R(1,0) G(1,1)
    
    if max_idx == 0 and min_idx == 3:  # R at (0,0), B at (1,1)
        return 'rggb'
    elif max_idx == 3 and min_idx == 0:  # R at (1,1), B at (0,0)
        return 'bggr'
    elif max_idx == 1 and min_idx == 2:  # R at (0,1), B at (1,0)
        return 'grbg'
    elif max_idx == 2 and min_idx == 1:  # R at (1,0), B at (0,1)
        return 'gbrg'
    else:
        print("Warning: Could not determine Bayer pattern, using RGGB as default")
        return 'rggb'


def demosaic_image_corrected(img: np.ndarray, pattern: str = 'auto') -> np.ndarray:
    """
    Demosaic RAW image with corrected color handling to match MATLAB
    """
    if pattern == 'auto':
        pattern = detect_bayer_pattern(img)
        print(f"Detected Bayer pattern: {pattern.upper()}")
    
    # Use OpenCV demosaicing with proper pattern
    # For RGGB pattern, we need to be careful about the interpretation
    if pattern == 'rggb':
        # Try different Bayer patterns to find the correct one
        # The issue might be that OpenCV's interpretation differs from the actual sensor layout
        try:
            # First try the standard RGGB interpretation
            demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
            print("Using cv2.COLOR_BayerRG2BGR")
        except:
            # If that fails, try alternative interpretations
            try:
                demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
                print("Using cv2.COLOR_BayerBG2BGR")
            except:
                demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
                print("Using cv2.COLOR_BayerGR2BGR")
    elif pattern == 'bggr':
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    elif pattern == 'grbg':
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    elif pattern == 'gbrg':
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    
    # Return BGR format directly (no conversion needed)
    return demosaiced


def demosaic_image_corrected_fixed(img: np.ndarray, pattern: str = 'auto') -> np.ndarray:
    """
    Demosaic RAW image with fixed Bayer pattern interpretation
    This version tries to correct the R/G channel swap issue
    """
    if pattern == 'auto':
        pattern = detect_bayer_pattern(img)
        print(f"Detected Bayer pattern: {pattern.upper()}")
    
    # For RGGB pattern, the issue is likely that OpenCV's interpretation
    # doesn't match the actual sensor layout. Let's try different approaches:
    if pattern == 'rggb':
        print("Processing RGGB pattern with channel correction...")
        
        # Method 1: Try standard RGGB interpretation
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
        
        # Method 2: If R/G channels are swapped, we can swap them back
        # This is a common issue where the Bayer pattern interpretation is correct
        # but the color channels are assigned incorrectly
        
        # Check if we need to swap R and G channels
        # We can do this by swapping the B and R channels in the BGR output
        # Since BGR is B(0), G(1), R(2), swapping B and R means swapping channels 0 and 2
        corrected = demosaiced.copy()
        corrected[:, :, 0] = demosaiced[:, :, 2]  # B = R
        corrected[:, :, 2] = demosaiced[:, :, 0]  # R = B
        
        return corrected
        
    elif pattern == 'bggr':
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
        return demosaiced
    elif pattern == 'grbg':
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
        return demosaiced
    elif pattern == 'gbrg':
        demosaiced = cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
        return demosaiced
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")


def demosaic_image_with_channel_test(img: np.ndarray, pattern: str = 'rggb') -> np.ndarray:
    """
    Demosaic RAW image and test different channel arrangements
    This function will try different Bayer pattern interpretations
    """
    print(f"Testing different Bayer pattern interpretations for {pattern.upper()}")
    
    # Try different OpenCV Bayer patterns
    bayer_patterns = {
        'rg2bgr': cv2.COLOR_BayerRG2BGR,
        'bg2bgr': cv2.COLOR_BayerBG2BGR,
        'gr2bgr': cv2.COLOR_BayerGR2BGR,
        'gb2bgr': cv2.COLOR_BayerGB2BGR
    }
    
    results = {}
    
    for name, cv_pattern in bayer_patterns.items():
        try:
            demosaiced = cv2.cvtColor(img, cv_pattern)
            results[name] = demosaiced
            print(f"  ✓ {name}: Success")
        except Exception as e:
            print(f"  ✗ {name}: Failed - {e}")
    
    # For now, return the standard RGGB interpretation
    # You can manually test the different results
    if 'rg2bgr' in results:
        return results['rg2bgr']
    elif 'bg2bgr' in results:
        return results['bg2bgr']
    else:
        # Fallback to any available result
        return list(results.values())[0] if results else None


def save_image(img: np.ndarray, filename: str, quality: int = 95) -> bool:
    """
    Save image to file
    
    Args:
        img: Image array
        filename: Output filename
        quality: JPEG quality (1-100) - not used for PNG
    
    Returns:
        True if successful
    """
    try:
        if filename.lower().endswith('.png'):
            # For PNG, use OpenCV directly (lossless compression)
            cv2.imwrite(filename, img)
            return True
        elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            # For JPEG, use OpenCV with quality setting
            # Image is already in BGR format, no conversion needed
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, buffer = cv2.imencode('.jpg', img, encode_param)
            if success:
                with open(filename, 'wb') as f:
                    f.write(buffer)
                return True
        else:
            # For other formats, use OpenCV directly
            cv2.imwrite(filename, img)
            return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def display_image(img: np.ndarray, title: str = "Image", cmap: Optional[str] = None):
    """
    Display image using matplotlib
    
    Args:
        img: Image array
        title: Figure title
        cmap: Colormap for grayscale images
    """
    plt.figure(figsize=(10, 8))
    
    if len(img.shape) == 3:
        # Color image - convert BGR to RGB for matplotlib display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        # Grayscale image - use grayscale colormap
        if cmap is None:
            cmap = 'gray'  # Default to grayscale for single-channel images
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    """Main function replicating MATLAB workflow"""
    # Configuration (equivalent to MATLAB variables)
    col = 3840
    row = 2160
    filename = r"F:\ZJU\Picture\25-08-25 142238.raw"
    
    print("=== RAW Image Reader ===")
    print(f"Expected dimensions: {col} x {row}")
    print(f"Input file: {filename}")
    
    try:
        # Read RAW image (equivalent to MATLAB fread)
        print("\nReading RAW file...")
        A = read_raw_image(filename, col, row, 'uint16')
        print(f"Image loaded: {A.shape}, dtype: {A.dtype}")
        
        # Calculate statistics (equivalent to MATLAB mean/max)
        avg = np.mean(A)
        maxs = np.max(A)
        print(f"Average value: {avg:.2f}")
        print(f"Maximum value: {maxs}")
        
        # Pixel swapping (equivalent to MATLAB loop)
        print("\nPerforming pixel swap...")
        B = swap_pixels_every_other(A)
        
        # Normalize to 8-bit (equivalent to MATLAB uint8(A/4095*255))
        print("Normalizing to 8-bit...")
        img8_A = normalize_to_8bit(A, 4095)
        img8_B = normalize_to_8bit(B, 4095)
        
        # Save images
        output_dir = Path(filename).parent
        output_A = output_dir / "825-000003.jpg"
        output_B = output_dir / "y825_B.jpg"
        
        print(f"\nSaving images...")
        if save_image(img8_A, str(output_A)):
            print(f"Saved: {output_A}")
        if save_image(img8_B, str(output_B)):
            print(f"Saved: {output_B}")
        
        # Display images
        print("\nDisplaying images...")
        display_image(img8_A, "Image A (Original)", cmap='gray')
        display_image(img8_B, "Image B (Swapped)", cmap='gray')
        
        # Demosaicing (equivalent to MATLAB demosaic)
        print("\nPerforming demosaicing...")
        try:
            # Use corrected demosaicing with automatic pattern detection
            color_image = demosaic_image_corrected_fixed(img8_A, 'rggb')
            print(f"Demosaiced image size: {color_image.shape}")
            
            # Save demosaiced image in PNG format (lossless)
            output_color = output_dir / "825-000003_color.png"
            
            if save_image(color_image, str(output_color)):
                print(f"Saved color image: {output_color}")
            
            # Display color image
            display_image(color_image, "Demosaiced Color Image")
            
        except Exception as e:
            print(f"Demosaicing failed: {e}")
            print("This might be due to unsupported Bayer pattern or image format")
        
        print("\n=== Processing Complete ===")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Set matplotlib backend for better display
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
    
    # Run main function
    success = main()
    
    if success:
        print("Script completed successfully!")
    else:
        print("Script failed!")
