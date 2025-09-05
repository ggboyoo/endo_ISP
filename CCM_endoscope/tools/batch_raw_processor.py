#!/usr/bin/env python3
"""
Batch RAW Image Processor
Processes all RAW files in a directory and converts them to PNG format
Uses functions from raw_reader.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple
import time
import logging
from datetime import datetime

# Import configuration
try:
    from batch_config import *
except ImportError:
    print("Warning: batch_config.py not found, using default settings")
    # Default settings
    INPUT_DIRECTORY = r"F:\ZJU\Picture"
    OUTPUT_DIRECTORY = "./data2raw"
    FORCE_DIMENSIONS = None
    RAW_EXTENSIONS = ['.raw', '.RAW', '.Raw']
    MAX_VALUE = 4095
    BAYER_PATTERN = 'auto'
    OUTPUT_FORMAT = 'png'
    PNG_COMPRESSION = 6
    JPEG_QUALITY = 95
    SHOW_PROGRESS = True
    SAVE_LOGS = True
    LOG_FILE = "batch_processing.log"

# Import functions from raw_reader.py
try:
    from raw_reader import (
        read_raw_image, 
        normalize_to_8bit, 
        demosaic_image_corrected,
        demosaic_image_corrected_fixed,  # Add the fixed version
        save_image
    )
except ImportError:
    print("Error: raw_reader.py not found in the same directory!")
    print("Please ensure raw_reader.py is in the same directory as this script.")
    sys.exit(1)


def setup_logging(log_file: str = None) -> logging.Logger:
    """Setup logging configuration"""
    if log_file is None:
        log_file = LOG_FILE
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def get_raw_files(input_dir: str, extensions: List[str] = None) -> List[Path]:
    """
    Get all RAW files from input directory
    
    Args:
        input_dir: Input directory path
        extensions: List of file extensions to process
        
    Returns:
        List of RAW file paths
    """
    if extensions is None:
        extensions = RAW_EXTENSIONS
    
    raw_files = []
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Input directory {input_dir} does not exist!")
        return []
    
    logging.info(f"Scanning directory: {input_dir}")
    
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:
            raw_files.append(file_path)
    
    logging.info(f"Found {len(raw_files)} RAW files")
    return raw_files


def detect_image_dimensions(raw_file: Path) -> Tuple[int, int]:
    """
    Try to detect image dimensions from RAW file
    Common dimensions for different camera types
    """
    # Get file size in bytes
    file_size = raw_file.stat().st_size
    
    # Common camera dimensions (width, height)
    common_dimensions = [
        (3840, 2160),  # 4K
        (1920, 1080),  # Full HD
        (2560, 1440),  # 2K
        (1280, 720),   # HD
        (640, 480),    # VGA
        (800, 600),    # SVGA
        (1024, 768),   # XGA
        (1440, 1080),  # HDV
        (1920, 1440),  # QXGA
        (2560, 1920),  # QSXGA
        (3200, 2400),  # QUXGA
        (4096, 3072),  # HXGA
        (5120, 3840),  # HUXGA
    ]
    
    # Try to find matching dimensions based on file size
    # Assuming uint16 data (2 bytes per pixel)
    for width, height in common_dimensions:
        expected_size = width * height * 2  # uint16 = 2 bytes
        if file_size == expected_size:
            return width, height
    
    # If no exact match, try to infer from file size
    total_pixels = file_size // 2  # uint16 = 2 bytes per pixel
    
    # Try to find reasonable dimensions
    for width, height in common_dimensions:
        if width * height == total_pixels:
            return width, height
    
    # If still no match, try to find square or common aspect ratios
    sqrt_pixels = int(np.sqrt(total_pixels))
    if sqrt_pixels * sqrt_pixels == total_pixels:
        return sqrt_pixels, sqrt_pixels
    
    # Try 4:3 aspect ratio
    width = int(np.sqrt(total_pixels * 4 / 3))
    height = total_pixels // width
    if width * height == total_pixels:
        return width, height
    
    # Try 16:9 aspect ratio
    width = int(np.sqrt(total_pixels * 16 / 9))
    height = total_pixels // width
    if width * height == total_pixels:
        return width, height
    
    # Default fallback
    logging.warning(f"Could not determine dimensions for {raw_file.name}")
    logging.warning(f"File size: {file_size} bytes, estimated pixels: {total_pixels}")
    return 1920, 1080  # Default fallback


def process_raw_file(raw_file: Path, output_dir: Path, 
                    force_dimensions: Tuple[int, int] = None,
                    max_value: int = None,
                    bayer_pattern: str = None,
                    demosaic_method: str = None) -> bool:
    """
    Process a single RAW file
    
    Args:
        raw_file: Path to RAW file
        output_dir: Output directory
        force_dimensions: Force specific dimensions (width, height)
        max_value: Maximum value in RAW data
        bayer_pattern: Bayer pattern for demosaicing
        demosaic_method: Demosaicing method to use
        
    Returns:
        True if successful, False otherwise
    """
    if max_value is None:
        max_value = MAX_VALUE
    if bayer_pattern is None:
        bayer_pattern = BAYER_PATTERN
    if demosaic_method is None:
        demosaic_method = DEMOSAIC_METHOD
    
    try:
        logging.info(f"Processing: {raw_file.name}")
        
        # Detect dimensions if not forced
        if force_dimensions:
            width, height = force_dimensions
            logging.info(f"Using forced dimensions: {width} x {height}")
        else:
            width, height = detect_image_dimensions(raw_file)
            logging.info(f"Detected dimensions: {width} x {height}")
        
        # Read RAW image
        logging.info("Reading RAW file...")
        raw_img = read_raw_image(str(raw_file), width, height, 'uint16')
        logging.info(f"Image loaded: {raw_img.shape}, dtype: {raw_img.dtype}")
        
        # Normalize to 8-bit
        logging.info("Normalizing to 8-bit...")
        img_8bit = normalize_to_8bit(raw_img, max_value)
        
        # Demosaic to color image using specified method
        logging.info(f"Performing demosaicing with method: {demosaic_method}")
        if demosaic_method == 'fixed':
            color_img = demosaic_image_corrected_fixed(img_8bit, bayer_pattern)
        elif demosaic_method == 'test':
            color_img = demosaic_image_with_channel_test(img_8bit, bayer_pattern)
        else:  # 'original'
            color_img = demosaic_image_corrected(img_8bit, bayer_pattern)
        
        logging.info(f"Demosaiced image size: {color_img.shape}")
        
        # Create output filename (same name, .png extension)
        output_filename = raw_file.stem + "." + OUTPUT_FORMAT
        output_path = output_dir / output_filename
        
        # Save image
        logging.info(f"Saving to: {output_path}")
        if save_image(color_img, str(output_path)):
            logging.info(f"✓ Successfully saved: {output_filename}")
            return True
        else:
            logging.error(f"✗ Failed to save: {output_filename}")
            return False
            
    except Exception as e:
        logging.error(f"✗ Error processing {raw_file.name}: {e}")
        return False


def main():
    """Main function for batch processing"""
    # Setup logging
    logger = setup_logging()
    
    print("=== Batch RAW Image Processor ===")
    print(f"Input directory: {INPUT_DIRECTORY}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print(f"Output format: {OUTPUT_FORMAT}")
    print(f"Force dimensions: {FORCE_DIMENSIONS}")
    print(f"Bayer pattern: {BAYER_PATTERN}")
    print(f"Demosaicing method: {DEMOSAIC_METHOD}")
    print(f"Max value: {MAX_VALUE}")
    
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIRECTORY)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified: {output_path.absolute()}")
    
    # Get all RAW files
    print(f"\nScanning for RAW files in: {INPUT_DIRECTORY}")
    raw_files = get_raw_files(INPUT_DIRECTORY, RAW_EXTENSIONS)
    
    if not raw_files:
        print("No RAW files found!")
        return False
    
    print(f"Found {len(raw_files)} RAW files:")
    for raw_file in raw_files:
        print(f"  - {raw_file.name}")
    
    # Process each RAW file
    print(f"\nStarting batch processing...")
    start_time = time.time()
    
    successful = 0
    failed = 0
    
    for i, raw_file in enumerate(raw_files, 1):
        print(f"\n[{i}/{len(raw_files)}] Processing: {raw_file.name}")
        
        if process_raw_file(raw_file, output_path, FORCE_DIMENSIONS, MAX_VALUE, BAYER_PATTERN, DEMOSAIC_METHOD):
            successful += 1
        else:
            failed += 1
    
    # Summary
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n=== Processing Complete ===")
    print(f"Total files: {len(raw_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average time per file: {processing_time/len(raw_files):.2f} seconds")
    print(f"Output directory: {output_path.absolute()}")
    
    if failed == 0:
        print("✓ All files processed successfully!")
        return True
    else:
        print(f"⚠ {failed} files failed to process")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nScript completed successfully!")
        else:
            print("\nScript completed with errors!")
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
