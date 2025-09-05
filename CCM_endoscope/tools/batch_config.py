#!/usr/bin/env python3
"""
Configuration file for batch RAW processor
Modify these settings as needed
"""

# Input and output directories
INPUT_DIRECTORY = r"F:\ZJU\Picture"  # Directory containing RAW files
OUTPUT_DIRECTORY = "./data2raw"       # Output directory for PNG files

# Processing settings
FORCE_DIMENSIONS = None  # Set to (width, height) to force specific dimensions
                          # Example: (3840, 2160) for 4K
                          # Set to None for automatic detection

# RAW file detection
RAW_EXTENSIONS = ['.raw', '.RAW', '.Raw']  # File extensions to process

# Image processing settings
MAX_VALUE = 4095  # Maximum value in RAW data (usually 4095 for 12-bit, 65535 for 16-bit)
BAYER_PATTERN = 'auto'  # Bayer pattern: 'auto', 'rggb', 'bggr', 'grbg', 'gbrg'
DEMOSAIC_METHOD = 'fixed'  # Demosaicing method: 'original', 'fixed', 'test'

# Output settings
OUTPUT_FORMAT = 'png'  # Output format: 'png', 'jpg'
PNG_COMPRESSION = 6    # PNG compression level (0-9, higher = smaller file but slower)
JPEG_QUALITY = 95      # JPEG quality (1-100, higher = better quality but larger file)

# Performance settings
SHOW_PROGRESS = True    # Show progress bar and detailed output
SAVE_LOGS = True        # Save processing logs to file
LOG_FILE = "batch_processing.log"  # Log file name
