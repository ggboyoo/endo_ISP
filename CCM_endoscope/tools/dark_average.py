#!/usr/bin/env python3
"""
Dark RAW Averaging Script
- Scan a directory for .raw files
- Compute per-pixel average image
- Save as average.raw
- Compare variance of the first RAW versus the averaged image

Configuration is defined inside this script.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Project raw reader
try:
	from raw_reader import read_raw_image
except ImportError:
	print("Error: raw_reader.py not found in the same directory!")
	print("Please ensure raw_reader.py is in the same directory as this script.")
	raise

# =========================
# Configuration (edit here)
# =========================
DARK_DIR: str = r"F:\ZJU\Picture\dark\g3"  # directory containing raw dark frames
IMAGE_WIDTH: int = 3840
IMAGE_HEIGHT: int = 2160
DATA_TYPE: str = 'uint16'  # 'uint8'|'uint16'|'uint32'
OUTPUT_PATH: str = str(Path(DARK_DIR) / 'average_dark.raw')
GLOB_PATTERN: str = '*.raw'


def list_raw_files(directory: str, pattern: str = '*.raw') -> List[Path]:
	"""List RAW files in a directory sorted by name."""
	d = Path(directory)
	files = sorted(d.glob(pattern))
	return [f for f in files if f.is_file()]


def save_raw_image(path: str, array: np.ndarray) -> None:
	"""Save ndarray to raw binary file (no header)."""
	out_path = Path(path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	array.astype(np.uint16).tofile(str(out_path))


def compute_average_raw(files: List[Path], width: int, height: int, data_type: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	"""
	Compute per-pixel average of RAW files.
	Returns (average_image, first_image).
	"""
	if not files:
		raise ValueError('No RAW files found to average')

	accumulator = np.zeros((height, width), dtype=np.float64)
	first_image: Optional[np.ndarray] = None

	for idx, f in enumerate(files):
		img = read_raw_image(str(f), width, height, data_type)
		if img is None:
			print(f"Warning: failed to read {f}, skipping")
			continue
		if img.shape != (height, width):
			raise ValueError(f"Dimension mismatch for {f.name}: got {img.shape}, expected {(height, width)}")
		if first_image is None:
			first_image = img.copy()
		accumulator += img.astype(np.float64)

	count = len(files)
	if count == 0:
		raise ValueError('No valid RAW files were read')

	avg = accumulator / float(count)
	avg = np.clip(avg, 0, np.iinfo(np.uint16).max).astype(np.uint16)
	return avg, first_image


def main() -> None:
	print('=' * 60)
	print('Dark RAW Averaging')
	print('=' * 60)
	print(f'Directory: {DARK_DIR}')
	print(f'Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}')
	print(f'Data type: {DATA_TYPE}')
	print(f'Output: {OUTPUT_PATH}')
	print('=' * 60)

	files = list_raw_files(DARK_DIR, GLOB_PATTERN)
	print(f'Found {len(files)} RAW files')
	if not files:
		return

	avg, first = compute_average_raw(files, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
	save_raw_image(OUTPUT_PATH, avg)
	print(f'Saved average image to {OUTPUT_PATH}')

	# Mean and Variance comparison
	first_mean = float(np.mean(first.astype(np.float64))) if first is not None else float('nan')
	avg_mean = float(np.mean(avg.astype(np.float64)))
	first_var = float(np.var(first.astype(np.float64))) if first is not None else float('nan')
	avg_var = float(np.var(avg.astype(np.float64)))
	print(f'Mean (first RAW): {first_mean:.3f}')
	print(f'Mean (average.raw): {avg_mean:.3f}')
	print(f'Variance (first RAW): {first_var:.3f}')
	print(f'Variance (average.raw): {avg_var:.3f}')

	# Optional: print expected reduction ~ 1/N for independent noise
	if len(files) > 1 and np.isfinite(first_var):
		expected = first_var / len(files)
		print(f'Expected variance if uncorrelated: {expected:.3f} (first/N)')


if __name__ == '__main__':
	main()
