#!/usr/bin/env python3
"""
Lens Shading Regression (Radial) Script
- Build a radial model of lens shading using grid means vs distance from center
- Normalize by center luminance to get gain curve
- Smooth and polynomial-fit the radial gain curve
- Reconstruct full-resolution correction map from the fitted curve

All heavy computations are in float64; quantization only at save time.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import cv2
import json
from datetime import datetime

# Project reader
try:
	from raw_reader import read_raw_image
except Exception:
	read_raw_image = None

# =========================
# Config (edit here)
# =========================
INPUT_RAW: str = r"F:\ZJU\Picture\lens shading\700.raw"
IMAGE_WIDTH: int = 3840
IMAGE_HEIGHT: int = 2160
DATA_TYPE: str = 'uint16'

GRID_SIZE: int = 64            # grid cell size for statistics
LOWER_BOUND: int = 50          # ignore pixels below this (dark / black)
UPPER_BOUND: int = 4095        # upper clamp
BLACK_THRESHOLD: int = 30      # additional ignore threshold
SMOOTH_KERNEL: int = 5         # odd window for smoothing the radial curve
POLY_DEGREE: int = 4           # polynomial degree for radial fitting

# Optional dark level correction
ENABLE_DARK_CORRECTION: bool = True
DARK_PATH: Optional[str] = r"F:\ZJU\Picture\dark\g8\average_dark.raw"

OUTPUT_DIR: Optional[str] = r"F:\ZJU\Picture\lens shading\regress"
SAVE_CORRECTION_RAW: bool = True
SAVE_CURVE_JSON: bool = True
SAVE_DEBUG_PLOTS: bool = True


def compute_grid_means(raw: np.ndarray, grid: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
	"""
	Compute grid means for RGGB sub-sampled planes.

	Returns for each plane a tuple (means, valid_mask), where valid_mask[i,j] is False
	if the corresponding block contains any pixel below BLACK_THRESHOLD or has no valid pixels.
	"""
	h, w = raw.shape
	yh = np.arange(grid//2, h, grid)
	xw = np.arange(grid//2, w, grid)
	# RGGB planes downsampled by 2
	R = raw[0::2, 0::2]
	G1 = raw[0::2, 1::2]
	G2 = raw[1::2, 0::2]
	B = raw[1::2, 1::2]
	# We will compute per-channel grid means on these planes with step grid//2
	stride = max(1, grid//2)
	def grid_mean_plane(plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		ph, pw = plane.shape
		gy = np.arange(stride//2, ph, stride)
		gx = np.arange(stride//2, pw, stride)
		means = np.zeros((gy.size, gx.size), dtype=np.float64)
		valid = np.zeros((gy.size, gx.size), dtype=bool)
		for i, cy in enumerate(gy):
			for j, cx in enumerate(gx):
				y0 = max(0, cy - stride//2)
				y1 = min(ph, cy + stride//2)
				x0 = max(0, cx - stride//2)
				x1 = min(pw, cx + stride//2)
				blk = plane[y0:y1, x0:x1]
				# If any pixel is below BLACK_THRESHOLD, exclude this block entirely
				if np.any(blk < BLACK_THRESHOLD):
					means[i, j] = 0.0
					valid[i, j] = False
					continue
				mask = (blk >= max(LOWER_BOUND, BLACK_THRESHOLD)) & (blk <= UPPER_BOUND)
				vals = blk[mask]
				if vals.size > 0:
					means[i, j] = float(np.mean(vals))
					valid[i, j] = True
				else:
					means[i, j] = 0.0
					valid[i, j] = False
		return means, valid
	return grid_mean_plane(R), grid_mean_plane(G1), grid_mean_plane(G2), grid_mean_plane(B)


def build_radial_samples(grid_means: np.ndarray, grid_valid: np.ndarray, plane_shape: Tuple[int, int], step: int) -> Tuple[np.ndarray, np.ndarray]:
	"""From grid mean map (on plane) build (r, gain) samples normalized by center."""
	gh, gw = grid_means.shape
	ph, pw = plane_shape
	# centers in plane coordinates
	gy = np.linspace(step//2, ph - step//2, gh)
	gx = np.linspace(step//2, pw - step//2, gw)
	YY, XX = np.meshgrid(gy, gx, indexing='ij')
	cy, cx = (ph - 1) / 2.0, (pw - 1) / 2.0
	r = np.sqrt((YY - cy) ** 2 + (XX - cx) ** 2)
	
	m = grid_means.astype(np.float64)
	# robust center mean: small window around center, fallback to global positive mean
	if gh > 2 and gw > 2:
		c_patch = m[max(0, gh//2-1):min(gh, gh//2+2), max(0, gw//2-1):min(gw, gw//2+2)]
		c_vals = c_patch[c_patch > 0]
		c_val = float(np.mean(c_vals)) if c_vals.size > 0 else float(np.mean(m[m>0]))
	else:
		c_val = float(np.mean(m[m>0]))
	c_val = c_val if (c_val is not None and c_val > 0) else 1.0
	gain = np.zeros_like(m)
	mask = (m > 0) & (grid_valid.astype(bool))
	gain[mask] = c_val / m[mask]
	return r[mask].ravel(), gain[mask].ravel()


def smooth_curve(y: np.ndarray, kernel: int) -> np.ndarray:
	if kernel <= 1:
		return y
	k = kernel if kernel % 2 == 1 else kernel + 1
	pad = k // 2
	yp = np.pad(y, (pad, pad), mode='edge')
	w = np.ones(k, dtype=np.float64) / k
	return np.convolve(yp, w, mode='valid')


def fit_polynomial(r: np.ndarray, g: np.ndarray, degree: int) -> np.ndarray:
	"""Fit polynomial ensuring monotonic increase from 0 to max distance."""
	from scipy.optimize import minimize
	
	degree = max(1, int(degree))
	r_max = np.max(r)
	
	# Initial guess using standard polyfit
	coeff_init = np.polyfit(r, g, degree)
	
	# Define objective function (sum of squared residuals)
	def objective(coeffs):
		p = np.poly1d(coeffs)
		y_pred = p(r)
		return np.sum((g - y_pred) ** 2)
	
	# Define constraint: derivative should be non-negative in [0, r_max]
	def monotonic_constraint(coeffs):
		# For polynomial p(x) = a_n*x^n + ... + a_1*x + a_0
		# Derivative p'(x) = n*a_n*x^(n-1) + ... + a_1
		# We want p'(x) >= 0 for x in [0, r_max]
		p_deriv = np.polyder(np.poly1d(coeffs))
		
		# Sample derivative at multiple points in [0, r_max]
		x_test = np.linspace(0, r_max, 20)
		deriv_values = p_deriv(x_test)
		return deriv_values  # Should be >= 0
	
	# Bounds: reasonable coefficient ranges
	bounds = [(-10, 10)] * (degree + 1)
	
	try:
		# Try constrained optimization
		result = minimize(
			objective, 
			coeff_init, 
			method='SLSQP',
			bounds=bounds,
			constraints={'type': 'ineq', 'fun': monotonic_constraint},
			options={'maxiter': 1000}
		)
		
		if result.success:
			coeff = result.x
		else:
			# Fallback to unconstrained if constrained fails
			coeff = coeff_init
			print(f"Warning: Monotonic constraint failed, using unconstrained fit")
			
	except Exception as e:
		# Fallback to standard polyfit if optimization fails
		coeff = coeff_init
		print(f"Warning: Optimization failed ({e}), using standard polyfit")
	
	return coeff

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def reconstruct_full_map(h: int, w: int, coeff: np.ndarray) -> np.ndarray:
	cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
	Y, X = np.indices((h, w))
	r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
	
	p = np.poly1d(coeff)
	corr = p(r)
	
	# Ensure monotonic behavior: if not monotonic, apply post-processing
	p_deriv = np.polyder(p)
	deriv_values = p_deriv(r)
	
	# If derivative becomes negative, enforce monotonicity
	if np.any(deriv_values < -1e-6):  # Small tolerance for numerical errors
		# Sort by radius and ensure monotonic increase
		r_flat = r.ravel()
		corr_flat = corr.ravel()
		sort_idx = np.argsort(r_flat)
		r_sorted = r_flat[sort_idx]
		corr_sorted = corr_flat[sort_idx]
		
		# Apply monotonic constraint: each value >= previous
		for i in range(1, len(corr_sorted)):
			if corr_sorted[i] < corr_sorted[i-1]:
				corr_sorted[i] = corr_sorted[i-1]
		
		# Restore original order
		corr_flat[sort_idx] = corr_sorted
		corr = corr_flat.reshape(corr.shape)
	
	corr = np.clip(corr, 0.5, 2.5)
	return corr.astype(np.float64)


def reconstruct_full_map_per_channel(raw_shape: Tuple[int, int], coeffs: Dict[str, np.ndarray]) -> np.ndarray:
	"""Reconstruct a full-size Bayer correction map from per-channel polynomial coeffs on planes."""
	h, w = raw_shape
	ph, pw = h // 2, w // 2
	cy, cx = (ph - 1) / 2.0, (pw - 1) / 2.0
	Yp, Xp = np.indices((ph, pw))
	rp = np.sqrt((Yp - cy) ** 2 + (Xp - cx) ** 2)
	
	def eval_poly(c):
		p = np.poly1d(c)
		v = p(rp)
		
		# Ensure monotonic behavior: if not monotonic, apply post-processing
		p_deriv = np.polyder(p)
		deriv_values = p_deriv(rp)
		
		# If derivative becomes negative, enforce monotonicity
		if np.any(deriv_values < -1e-6):  # Small tolerance for numerical errors
			# Sort by radius and ensure monotonic increase
			r_flat = rp.ravel()
			v_flat = v.ravel()
			sort_idx = np.argsort(r_flat)
			r_sorted = r_flat[sort_idx]
			v_sorted = v_flat[sort_idx]
			
			# Apply monotonic constraint: each value >= previous
			for i in range(1, len(v_sorted)):
				if v_sorted[i] < v_sorted[i-1]:
					v_sorted[i] = v_sorted[i-1]
			
			# Restore original order
			v_flat[sort_idx] = v_sorted
			v = v_flat.reshape(v.shape)
		
		return np.clip(v, 0.5, 2.5)
	
	Rmap = eval_poly(coeffs['R'])
	G1map = eval_poly(coeffs['G1'])
	G2map = eval_poly(coeffs['G2'])
	Bmap = eval_poly(coeffs['B'])
	full = np.ones((h, w), dtype=np.float64)
	full[0::2, 0::2] = Rmap
	full[0::2, 1::2] = G1map
	full[1::2, 0::2] = G2map
	full[1::2, 1::2] = Bmap
	return full


def reconstruct_plane_maps(raw_shape: Tuple[int, int], coeffs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
	"""Return per-plane (H/2,W/2) correction maps for R,G1,G2,B based on polynomial coeffs."""
	h, w = raw_shape
	ph, pw = h // 2, w // 2
	cy, cx = (ph - 1) / 2.0, (pw - 1) / 2.0
	Yp, Xp = np.indices((ph, pw))
	rp = np.sqrt((Yp - cy) ** 2 + (Xp - cx) ** 2)
	
	def eval_poly(c):
		p = np.poly1d(c)
		v = p(rp)
		
		# Ensure monotonic behavior: if not monotonic, apply post-processing
		p_deriv = np.polyder(p)
		deriv_values = p_deriv(rp)
		
		# If derivative becomes negative, enforce monotonicity
		if np.any(deriv_values < -1e-6):  # Small tolerance for numerical errors
			# Sort by radius and ensure monotonic increase
			r_flat = rp.ravel()
			v_flat = v.ravel()
			sort_idx = np.argsort(r_flat)
			r_sorted = r_flat[sort_idx]
			v_sorted = v_flat[sort_idx]
			
			# Apply monotonic constraint: each value >= previous
			for i in range(1, len(v_sorted)):
				if v_sorted[i] < v_sorted[i-1]:
					v_sorted[i] = v_sorted[i-1]
			
			# Restore original order
			v_flat[sort_idx] = v_sorted
			v = v_flat.reshape(v.shape)
		
		return np.clip(v, 0.5, 2.5).astype(np.float64)
	
	return {
		'R': eval_poly(coeffs['R']),
		'G1': eval_poly(coeffs['G1']),
		'G2': eval_poly(coeffs['G2']),
		'B': eval_poly(coeffs['B']),
	}


def main() -> None:
	print("=== Lens Shading Regression (Radial) ===")
	if read_raw_image is None:
		raise RuntimeError('raw_reader.read_raw_image not available')
	if not os.path.exists(INPUT_RAW):
		raise FileNotFoundError(INPUT_RAW)

	raw = read_raw_image(INPUT_RAW, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
	print(f"Loaded RAW: {raw.shape}, dtype={raw.dtype}, range {np.min(raw)}-{np.max(raw)}")

	# Optional dark correction
	if ENABLE_DARK_CORRECTION and DARK_PATH and os.path.exists(DARK_PATH):
		try:
			dark = read_raw_image(DARK_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
			if dark is not None and dark.shape == raw.shape:
				raw = np.clip(raw.astype(np.float64) - dark.astype(np.float64), 0, None)
				print(f"Applied dark correction: range {raw.min()}-{raw.max()}")
		except Exception as e:
			print(f"Warn: failed dark correction: {e}")

	# Compute grid means and validity on RGGB planes
	(Rgm, Rval), (G1gm, G1val), (G2gm, G2val), (Bgm, Bval) = compute_grid_means(raw, GRID_SIZE)
	ph, pw = raw[0::2, 0::2].shape
	step = max(1, GRID_SIZE // 2)

	# Per-channel radial fitting
	coeffs: Dict[str, np.ndarray] = {}
	curve_debug = {}
	for name, gm, gv in [('R', Rgm, Rval), ('G1', G1gm, G1val), ('G2', G2gm, G2val), ('B', Bgm, Bval)]:
		r_i, g_i = build_radial_samples(gm, gv, (ph, pw), step)
		idx = np.argsort(r_i)
		r_sorted = r_i[idx]
		g_sorted = g_i[idx]
		g_sm = smooth_curve(g_sorted, SMOOTH_KERNEL)
		coeffs[name] = fit_polynomial(r_sorted, g_sm, POLY_DEGREE)
		# jitter radii slightly to avoid overlap for identical r
		r_plot = r_sorted + 0.01 * np.arange(r_sorted.size)
		p = np.poly1d(coeffs[name])
		y_fit = p(r_sorted)
		r2 = compute_r2(g_sm, y_fit)
		
		curve_debug[name] = {
			'r': r_sorted.tolist(),
			'g_raw': g_sorted.tolist(),
			'g_sm': g_sm.tolist(),
			'coeff': coeffs[name].tolist(),
			'r_plot': r_plot.tolist(),
			'r2': r2,
		}
		print(f"Fitted {name} poly (deg={POLY_DEGREE}): {coeffs[name]}  R2={r2:.4f}")

	# Reconstruct full Bayer correction map from per-channel coeffs
	corr = reconstruct_full_map_per_channel((IMAGE_HEIGHT, IMAGE_WIDTH), coeffs)
	print(f"Correction map: range {corr.min():.3f}-{corr.max():.3f}")

	# Build per-plane arrays for saving/applying elsewhere
	plane_maps = reconstruct_plane_maps((IMAGE_HEIGHT, IMAGE_WIDTH), coeffs)

	# Save results
	out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(INPUT_RAW).parent
	out_dir.mkdir(parents=True, exist_ok=True)

	if SAVE_CORRECTION_RAW:
		corr_u16 = np.clip(corr * 1024.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)  # scaled save
		(out_dir / 'lens_shading_correction_regress.raw').write_bytes(corr_u16.tobytes())
		print(f"Saved correction map (scaled) to: {out_dir / 'lens_shading_correction_regress.raw'}")
		# Save per-plane maps as npy for convenience
		for k, v in plane_maps.items():
			np.save(out_dir / f'lens_shading_plane_{k}.npy', v)
		print(f"Saved per-plane correction arrays (R,G1,G2,B) as .npy")

	if SAVE_CURVE_JSON:
		meta = {
			'timestamp': datetime.now().isoformat(),
			'input_raw': INPUT_RAW,
			'grid_size': GRID_SIZE,
			'lower_bound': LOWER_BOUND,
			'poly_degree': POLY_DEGREE,
			'smooth_kernel': SMOOTH_KERNEL,
			'per_channel_coeff': {k: v.tolist() for k, v in coeffs.items()},
			'per_channel_curves': {k: {'len': len(v['r'])} for k, v in curve_debug.items()}
		}
		with open(out_dir / 'lens_shading_regress_curve.json', 'w', encoding='utf-8') as f:
			json.dump(meta, f, indent=2, ensure_ascii=False)
		print(f"Saved curve meta to: {out_dir / 'lens_shading_regress_curve.json'}")

	# Save combined JSON for ISP lensshading interface (RGGB small maps)
	try:
		combined = {
			'analysis_timestamp': datetime.now().isoformat(),
			'image_dimensions': [IMAGE_HEIGHT, IMAGE_WIDTH],
			'grid_size': GRID_SIZE,
			'channel_names': ['R','G1','G2','B'],
			'correction_matrices': {
				'R': {
					'average_correction': plane_maps['R'].round(6).tolist(),
					'shape': [plane_maps['R'].shape[0], plane_maps['R'].shape[1]]
				},
				'G1': {
					'average_correction': plane_maps['G1'].round(6).tolist(),
					'shape': [plane_maps['G1'].shape[0], plane_maps['G1'].shape[1]]
				},
				'G2': {
					'average_correction': plane_maps['G2'].round(6).tolist(),
					'shape': [plane_maps['G2'].shape[0], plane_maps['G2'].shape[1]]
				},
				'B': {
					'average_correction': plane_maps['B'].round(6).tolist(),
					'shape': [plane_maps['B'].shape[0], plane_maps['B'].shape[1]]
				}
			}
		}
		with open(out_dir / 'combined_lens_shading_correction.json', 'w', encoding='utf-8') as f:
			json.dump(combined, f, indent=2, ensure_ascii=False)
		print(f"Saved combined RGGB plane maps to: {out_dir / 'combined_lens_shading_correction.json'}")
	except Exception as e:
		print(f"Warn: failed to save combined_lens_shading_correction.json: {e}")

	if SAVE_DEBUG_PLOTS:
		try:
			import matplotlib.pyplot as plt
			for name, color in [('R','red'),('G1','green'),('G2','lime'),('B','blue')]:
				cd = curve_debug[name]
				coeff = coeffs[name]
				p = np.poly1d(coeff)
				r_arr = np.array(cd['r'], dtype=np.float64)
				r_plot = np.array(cd['r_plot'], dtype=np.float64)
				g_raw = np.array(cd['g_raw'], dtype=np.float64)
				g_sm = np.array(cd['g_sm'], dtype=np.float64)
				y_fit = p(r_arr)
				r2 = cd['r2']
				# Create per-channel figure
				plt.figure(figsize=(9,4))
				plt.plot(r_plot, g_raw, '.', alpha=0.12, color=color, label='samples')
				plt.plot(r_plot, g_sm, '-', linewidth=1.2, color=color, label='smoothed')
				plt.plot(r_plot, y_fit, '--', linewidth=1.0, color='k', label='poly fit')
				# Text: polynomial and R2
				coeff_txt = ' + '.join([f"{c:.3e} r^{len(coeff)-i-1}" for i, c in enumerate(coeff)])
				txt = f"fit: {coeff_txt}\nR² = {r2:.4f}"
				plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va='top', ha='left', fontsize=8,
						 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
				plt.xlabel('radius (pixels)')
				plt.ylabel('gain (center/mean)')
				plt.title(f'Radial Gain Curve - {name}')
				plt.grid(True, alpha=0.3)
				plt.legend(loc='lower right', fontsize=8)
				plt.tight_layout()
				plt.savefig(out_dir / f'lens_shading_regress_curve_{name}.png', dpi=300)
				plt.show()
		except Exception as e:
			print(f"Warn: failed to plot debug curve: {e}")

	# Show correction effect (preview)
	try:
		# Apply correction to original raw (float path)
		raw_float = raw.astype(np.float64)
		corrected = np.clip(raw_float * corr, 0, 4095)
		import matplotlib.pyplot as plt
		fig, axes = plt.subplots(1, 4, figsize=(20,4))
		axes[0].imshow(raw_float, cmap='gray')
		axes[0].set_title('Original RAW')
		axes[0].axis('off')
		axes[1].imshow(corr, cmap='viridis')
		axes[1].set_title('Correction Map')
		axes[1].axis('off')
		axes[2].imshow(corrected, cmap='gray')
		axes[2].set_title('Corrected RAW')
		axes[2].axis('off')
		# Difference map (Corrected - Original), symmetric color limits
		diff = corrected - raw_float
		amax = float(np.max(np.abs(diff))) if np.isfinite(diff).any() else 1.0
		v = max(1.0, amax)
		im = axes[3].imshow(diff, cmap='coolwarm', vmin=-v, vmax=v)
		axes[3].set_title('Difference (Corr - Orig)')
		axes[3].axis('off')
		plt.tight_layout()
		plt.savefig(out_dir / 'lens_shading_regress_preview.png', dpi=300)
		plt.show()
	except Exception as e:
		print(f"Warn: failed to render correction preview: {e}")

	print("Done.")


if __name__ == '__main__':
	main()
