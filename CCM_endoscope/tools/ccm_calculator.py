import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import minimize


# ========================= Reference Data (Macbeth ColorChecker Classic) =========================
# sRGB (D65) 8-bit values from widely used BabelColor dataset, ordered row-wise (6x4)
# Order: row1 -> row4, each row has 6 patches
# Source commonly referenced as: http://www.babelcolor.com/index_htm_files/ColorChecker_RGB_and_spectra.xls

SRGB_24PATCH_D65_8BIT = np.array([
    [115,  82,  68], [194, 150, 130], [ 98, 122, 157], [ 87, 108,  67], [133, 128, 177], [103, 189, 170],
    [214, 126,  44], [ 80,  91, 166], [193,  90,  99], [ 94,  60, 108], [157, 188,  64], [224, 163,  46],
    [ 56,  61, 150], [ 70, 148,  73], [175,  54,  60], [231, 199,  31], [187,  86, 149], [  8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [ 85,  85,  85], [ 52,  52,  52],
], dtype=np.float32)


# ========================= Color Space Utilities =========================

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(srgb, 0.0, 1.0)
    threshold = 0.04045
    low = srgb <= threshold
    high = ~low
    out = np.zeros_like(srgb)
    out[low] = srgb[low] / 12.92
    out[high] = ((srgb[high] + 0.055) / 1.055) ** 2.4
    return out


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    threshold = 0.0031308
    low = lin <= threshold
    high = ~low
    out = np.zeros_like(lin)
    out[low] = lin[low] * 12.92
    out[high] = 1.055 * (lin[high] ** (1 / 2.4)) - 0.055
    return out


# sRGB (linear) to XYZ (D65) matrix (from IEC 61966-2-1)
M_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)


def rgb_linear_to_xyz(rgb_lin: np.ndarray) -> np.ndarray:
    return rgb_lin @ M_RGB_TO_XYZ.T


# D65 reference white (CIE 1931 2°)
XYZ_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    # Normalize by white
    xyz_n = xyz / XYZ_WHITE_D65

    def f(t):
        delta = 6/29
        return np.where(
            t > delta**3,
            np.cbrt(t),
            t / (3 * delta**2) + 4/29,
        )

    fx, fy, fz = f(xyz_n[..., 0]), f(xyz_n[..., 1]), f(xyz_n[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    diff = lab1 - lab2
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ========================= Geometry and Sampling =========================

def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    # Robust image read supporting non-ASCII Windows paths
    if not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite_unicode(path: str, image_bgr: np.ndarray, params: Optional[List[int]] = None) -> bool:
    # Robust image write supporting non-ASCII Windows paths
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".jpg"
        path = path + ext
    success, buf = cv2.imencode(ext, image_bgr, params if params is not None else [])
    if not success:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False

def order_corners(points: np.ndarray) -> np.ndarray:
    # Accept 4 points in any order; return [tl, tr, br, bl]
    assert points.shape == (4, 2)
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).reshape(-1)
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmin(d)]
    bl = points[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_chart(image_bgr: np.ndarray, corners_xy: Optional[np.ndarray], target_size: Tuple[int, int]) -> np.ndarray:
    height, width = target_size[1], target_size[0]
    if corners_xy is None:
        return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)

    pts_src = order_corners(corners_xy.astype(np.float32))
    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image_bgr, H, (width, height))
    return warped


def sample_chart_6x4_rgb(image_bgr: np.ndarray, cell_margin: float = 0.15) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cell_w = w / 6.0
    cell_h = h / 4.0
    margin_x = cell_w * cell_margin
    margin_y = cell_h * cell_margin

    samples = []
    for row in range(4):
        for col in range(6):
            x0 = int(col * cell_w + margin_x)
            x1 = int((col + 1) * cell_w - margin_x)
            y0 = int(row * cell_h + margin_y)
            y1 = int((row + 1) * cell_h - margin_y)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w - 1), min(y1, h - 1)
            roi = image_bgr[y0:y1, x0:x1]
            mean_bgr = roi.reshape(-1, 3).mean(axis=0)
            mean_rgb = mean_bgr[::-1]
            samples.append(mean_rgb)
    return np.array(samples, dtype=np.float64)


def crop_roi_interactive(image_bgr: np.ndarray) -> np.ndarray:
    # User selects ROI; returns cropped image. Press Enter/Space to confirm, c to cancel.
    r = cv2.selectROI("Select ROI (Enter/Space confirm)", image_bgr, False, False)
    x, y, w, h = map(int, r)
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection canceled or invalid. Please try again.")
    roi = image_bgr[y:y+h, x:x+w]
    cv2.destroyWindow("Select ROI (Enter/Space confirm)")
    return roi


def get_patch_boxes(image_bgr: np.ndarray, cell_margin: float = 0.15) -> List[Tuple[int, int, int, int]]:
    # Returns list of 24 boxes (x0, y0, x1, y1) row-major 4x6
    h, w = image_bgr.shape[:2]
    cell_w = w / 6.0
    cell_h = h / 4.0
    margin_x = cell_w * cell_margin
    margin_y = cell_h * cell_margin
    boxes: List[Tuple[int, int, int, int]] = []
    for row in range(4):
        for col in range(6):
            x0 = int(col * cell_w + margin_x)
            x1 = int((col + 1) * cell_w - margin_x)
            y0 = int(row * cell_h + margin_y)
            y1 = int((row + 1) * cell_h - margin_y)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w - 1), min(y1, h - 1)
            boxes.append((x0, y0, x1, y1))
    return boxes


def visualize_patches(
    image_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    save_overlay_path: Optional[str] = None,
    save_montage_path: Optional[str] = None,
    show_windows: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    # Draw overlay
    overlay = image_bgr.copy()
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        color = (0, 255, 0)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(overlay, str(idx + 1), (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Build montage 4 rows x 6 cols
    tile_size = 100
    montage = np.zeros((4 * tile_size, 6 * tile_size, 3), dtype=np.uint8)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        patch = image_bgr[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        patch_resized = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, 6)
        montage[r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size] = patch_resized

    if save_overlay_path is not None:
        imwrite_unicode(save_overlay_path, overlay)
    if save_montage_path is not None:
        imwrite_unicode(save_montage_path, montage)

    if show_windows:
        cv2.imshow("ROI with 24 cells", overlay)
        cv2.imshow("Patches montage (4x6)", montage)
        cv2.waitKey(0)
        cv2.destroyWindow("ROI with 24 cells")
        cv2.destroyWindow("Patches montage (4x6)")

    return overlay, montage


# ========================= CCM Solve =========================

@dataclass
class CCMSolveConfig:
    model: str = "affine3x4"  # "linear3x3" or "affine3x4"
    lambda_reg: float = 0.0
    regularize_bias: bool = False
    use_gradient_optimization: bool = False
    max_iterations: int = 100
    tolerance: float = 1e-6
    preserve_white: bool = False  # Ensure CCM rows sum to 1


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    
    def f_inv(t):
        delta = 6/29
        return np.where(
            t > delta,
            t**3,
            3 * delta**2 * (t - 4/29)
        )
    
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    xyz_n = np.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=-1)
    return xyz_n * XYZ_WHITE_D65


def xyz_to_rgb_linear(xyz: np.ndarray) -> np.ndarray:
    return xyz @ np.linalg.inv(M_RGB_TO_XYZ).T


def compute_delta_e_loss(ccm_params: np.ndarray, X: np.ndarray, Y: np.ndarray, model: str) -> float:
    """Compute total DeltaE loss for gradient optimization"""
    # Reshape parameters back to matrix
    if model == "linear3x3":
        M = ccm_params.reshape(3, 3)
        corrected = X @ M
    else:  # affine3x4
        M = ccm_params.reshape(4, 3)
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        A = np.concatenate([X, ones], axis=1)
        corrected = A @ M
    
    # Clip to valid range
    corrected = np.clip(corrected, 0.0, 1.0)
    
    # Convert to Lab for DeltaE computation
    corrected_xyz = rgb_linear_to_xyz(corrected)
    reference_xyz = rgb_linear_to_xyz(Y)
    
    corrected_lab = xyz_to_lab(corrected_xyz)
    reference_lab = xyz_to_lab(reference_xyz)
    
    # Compute DeltaE
    delta_e = delta_e_cie76(corrected_lab, reference_lab)
    
    # Return mean DeltaE as loss
    return float(np.mean(delta_e))


def solve_ccm_with_white_constraint(measured_rgb: np.ndarray, reference_rgb: np.ndarray, cfg: CCMSolveConfig) -> np.ndarray:
    """Solve CCM with constraint that each row sums to 1 (preserve white)"""
    assert measured_rgb.shape == reference_rgb.shape == (24, 3)
    X = measured_rgb
    Y = reference_rgb

    if cfg.model == "linear3x3":
        # For 3x3 matrix, solve with constraint: sum of each row = 1
        # This ensures white input (1,1,1) maps to white output (1,1,1)
        
        # Set up the constrained least squares problem
        # We want to minimize ||AX - Y||^2 subject to A*[1,1,1]^T = [1,1,1]^T
        # This means each row of A must sum to 1
        
        # Use Lagrange multipliers approach
        # Solve: [X^T X, C^T; C, 0] [A^T; lambda] = [X^T Y; d]
        # where C = [1,1,1] (constraint matrix) and d = [1,1,1] (desired output)
        
        XtX = X.T @ X
        XtY = X.T @ Y
        C = np.ones((1, 3))  # Constraint: sum of each row = 1
        d = np.ones((1, 3))  # Desired output for white input
        
        # Build augmented system
        n = XtX.shape[0]
        m = C.shape[0]
        A_aug = np.zeros((n + m, n + m))
        b_aug = np.zeros((n + m, 3))
        
        A_aug[:n, :n] = XtX
        A_aug[:n, n:] = C.T
        A_aug[n:, :n] = C
        b_aug[:n, :] = XtY
        b_aug[n:, :] = d
        
        # Solve augmented system
        solution = np.linalg.solve(A_aug, b_aug)
        M = solution[:n, :].T
        
    elif cfg.model == "affine3x4":
        # For 4x4 affine matrix, we need to ensure that for input [1,1,1,1]
        # the output is [1,1,1] (excluding the bias term)
        # This means: A[1,1,1] + b = [1,1,1] where A is 3x3 and b is bias
        # So A[1,1,1] = [1,1,1] - b, meaning each row of A must sum to (1 - bias)
        
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        A_full = np.concatenate([X, ones], axis=1)
        
        # For affine model, we can't easily enforce the constraint
        # So we'll solve normally and then normalize the 3x3 part
        AtA = A_full.T @ A_full
        reg = np.eye(AtA.shape[0], dtype=A_full.dtype) * cfg.lambda_reg
        if not cfg.regularize_bias:
            reg[-1, -1] = 0.0
        AtY = A_full.T @ Y
        M_full = np.linalg.solve(AtA + reg, AtY)
        
        # Extract 3x3 part and bias
        M_3x3 = M_full[:3, :].T
        bias = M_full[3, :]
        
        # Normalize 3x3 part so each row sums to (1 - bias)
        row_sums = M_3x3.sum(axis=1)
        target_sums = 1.0 - bias
        scale_factors = target_sums / row_sums
        M_3x3 = M_3x3 * scale_factors.reshape(-1, 1)
        
        # Reconstruct full matrix
        M = np.concatenate([M_3x3.T, bias.reshape(1, -1)], axis=0)
        
    else:
        raise ValueError("model must be 'linear3x3' or 'affine3x4'")
    
    return M


def solve_ccm_gradient_optimization(measured_rgb: np.ndarray, reference_rgb: np.ndarray, cfg: CCMSolveConfig) -> np.ndarray:
    """Solve CCM using gradient-based optimization in Lab space"""
    assert measured_rgb.shape == reference_rgb.shape == (24, 3)
    
    # Initial guess using least squares
    if cfg.model == "linear3x3":
        A = measured_rgb
        M_init = np.linalg.lstsq(A, reference_rgb, rcond=None)[0].T
        initial_params = M_init.flatten()
    else:  # affine3x4
        ones = np.ones((measured_rgb.shape[0], 1), dtype=measured_rgb.dtype)
        A = np.concatenate([measured_rgb, ones], axis=1)
        M_init = np.linalg.lstsq(A, reference_rgb, rcond=None)[0].T
        initial_params = M_init.flatten()
    
    # Define objective function
    def objective(params):
        return compute_delta_e_loss(params, measured_rgb, reference_rgb, cfg.model)
    
    # Add constraint for white preservation if enabled
    if cfg.preserve_white:
        if cfg.model == "linear3x3":
            # Constraint: each row sums to 1
            def constraint(params):
                M = params.reshape(3, 3)
                return M.sum(axis=1) - 1.0
            
            from scipy.optimize import LinearConstraint
            constraints = LinearConstraint(
                A=np.kron(np.eye(3), np.ones(3)),
                lb=np.ones(3),
                ub=np.ones(3)
            )
        else:
            # For affine model, we'll handle constraint in the loss function
            constraints = None
    else:
        constraints = None
    
    # Run optimization
    result = minimize(
        objective,
        initial_params,
        method='SLSQP' if cfg.preserve_white else 'L-BFGS-B',
        constraints=constraints,
        options={
            'maxiter': cfg.max_iterations,
            'ftol': cfg.tolerance,
        }
    )
    
    if not result.success:
        print(f"Warning: Optimization may not have converged. Status: {result.message}")
    
    # Reshape result back to matrix
    if cfg.model == "linear3x3":
        M = result.x.reshape(3, 3)
    else:  # affine3x4
        M = result.x.reshape(4, 3)
    
    return M


def solve_ccm(measured_rgb: np.ndarray, reference_rgb: np.ndarray, cfg: CCMSolveConfig) -> np.ndarray:
    assert measured_rgb.shape == reference_rgb.shape == (24, 3)
    
    if cfg.preserve_white and not cfg.use_gradient_optimization:
        return solve_ccm_with_white_constraint(measured_rgb, reference_rgb, cfg)
    
    if cfg.use_gradient_optimization:
        return solve_ccm_gradient_optimization(measured_rgb, reference_rgb, cfg)
    
    X = measured_rgb
    Y = reference_rgb

    if cfg.model == "linear3x3":
        A = X  # (N,3)
    elif cfg.model == "affine3x4":
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        A = np.concatenate([X, ones], axis=1)  # (N,4)
    else:
        raise ValueError("model must be 'linear3x3' or 'affine3x4'")

    # Ridge: (A^T A + lambda*I) M = A^T Y
    AtA = A.T @ A
    reg = np.eye(AtA.shape[0], dtype=A.dtype) * cfg.lambda_reg
    if cfg.model == "affine3x4" and not cfg.regularize_bias:
        reg[-1, -1] = 0.0
    AtY = A.T @ Y
    M = np.linalg.solve(AtA + reg, AtY)  # shape: (in_dim, 3)
    return M


def apply_ccm(rgb: np.ndarray, M: np.ndarray, model: str) -> np.ndarray:
    if model == "linear3x3":
        return rgb @ M
    elif model == "affine3x4":
        ones = np.ones((rgb.shape[0], 1), dtype=rgb.dtype)
        A = np.concatenate([rgb, ones], axis=1)
        return A @ M
    else:
        raise ValueError("Unknown model")


# ========================= Main Pipeline =========================

def correct_image_with_ccm(
    image_bgr: np.ndarray,
    M: np.ndarray,
    model: str,
    linearize_srgb: bool,
    wb_gains: Optional[Tuple[float, float, float]] = None,
    pre_gain: Optional[float] = None,
) -> np.ndarray:
    # Convert BGR uint8 -> RGB float
    rgb = image_bgr[..., ::-1].astype(np.float64) / 255.0
    if linearize_srgb:
        lin = srgb_to_linear(rgb)
    else:
        lin = rgb

    # Optional neutral white balance in linear domain
    if wb_gains is not None:
        gains = np.array(wb_gains, dtype=lin.dtype).reshape(1, 1, 3)
        lin = np.clip(lin * gains, 0.0, 1.0)

    # Optional global luminance normalization gain
    if pre_gain is not None:
        lin = np.clip(lin * float(pre_gain), 0.0, 1.0)

    h, w = lin.shape[:2]
    flat = lin.reshape(-1, 3)
    corrected_lin = np.clip(apply_ccm(flat, M, model), 0.0, 1.0).reshape(h, w, 3)

    if linearize_srgb:
        corrected = linear_to_srgb(corrected_lin)
    else:
        corrected = corrected_lin

    corrected_bgr = (np.clip(corrected, 0.0, 1.0) * 255.0).round().astype(np.uint8)[..., ::-1]
    return corrected_bgr


def apply_white_balance_only(
    image_bgr: np.ndarray,
    wb_gains: Optional[Tuple[float, float, float]] = None,
    pre_gain: Optional[float] = None,
    linearize_srgb: bool = True,
) -> np.ndarray:
    """Apply only white balance and luminance normalization, no CCM"""
    # Convert BGR uint8 -> RGB float
    rgb = image_bgr[..., ::-1].astype(np.float64) / 255.0
    if linearize_srgb:
        lin = srgb_to_linear(rgb)
    else:
        lin = rgb

    # Apply white balance
    if wb_gains is not None:
        gains = np.array(wb_gains, dtype=lin.dtype).reshape(1, 1, 3)
        lin = np.clip(lin * gains, 0.0, 1.0)

    # Apply luminance normalization
    if pre_gain is not None:
        lin = np.clip(lin * float(pre_gain), 0.0, 1.0)

    # Convert back to sRGB if needed
    if linearize_srgb:
        result = linear_to_srgb(lin)
    else:
        result = lin

    result_bgr = (np.clip(result, 0.0, 1.0) * 255.0).round().astype(np.uint8)[..., ::-1]
    return result_bgr


def compute_ccm_from_bgr(
    image_bgr: np.ndarray,
    cell_margin: float = 0.15,
    model: str = "affine3x4",
    lambda_reg: float = 0.0,
    regularize_bias: bool = False,
    measured_is_srgb: bool = True,
    reference_is_srgb: bool = True,
    wb_from_patches_19_20: bool = False,
    normalize_luma_with_patch19: bool = True,
    use_gradient_optimization: bool = False,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    preserve_white: bool = False,
) -> Tuple[np.ndarray, dict]:
    measured_rgb = sample_chart_6x4_rgb(image_bgr, cell_margin=cell_margin) / 255.0

    # Prepare reference
    ref_srgb = (SRGB_24PATCH_D65_8BIT / 255.0).astype(np.float64)

    # Decide linearization independently for measured and reference
    X = srgb_to_linear(measured_rgb) if measured_is_srgb else measured_rgb
    Y = srgb_to_linear(ref_srgb) if reference_is_srgb else ref_srgb

    wb_gains: Optional[np.ndarray] = None
    if wb_from_patches_19_20:
        # Use patches 19 and 20 (1-based indexing) => indices 18 and 19 in 0-based
        idx = [18, 19]
        neutrals = X[idx, :]  # shape (2,3)
        channel_means = neutrals.mean(axis=0)  # (3,)
        gray_target = float(channel_means.mean())
        # Gains so that channel_means * gains == gray_target for all channels
        with np.errstate(divide='ignore', invalid='ignore'):
            wb_gains = np.where(channel_means > 0, gray_target / channel_means, 1.0)
        wb_gains = np.clip(wb_gains, 0.2, 5.0)
        X = np.clip(X * wb_gains.reshape(1, 3), 0.0, 1.0)

    pre_gain_value: Optional[float] = None
    if normalize_luma_with_patch19:
        # Match luminance of patch 19 between measured and reference in linear domain
        measured_xyz_full = rgb_linear_to_xyz(np.clip(X, 0.0, 1.0))
        reference_xyz_full = rgb_linear_to_xyz(np.clip(Y, 0.0, 1.0))
        Y_meas = float(measured_xyz_full[18, 1])
        Y_ref = float(reference_xyz_full[18, 1])
        if Y_meas > 1e-8:
            g = Y_ref / Y_meas
            pre_gain_value = float(np.clip(g, 0.2, 5.0))
            X = np.clip(X * pre_gain_value, 0.0, 1.0)

    cfg = CCMSolveConfig(
        model=model, 
        lambda_reg=lambda_reg, 
        regularize_bias=regularize_bias,
        use_gradient_optimization=use_gradient_optimization,
        max_iterations=max_iterations,
        tolerance=tolerance,
        preserve_white=preserve_white
    )
    M = solve_ccm(X, Y, cfg)

    # Apply CCM
    corrected_lin = np.clip(apply_ccm(X, M, model), 0.0, 1.0)

    # DeltaE evaluation (convert linear RGB -> XYZ -> Lab)
    measured_xyz = rgb_linear_to_xyz(np.clip(X, 0.0, 1.0))
    corrected_xyz = rgb_linear_to_xyz(corrected_lin)
    reference_xyz = rgb_linear_to_xyz(np.clip(Y, 0.0, 1.0))

    measured_lab = xyz_to_lab(measured_xyz)
    corrected_lab = xyz_to_lab(corrected_xyz)
    reference_lab = xyz_to_lab(reference_xyz)

    de_before = delta_e_cie76(measured_lab, reference_lab)
    de_after = delta_e_cie76(corrected_lab, reference_lab)

    def stats(arr: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
        }

    result = {
        "image": None,
        "model": model,
        "lambda_reg": lambda_reg,
        "regularize_bias": regularize_bias,
        "measured_is_srgb": measured_is_srgb,
        "reference_is_srgb": reference_is_srgb,
        "matrix": M.T.tolist(),  # 3x3 or 3x4 as row-major by output channel
        "deltaE_before": de_before.tolist(),
        "deltaE_after": de_after.tolist(),
        "deltaE_before_stats": stats(de_before),
        "deltaE_after_stats": stats(de_after),
        "measured_rgb_srgb": (measured_rgb).tolist(),
        "corrected_rgb_srgb": (linear_to_srgb(corrected_lin)).tolist(),
        "reference_rgb_srgb": (ref_srgb).tolist(),
        "wb_from_patches_19_20": bool(wb_from_patches_19_20),
        "wb_gains": (wb_gains.tolist() if wb_gains is not None else None),
        "normalize_luma_with_patch19": bool(normalize_luma_with_patch19),
        "pre_gain": (float(pre_gain_value) if pre_gain_value is not None else None),
        "use_gradient_optimization": bool(use_gradient_optimization),
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "preserve_white": bool(preserve_white),
    }
    return M, result


def parse_roi(roi_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi_str:
        return None
    parts = roi_str.split(",")
    if len(parts) != 4:
        raise ValueError("Expected ROI as 'x,y,w,h'")
    x, y, w, h = map(int, parts)
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height must be positive")
    return x, y, w, h


def main():
    parser = argparse.ArgumentParser(description="Compute Color Correction Matrix (CCM) from a 24-patch ColorChecker image")
    parser.add_argument("--image", default=r"E:\\云存储空间\\OneDrive\\资料\\项目\\海泰\\ISP\\CCM_endoscope\\CCM_endoscope\\tools\\Colorcheck_1_raw_1004W.jpg",help="Path to the captured 24-patch chart image")
    parser.add_argument("-o", "--output", default=None, help="Path to save JSON result (default: alongside image)")
    parser.add_argument("--roi", default=None, help="ROI as 'x,y,w,h'. If not given, an interactive selector will pop up.")
    parser.add_argument("--cell_margin", type=float, default=0.15, help="Fractional margin inside each cell for sampling")
    show_grp = parser.add_mutually_exclusive_group()
    show_grp.add_argument("--show_patches", dest="show_patches", action="store_true", help="Show overlay and montage of the 24 sampled regions before solving")
    show_grp.add_argument("--no_show_patches", dest="show_patches", action="store_false", help="Do not show overlay/montage windows")
    parser.set_defaults(show_patches=True)
    parser.add_argument("--save_patches", action="store_true", help="Save overlay and montage images alongside the output JSON")
    parser.add_argument("--model", choices=["linear3x3", "affine3x4"], default="linear3x3", help="CCM model type")
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=0.0, help="Tikhonov regularization strength")
    parser.add_argument("--regularize_bias", action="store_true", help="Include bias term in regularization (affine3x4)")
    # Independent linearization controls
    parser.add_argument("--measured_linear", dest="measured_is_srgb", action="store_false", help="Input image patches are already linear (do NOT linearize)")
    parser.add_argument("--measured_srgb", dest="measured_is_srgb", action="store_true", help="Input image patches are sRGB-encoded (linearize to linear)")
    parser.set_defaults(measured_is_srgb=False)
    parser.add_argument("--ref_linear", dest="reference_is_srgb", action="store_false", help="Reference SRGB_24PATCH_D65_8BIT is already linear (do NOT linearize)")
    parser.add_argument("--ref_srgb", dest="reference_is_srgb", action="store_true", help="Reference SRGB_24PATCH_D65_8BIT is sRGB (linearize to linear)")
    parser.set_defaults(reference_is_srgb=True)
    wb_grp = parser.add_mutually_exclusive_group()
    wb_grp.add_argument("--wb_19_20", dest="wb_19_20", action="store_true", help="Enable neutral white balance from patches 19 and 20 (default)")
    wb_grp.add_argument("--no_wb_19_20", dest="wb_19_20", action="store_false", help="Disable white balance from patches 19 and 20")
    parser.set_defaults(wb_19_20=True)

    # Luminance normalization using patch 19 (default on)
    luma_grp = parser.add_mutually_exclusive_group()
    luma_grp.add_argument("--normalize_luma_19", dest="normalize_luma_19", action="store_true", help="Normalize luminance using patch 19 (default)")
    luma_grp.add_argument("--no_normalize_luma_19", dest="normalize_luma_19", action="store_false", help="Do not normalize luminance with patch 19")
    parser.set_defaults(normalize_luma_19=True)

    # Gradient optimization options
    parser.add_argument("--gradient_opt", action="store_true", help="Use gradient-based optimization in Lab space instead of least squares")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations for gradient optimization")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for gradient optimization convergence")

    # White preservation constraint
    parser.add_argument("--preserve_white", action="store_true", help="Ensure CCM matrix rows sum to 1 (preserve white)")

    args = parser.parse_args()

    img_bgr = imread_unicode(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    roi_spec = parse_roi(args.roi)
    if roi_spec is None:
        roi_bgr = crop_roi_interactive(img_bgr)
    else:
        x, y, w, h = roi_spec
        roi_bgr = img_bgr[y:y+h, x:x+w]

    # Optionally visualize sampled patches before solving
    if args.show_patches or args.save_patches:
        boxes = get_patch_boxes(roi_bgr, cell_margin=args.cell_margin)
        base, _ = os.path.splitext(args.image)
        save_overlay = (base + "_patch_overlay.jpg") if args.save_patches else None
        save_montage = (base + "_patch_montage.jpg") if args.save_patches else None
        visualize_patches(
            roi_bgr,
            boxes,
            save_overlay_path=save_overlay,
            save_montage_path=save_montage,
            show_windows=args.show_patches,
        )

    M, result = compute_ccm_from_bgr(
        image_bgr=roi_bgr,
        cell_margin=args.cell_margin,
        model=args.model,
        lambda_reg=args.lambda_reg,
        regularize_bias=args.regularize_bias,
        measured_is_srgb=args.measured_is_srgb,
        reference_is_srgb=args.reference_is_srgb,
        wb_from_patches_19_20=args.wb_19_20,
        normalize_luma_with_patch19=args.normalize_luma_19,
        use_gradient_optimization=args.gradient_opt,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
        preserve_white=args.preserve_white,
    )
    result["image"] = os.path.basename(args.image)

    out_path = args.output
    if out_path is None:
        base, _ = os.path.splitext(args.image)
        out_path = base + "_ccm.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved CCM results to: {out_path}")
    print("Model:", result["model"]) 
    print("Matrix (rows=R,G,B):")
    M_arr = np.array(result["matrix"])  # 3 x (3 or 4)
    for i, row in enumerate(M_arr):
        print([round(x, 6) for x in row])
    print("DeltaE before (mean/median/max):", result["deltaE_before_stats"]) 
    print("DeltaE after  (mean/median/max):", result["deltaE_after_stats"]) 

    # Save corrected full image as JPG next to JSON
    base_no_ext, _ = os.path.splitext(out_path)
    out_img_path = base_no_ext + f"_{args.model}"+".jpg"
    wb_gains = result.get("wb_gains", None)
    pre_gain = result.get("pre_gain", None)
    corrected_bgr = correct_image_with_ccm(
        img_bgr,
        np.array(M_arr).T,
        args.model,
        linearize_srgb=args.measured_is_srgb,
        wb_gains=(tuple(wb_gains) if wb_gains is not None else None),
        pre_gain=(float(pre_gain) if pre_gain is not None else None),
    )
    if imwrite_unicode(out_img_path, corrected_bgr, params=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        print(f"Saved corrected image to: {out_img_path}")
    else:
        print(f"Failed to save corrected image: {out_img_path}")

    # Save white-balanced image (before CCM) if white balance was applied
    if wb_gains is not None or pre_gain is not None:
        wb_img_path = base_no_ext + "_white_balanced.jpg"
        wb_bgr = apply_white_balance_only(
            img_bgr,
            wb_gains=(tuple(wb_gains) if wb_gains is not None else None),
            pre_gain=(float(pre_gain) if pre_gain is not None else None),
            linearize_srgb=args.measured_is_srgb,
        )
        if imwrite_unicode(wb_img_path, wb_bgr, params=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            print(f"Saved white-balanced image to: {wb_img_path}")
        else:
            print(f"Failed to save white-balanced image: {wb_img_path}")


if __name__ == "__main__":
    main()


