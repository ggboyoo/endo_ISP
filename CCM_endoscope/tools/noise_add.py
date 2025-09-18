import os
from re import T
from typing import Optional, Tuple

import numpy as np

try:
    import imageio.v2 as imageio  # Fallback for non-RAW images
except Exception:  # pragma: no cover
    imageio = None

# Prefer the project's RAW reader and demosaic for .raw files
try:
    from raw_reader import read_raw_image as project_read_raw_image
    from raw_reader import demosaic_image_corrected_fixed as project_demosaic_fixed
except Exception:  # pragma: no cover
    project_read_raw_image = None
    project_demosaic_fixed = None


# =========================
# Global configuration
# =========================
# Required
INPUT_PATH: str = r"F:\ZJU\Picture\invert_isp\inverted_output.raw"  # e.g. r"E:\\path\\to\\input.raw" or .npy/.png
g:int = 8
GAIN: float = 0.3754*g+0.7559  # must be > 0
OUTPUT_PATH: str = rf"F:\ZJU\Picture\noise_add\g{g}dark_shading_hot.png"  # e.g. r"E:\\path\\to\\output.png"


# Optional
SEED: Optional[int] = None  # e.g. 123 or None
RESTORE_GAIN: bool = True  # multiply sampled photons by gain
CLIP_MIN: Optional[float] = 0  # e.g. 0 or None
CLIP_MAX: Optional[float] = 4095  # e.g. 4095 or None
SAVE_DTYPE: Optional[str] = None  # one of: uint8|uint16|float32 or None

# RAW-specific (only used when INPUT_PATH endswith .raw)
RAW_WIDTH: Optional[int] = None   # required for .raw
RAW_HEIGHT: Optional[int] = None  # required for .raw
RAW_DTYPE: str = 'uint16'         # uint8|uint16|uint32|float32|float64
PRESET_RESOLUTION: Optional[str] = "1k"  # '1k' or '4k' or None

# Preset resolution map (width, height)
_RES_PRESETS = {
    '1k': (1920, 1080),   # adjust if your 1K is different
    '4k': (3840, 2160),
}

# ISP control
RUN_ISP: bool = True  # if True and input is .raw, run ISP and save PNG
ISP_BAYER_PATTERN: str = 'rggb'  # used by demosaic/simple path
FULL_ISP: bool = True  # use full ISP.process_raw_array instead of simple demosaic
GAMMA_VALUE: float = 2.2
USE_WHITE_BALANCE: bool = True  # enable and set WB_PARAMS_PATH to load from ISP
WB_PARAMS_PATH: Optional[str] = r"F:\ZJU\Picture\wb\wb_output"  # file or directory containing wb json
USE_CCM: bool = True  # enable and set CCM_MATRIX_PATH to load from ISP
CCM_MATRIX_PATH: Optional[str] = r"F:\ZJU\Picture\ccm\ccm_2\ccm_output_20250905_162714"  # file or directory containing ccm json

# Additive dark noise (optional)
ADDITIVE_NOISE_ENABLED: bool = True
ADDITIVE_DARK_DIR: Optional[str] = f"F:\ZJU\Picture\dark\g{g}"  # directory containing dark raw frames
ADDITIVE_DARK_WIDTH: int = 3840
ADDITIVE_DARK_HEIGHT: int = 2160
ADDITIVE_SCALE: float = 1.0  # scale factor for additive dark
ADDITIVE_ZERO_MEAN: bool = False  # subtract mean of the selected patch before adding

# Subtract dark average (dark level and fixed pattern removal)
SUBTRACT_DARK_AVERAGE_ENABLED: bool = True
DARK_AVERAGE_FILENAME: Optional[str] = None  # if None, try 'dark_average.raw' then 'average.raw'


def load_input_array(
    input_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    data_type: str = 'uint16',
) -> Tuple[np.ndarray, str]:
    """
    Load input array.

    - .raw: uses project `raw_reader.read_raw_image` and requires width/height and data_type
    - .npy: NumPy array
    - images (.png/.tif/...): via imageio as a fallback
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.raw':
        if project_read_raw_image is None:
            raise RuntimeError('raw_reader.read_raw_image not available for .raw files')
        if width is None or height is None:
            raise ValueError('width and height are required for .raw input')
        arr = project_read_raw_image(input_path, col=int(width), row=int(height), data_type=data_type)
        return arr, 'raw'
    if ext == '.npy':
        arr = np.load(input_path)
        return arr, 'npy'
    # Generic image fallback
    if imageio is None:
        raise RuntimeError('imageio not available to read image files')
    arr = imageio.imread(input_path)
    return arr, 'image'


def simulate_poisson_noise(
    raw_data: np.ndarray,
    gain: float,
    seed: Optional[int] = None,
    restore_gain: bool = True,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> np.ndarray:
    """
    Simulate shot noise (Poisson) following the user's model:
    1) Convert RAW counts to photon energy by dividing by gain.
    2) Sample Poisson with lambda equal to energy per pixel.
    3) Optionally multiply by gain to return to RAW scale.

    Returns float32 array.
    """
    if gain <= 0:
        raise ValueError('gain must be > 0')

    data = raw_data.astype(np.float32)
    if clip_min is not None or clip_max is not None:
        low = -np.inf if clip_min is None else clip_min
        high = np.inf if clip_max is None else clip_max
        data = np.clip(data, low, high)

    photons = data / float(gain)
    photons = np.maximum(photons, 0.0)

    rng = np.random.default_rng(seed)
    noisy_photons = rng.poisson(lam=photons.astype(np.float64)).astype(np.float32)

    if restore_gain:
        noisy = noisy_photons * float(gain)
    else:
        noisy = noisy_photons

    return noisy


def save_array(arr: np.ndarray, out_path: str, dtype: Optional[str] = None) -> None:
    """
    Save array to out_path with an optional dtype conversion.

    - .npy: saves raw array
    - image extensions: saves with imageio, casting per dtype if provided
    """
    # Ensure parent directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(out_path)[1].lower()
    if dtype:
        dt = dtype.lower()
        if dt in {'uint8', 'u8'}:
            arr_to_save = np.clip(arr, 0, 255).astype(np.uint8)
        elif dt in {'uint16', 'u16'}:
            arr_to_save = np.clip(arr, 0, 4095).astype(np.uint16)
        elif dt in {'float32', 'f32'}:
            arr_to_save = arr.astype(np.float32)
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
    else:
        arr_to_save = arr

    if ext == '.npy':
        np.save(out_path, arr_to_save)
        return

    if imageio is None:
        raise RuntimeError('imageio not available to write images')

    # Convert BGR to RGB for 3-channel images before saving
    if isinstance(arr_to_save, np.ndarray) and arr_to_save.ndim == 3 and arr_to_save.shape[2] == 3:
        arr_to_save = arr_to_save[:, :, ::-1]

    imageio.imwrite(out_path, arr_to_save)


def _list_raw_files(directory: str) -> list:
    try:
        from pathlib import Path
        p = Path(directory)
        return sorted([f for f in p.glob('*.raw') if f.is_file()])
    except Exception:
        return []


def _select_random_even(value_max: int, rng: np.random.Generator) -> int:
    if value_max <= 0:
        return 0
    # pick an even start in [0, value_max]
    max_even = value_max - (value_max % 2)
    # number of even positions
    num = max_even // 2 + 1
    k = int(rng.integers(0, num))
    return int(2 * k)


def _fix_hot_pixels_nearest(src: np.ndarray, threshold: float = 200.0) -> np.ndarray:
    """
    Detect hot pixels (> threshold) and replace each with the nearest non-hot neighbor value.

    Uses a small expanding neighborhood search with Manhattan distance priority.
    """
    img = src.copy()
    if img.ndim != 2:
        return img
    h, w = img.shape
    hot_mask = img.astype(np.float32) > float(threshold)
    if not np.any(hot_mask):
        return img

    hot_coords = np.argwhere(hot_mask)

    # Search offsets ordered by distance
    offsets = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (0, 2), (0, -2), (2, 0), (-2, 0),
        (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2),
    ]

    src_f = src.astype(np.float32)
    thr = float(threshold)
    for y, x in hot_coords:
        replaced = False
        for dy, dx in offsets:
            ny = y + dy
            nx = x + dx
            if 0 <= ny < h and 0 <= nx < w:
                val = src_f[ny, nx]
                if val <= thr:
                    img[y, x] = val.astype(img.dtype) if hasattr(val, 'astype') else val
                    replaced = True
                    break
        if not replaced:
            # Fallback to threshold cap
            img[y, x] = src.dtype.type(threshold) if hasattr(src.dtype, 'type') else threshold
    return img


def _fix_hot_pixels_adaptive(src: np.ndarray, window_size: int = 5, k_std: float = 5.0, abs_min: float = 200.0) -> np.ndarray:
    """
    Adaptive hot-pixel detection: mark pixel as hot if
        value > local_mean + k_std * local_std and value > abs_min.
    Replace with nearest non-hot neighbor (same strategy as _fix_hot_pixels_nearest).
    """
    if src.ndim != 2:
        return src
    if window_size < 3 or window_size % 2 == 0:
        window_size = 5

    try:
        from scipy.ndimage import uniform_filter
        src_f = src.astype(np.float32)
        mean = uniform_filter(src_f, size=window_size, mode='nearest')
        sq_mean = uniform_filter(src_f * src_f, size=window_size, mode='nearest')
        var = np.maximum(sq_mean - mean * mean, 0.0)
        std = np.sqrt(var)
        thresh = mean + float(k_std) * std
        hot_mask = (src_f > thresh) & (src_f > float(abs_min))
        if not np.any(hot_mask):
            return src
        # Use nearest replacement on only hot pixels
        fixed = src.copy()
        h, w = src.shape
        offsets = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (0, 2), (0, -2), (2, 0), (-2, 0),
            (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2),
        ]
        for y, x in np.argwhere(hot_mask):
            replaced = False
            for dy, dx in offsets:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if not hot_mask[ny, nx]:
                        fixed[y, x] = src[ny, nx]
                        replaced = True
                        break
            if not replaced:
                fixed[y, x] = src.dtype.type(abs_min) if hasattr(src.dtype, 'type') else abs_min
        return fixed
    except Exception:
        # Fallback to fixed threshold method
        return _fix_hot_pixels_nearest(src, threshold=abs_min)


def add_additive_dark_noise(
    noisy: np.ndarray,
    bayer_pattern: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add an additive dark frame patch aligned to the Bayer mosaic.

    - Loads a random .raw from ADDITIVE_DARK_DIR with size ADDITIVE_DARK_WIDTH x ADDITIVE_DARK_HEIGHT
    - If sizes differ, crops a patch matching `noisy` shape
    - Ensures patch starts at an R site for RGGB-like mosaics (even row, even col for 'rggb')
    - Optionally zero-mean the patch, then scales and adds
    """
    if not ADDITIVE_NOISE_ENABLED or not ADDITIVE_DARK_DIR:
        return noisy

    if project_read_raw_image is None:
        return noisy

    files = _list_raw_files(ADDITIVE_DARK_DIR)
    if not files:
        return noisy

    h, w = noisy.shape[:2]
    # choose random dark file
    idx = int(rng.integers(0, len(files)))
    dark_path = str(files[idx])

    try:
        dark_full = project_read_raw_image(dark_path, col=int(ADDITIVE_DARK_WIDTH), row=int(ADDITIVE_DARK_HEIGHT), data_type='uint16')
    except Exception:
        return noisy

    if dark_full is None:
        return noisy

    # Hot pixel fix on the selected dark frame (adaptive if available)
    try:
        dark_full = _fix_hot_pixels_adaptive(dark_full, window_size=5, k_std=5.0, abs_min=200.0)
    except Exception:
        dark_full = _fix_hot_pixels_nearest(dark_full, threshold=200.0)

    dh, dw = dark_full.shape[:2]
    if dh < h or dw < w:
        # cannot crop larger than source; fallback: tile crop center
        y0 = max(0, (dh - h) // 2)
        x0 = max(0, (dw - w) // 2)
        # enforce even alignment for R at top-left when 'rggb'
        if bayer_pattern.lower() == 'rggb':
            y0 -= (y0 % 2)
            x0 -= (x0 % 2)
        patch = dark_full[y0:y0 + h, x0:x0 + w]
    elif dh == h and dw == w:
        # same size
        patch = dark_full
    else:
        # random crop with Bayer alignment
        max_y = dh - h
        max_x = dw - w
        if bayer_pattern.lower() == 'rggb':
            y0 = _select_random_even(max_y, rng)
            x0 = _select_random_even(max_x, rng)
        else:
            # generic even alignment to preserve 2x2 blocks
            y0 = _select_random_even(max_y, rng)
            x0 = _select_random_even(max_x, rng)
        patch = dark_full[y0:y0 + h, x0:x0 + w]

    patch_f = patch.astype(np.float32)
    if ADDITIVE_ZERO_MEAN:
        patch_f = patch_f - float(np.mean(patch_f))
    if ADDITIVE_SCALE != 1.0:
        patch_f = patch_f * float(ADDITIVE_SCALE)

    out = noisy.astype(np.float32) + patch_f
    return out


def subtract_dark_average(
    image: np.ndarray,
    bayer_pattern: str,
) -> np.ndarray:
    """
    Subtract a Bayer-aligned dark average from the image to remove fixed pattern noise.

    - Looks for DARK_AVERAGE_FILENAME in ADDITIVE_DARK_DIR. If None, tries
      'dark_average.raw' then 'average.raw'.
    - Aligns by cropping the dark average with the same Bayer-aligned rule as additive noise.
    """
    if not SUBTRACT_DARK_AVERAGE_ENABLED or not ADDITIVE_DARK_DIR:
        return image

    if project_read_raw_image is None:
        return image

    # Resolve path
    from pathlib import Path
    base = Path(ADDITIVE_DARK_DIR)
    candidates = []
    if DARK_AVERAGE_FILENAME:
        candidates.append(base / DARK_AVERAGE_FILENAME)
    candidates.append(base / 'average_dark.raw')
    candidates.append(base / 'average.raw')

    dark_avg_full = None
    for p in candidates:
        try:
            if p.exists():
                arr = project_read_raw_image(str(p), col=int(ADDITIVE_DARK_WIDTH), row=int(ADDITIVE_DARK_HEIGHT), data_type='uint16')
                if arr is not None:
                    # Hot pixel fix on average dark as well (adaptive)
                    try:
                        dark_avg_full = _fix_hot_pixels_adaptive(arr, window_size=5, k_std=5.0, abs_min=200.0)
                    except Exception:
                        dark_avg_full = _fix_hot_pixels_nearest(arr, threshold=200.0)
                    break
        except Exception:
            continue

    if dark_avg_full is None:
        return image

    h, w = image.shape[:2]
    dh, dw = dark_avg_full.shape[:2]

    # Bayer-aligned crop similar to additive path
    if dh < h or dw < w:
        y0 = max(0, (dh - h) // 2)
        x0 = max(0, (dw - w) // 2)
        if bayer_pattern.lower() == 'rggb':
            y0 -= (y0 % 2)
            x0 -= (x0 % 2)
        dark_avg = dark_avg_full[y0:y0 + h, x0:x0 + w]
    elif dh == h and dw == w:
        dark_avg = dark_avg_full
    else:
        # align to even grid
        max_y = dh - h
        max_x = dw - w
        y0 = max_y - (max_y % 2)
        x0 = max_x - (max_x % 2)
        dark_avg = dark_avg_full[y0:y0 + h, x0:x0 + w]

    out = image.astype(np.float32) - dark_avg.astype(np.float32)
    return out


def main() -> None:
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    gain = GAIN
    seed = SEED
    restore_gain = RESTORE_GAIN
    clip_min = CLIP_MIN
    clip_max = CLIP_MAX
    save_dtype = SAVE_DTYPE

    if not input_path:
        raise ValueError('INPUT_PATH is required')
    if not output_path:
        raise ValueError('OUTPUT_PATH is required')
    if gain is None or gain <= 0:
        raise ValueError('GAIN must be > 0')

    ext = os.path.splitext(input_path)[1].lower()
    width = RAW_WIDTH
    height = RAW_HEIGHT
    raw_dtype = RAW_DTYPE
    if ext == '.raw':
        if (width is None or height is None) and PRESET_RESOLUTION:
            preset_key = PRESET_RESOLUTION.lower()
            if preset_key in _RES_PRESETS:
                width, height = _RES_PRESETS[preset_key]
        if width is None or height is None:
            raise ValueError('RAW_WIDTH and RAW_HEIGHT are required for .raw input (or set PRESET_RESOLUTION to \"1k\" or \"4k\")')

    raw, _ = load_input_array(
        input_path=input_path,
        width=width,
        height=height,
        data_type=raw_dtype,
    )

    noisy = simulate_poisson_noise(
        raw_data=raw,
        gain=gain,
        seed=seed,
        restore_gain=restore_gain,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    # Additive dark noise after Poisson noise
    if ADDITIVE_NOISE_ENABLED:
        rng = np.random.default_rng(seed)
        noisy = add_additive_dark_noise(
            noisy=noisy,
            bayer_pattern=ISP_BAYER_PATTERN,
            rng=rng,
        )


    # Subtract dark average to remove fixed pattern and hot pixels
    if SUBTRACT_DARK_AVERAGE_ENABLED:
        noisy = subtract_dark_average(
            image=noisy,
            bayer_pattern=ISP_BAYER_PATTERN,
        )
        
        noisy = np.clip(noisy, 0, 4095)

    # If requested, pass through ISP and save PNG
    if RUN_ISP and ext == '.raw':
        out_png = output_path if output_path.lower().endswith('.png') else (output_path + '.png')
        if FULL_ISP:
            # Use full ISP without writing temp file (direct ndarray entrypoint)
            try:
                from ISP import process_raw_array, load_white_balance_parameters, load_ccm_matrix
            except Exception as e:
                raise RuntimeError(f'Cannot import ISP helpers: {e}')

            dark_data = np.zeros((height, width), dtype=np.uint16)
            lens_shading_params = {}

            # Ensure dtype compatibility
            raw_for_isp = np.clip(noisy, 0, 4095).astype(np.uint16)

            # Optionally load WB/CCM
            wb_params = None
            white_balance_enabled = False
            if USE_WHITE_BALANCE and WB_PARAMS_PATH:
                try:
                    wb_params = load_white_balance_parameters(WB_PARAMS_PATH)
                    white_balance_enabled = wb_params is not None
                except Exception:
                    white_balance_enabled = False

            ccm_matrix = None
            ccm_enabled = False
            ccm_matrix_path = None
            if USE_CCM and CCM_MATRIX_PATH:
                try:
                    ccm_loaded = load_ccm_matrix(CCM_MATRIX_PATH)
                    if ccm_loaded is not None:
                        ccm_matrix, _ = ccm_loaded
                        ccm_enabled = True
                except Exception:
                    ccm_enabled = False

            result = process_raw_array(
                raw_data=raw_for_isp,
                dark_data=dark_data,
                lens_shading_params=lens_shading_params,
                width=width,
                height=height,
                data_type='uint16',
                wb_params=wb_params,
                dark_subtraction_enabled=False,
                lens_shading_enabled=False,
                white_balance_enabled=white_balance_enabled,
                ccm_enabled=ccm_enabled,
                ccm_matrix_path=ccm_matrix_path,
                ccm_matrix=ccm_matrix,
                gamma_correction_enabled=True,
                gamma_value=GAMMA_VALUE,
                demosaic_output=True,
            )
            if not result.get('processing_success', False):
                raise RuntimeError(f"Full ISP processing failed: {result.get('error', 'unknown error')}")
            color_img = result.get('color_img')
            if color_img is None:
                raise RuntimeError('Full ISP returned no color image')
            save_array(color_img, out_png, dtype=None)
        else:
            if project_demosaic_fixed is None:
                raise RuntimeError('Cannot run simple ISP: demosaic function not available')
            # Ensure uint16 for demosaicing
            noisy_u16 = np.clip(noisy, 0, 4095).astype(np.uint16)   
            color16 = project_demosaic_fixed(noisy_u16, ISP_BAYER_PATTERN)
            max_val = int(np.max(color16))
            if max_val > 0:
                color8 = (color16.astype(np.float32) / float(max_val) * 255.0).astype(np.uint8)
            else:
                color8 = np.zeros_like(color16, dtype=np.uint8)
            save_array(color8, out_png, dtype=None)
    else:
        # Save noisy data directly
        save_array(noisy, output_path, dtype=save_dtype)


if __name__ == '__main__':
    main()


