import os
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
OUTPUT_PATH: str = r"F:\ZJU\Picture\noise_add\3.png"  # e.g. r"E:\\path\\to\\output.png"
GAIN: float = 3.0  # must be > 0

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
            noisy_u16 = np.clip(noisy, 0, 65535).astype(np.uint16)
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


