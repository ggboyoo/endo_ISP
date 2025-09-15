import os
import argparse
from typing import Optional, Tuple

import numpy as np

try:
    import hdf5storage  # For .mat files
except Exception:  # pragma: no cover
    hdf5storage = None

try:
    import rawpy  # For .dng/.nef/etc
except Exception:  # pragma: no cover
    rawpy = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


def _infer_key_from_mat(mat_dict: dict) -> Optional[str]:
    """
    Try common keys used to store raw arrays in .mat files.
    """
    if not isinstance(mat_dict, dict):
        return None
    candidate_keys = [
        'raw', 'image', 'img', 'data', 'I', 'Y'
    ]
    for key in candidate_keys:
        if key in mat_dict and isinstance(mat_dict[key], np.ndarray):
            return key
    # fallback: first ndarray
    for key, value in mat_dict.items():
        if isinstance(value, np.ndarray):
            return key
    return None


def load_raw_array(input_path: str) -> Tuple[np.ndarray, str]:
    """
    Load a RAW-like array from various formats.

    Supported formats:
    - .npy: NumPy array
    - .mat: MATLAB file (requires hdf5storage)
    - .dng/.nef/.cr2/...: via rawpy if available (returns debayered raw as float)
    - .tiff/.tif/.png/.bmp/.jpg: via imageio as a generic fallback

    Returns (array, hint_format)
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.npy':
        arr = np.load(input_path)
        return arr, 'npy'
    if ext == '.mat':
        if hdf5storage is None:
            raise RuntimeError('hdf5storage not available to read .mat files')
        content = hdf5storage.loadmat(input_path)
        key = _infer_key_from_mat(content)
        if key is None:
            raise ValueError('No ndarray found in .mat file')
        arr = content[key]
        return arr, 'mat'
    if ext in {'.dng', '.nef', '.cr2', '.arw', '.raw', '.orf', '.rw2'}:
        if rawpy is None:
            raise RuntimeError('rawpy not available to read RAW files')
        with rawpy.imread(input_path) as raw:
            # Use the raw image (postprocess to linear, no gamma) to approximate sensor space
            rgb = raw.postprocess(output_bps=16, gamma=(1, 1), no_auto_bright=True, use_camera_wb=True)
            arr = rgb.astype(np.float32)
        return arr, 'rawpy'
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
    Simulate shot noise using a Poisson process.

    Process:
    1) Convert RAW counts to photon energy by dividing by gain.
    2) Sample Poisson with lambda equal to energy per pixel.
    3) Optionally multiply by gain to return to RAW scale.

    Args:
        raw_data: Input RAW data as ndarray (can be any numeric dtype).
        gain: Gain used to convert photons to RAW counts.
        seed: RNG seed for reproducibility.
        restore_gain: If True, multiply sampled photons by gain to return to RAW scale.
        clip_min: If provided, clip input before noise simulation at this minimum.
        clip_max: If provided, clip input before noise simulation at this maximum.

    Returns:
        noisy array as float32.
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
    # Poisson expects non-negative lambdas; cast to float64 for stability then back to float32
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
    ext = os.path.splitext(out_path)[1].lower()
    if dtype:
        if dtype.lower() in {'uint8', 'u8'}:
            arr_to_save = np.clip(arr, 0, 255).astype(np.uint8)
        elif dtype.lower() in {'uint16', 'u16'}:
            arr_to_save = np.clip(arr, 0, 65535).astype(np.uint16)
        elif dtype.lower() in {'float32', 'f32'}:
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
    parser = argparse.ArgumentParser(description='Simulate Poisson shot noise on RAW data')
    parser.add_argument('--in', dest='input_path', type=str, required=True, help='Path to input RAW file (.npy/.mat/.dng/...)')
    parser.add_argument('--out', dest='output_path', type=str, required=True, help='Path to save noisy output (.npy/.png/.tif/...)')
    parser.add_argument('--gain', type=float, required=True, help='Gain used to convert photons to RAW counts')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--restore-gain', action='store_true', help='Multiply sampled photons by gain before saving (default True)')
    parser.add_argument('--no-restore-gain', dest='restore_gain', action='store_false')
    parser.set_defaults(restore_gain=True)
    parser.add_argument('--clip-min', type=float, default=None, help='Clip minimum before noise simulation')
    parser.add_argument('--clip-max', type=float, default=None, help='Clip maximum before noise simulation')
    parser.add_argument('--save-dtype', type=str, default=None, help='Optional dtype for saving: uint8|uint16|float32')

    args = parser.parse_args()

    raw, _ = load_raw_array(args.input_path)
    noisy = simulate_poisson_noise(
        raw_data=raw,
        gain=args.gain,
        seed=args.seed,
        restore_gain=args.restore_gain,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    save_array(noisy, args.output_path, dtype=args.save_dtype)


if __name__ == '__main__':
    main()


