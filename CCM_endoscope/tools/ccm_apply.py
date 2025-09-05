import argparse
import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
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


def apply_ccm(flat_rgb: np.ndarray, M: np.ndarray, model: str) -> np.ndarray:
    if model == "linear3x3":
        return flat_rgb @ M
    elif model == "affine3x4":
        ones = np.ones((flat_rgb.shape[0], 1), dtype=flat_rgb.dtype)
        A = np.concatenate([flat_rgb, ones], axis=1)
        return A @ M
    else:
        raise ValueError("model must be 'linear3x3' or 'affine3x4'")


def correct_image_with_ccm(image_bgr: np.ndarray, M: np.ndarray, model: str, linearize_srgb: bool) -> np.ndarray:
    rgb = image_bgr[..., ::-1].astype(np.float64) / 255.0
    if linearize_srgb:
        lin = srgb_to_linear(rgb)
    else:
        lin = rgb
    h, w = lin.shape[:2]
    flat = lin.reshape(-1, 3)
    corrected_lin = np.clip(apply_ccm(flat, M, model), 0.0, 1.0).reshape(h, w, 3)
    if linearize_srgb:
        corrected = linear_to_srgb(corrected_lin)
    else:
        corrected = corrected_lin
    corrected_bgr = (np.clip(corrected, 0.0, 1.0) * 255.0).round().astype(np.uint8)[..., ::-1]
    return corrected_bgr


def load_matrix_from_json(json_path: str) -> Tuple[np.ndarray, str, Optional[bool]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matrix = np.array(data["matrix"], dtype=np.float64)  # 3 x (3 or 4)
    model = data.get("model", "affine3x4")
    linearize_srgb = data.get("linearize_srgb", None)
    # Convert to (in_dim x 3) for multiplication
    M = matrix.T
    return M, model, linearize_srgb


def parse_matrix_csv(matrix_csv: str) -> np.ndarray:
    # Format: "rrow;rrow;rrow" where each row is comma-separated numbers
    rows = [r.strip() for r in matrix_csv.split(";") if r.strip()]
    if len(rows) != 3:
        raise ValueError("matrix_csv must contain 3 rows separated by ';'")
    mat = []
    for r in rows:
        nums = [float(x) for x in r.split(",")]
        mat.append(nums)
    matrix = np.array(mat, dtype=np.float64)  # 3 x (3 or 4)
    return matrix.T  # return (in_dim x 3)


def main():
    parser = argparse.ArgumentParser(description="Apply a Color Correction Matrix (CCM) to an image")
    parser.add_argument("--image", required=True, help="Path to the image to correct")
    parser.add_argument("--ccm_json", default=None, help="Path to CCM JSON produced by ccm_calculator.py")
    parser.add_argument("--matrix_csv", default=None, help="Matrix rows as 'r1;r2;r3', each row comma-separated. For affine3x4, provide 4 values per row.")
    parser.add_argument("--model", choices=["linear3x3", "affine3x4"], default=None, help="Model type. If omitted, will read from JSON (if provided)")
    parser.add_argument("--no_linearize", action="store_true", help="Disable sRGB linearization before applying CCM")
    parser.add_argument("-o", "--output", default=None, help="Output image path (default: alongside input with _apply.jpg)")

    args = parser.parse_args()

    if args.ccm_json is None and args.matrix_csv is None:
        raise ValueError("Provide either --ccm_json or --matrix_csv")

    if args.ccm_json is not None:
        M, model_json, linearize_json = load_matrix_from_json(args.ccm_json)
        model = args.model if args.model is not None else model_json
        if model not in ("linear3x3", "affine3x4"):
            raise ValueError("Invalid model in JSON or args")
        linearize = (linearize_json if linearize_json is not None else True) and (not args.no_linearize)
    else:
        M = parse_matrix_csv(args.matrix_csv)
        model = args.model or ("affine3x4" if M.shape[0] == 4 else "linear3x3")
        linearize = not args.no_linearize

    img_bgr = imread_unicode(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    corrected_bgr = correct_image_with_ccm(img_bgr, M, model, linearize)

    out_path = args.output
    if out_path is None:
        base, _ = os.path.splitext(args.image)
        out_path = base + "_apply.jpg"

    if imwrite_unicode(out_path, corrected_bgr, params=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        print(f"Saved corrected image to: {out_path}")
    else:
        raise RuntimeError(f"Failed to save corrected image: {out_path}")


if __name__ == "__main__":
    main()


