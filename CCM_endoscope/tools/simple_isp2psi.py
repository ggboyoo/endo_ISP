# 导入所需模块
import numpy as np

try:
    from ISP import process_single_image, load_dark_reference, load_correction_parameters, load_white_balance_parameters, load_ccm_matrix,demosaic_16bit,apply_gamma_correction_16bit,demosaic_16bit,demosaic_easy
    from ISP import apply_ccm_16bit
    from invert_ISP import inverse_gamma_correction,inverse_demosaic,inverse_ccm_correction
    from raw_reader import read_raw_image
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are available.")


Input_path = r"F:\ZJU\Picture\ccm\ccm_2\dark_24\25-09-05 103522.raw"
ccm_matrix=np.array([
    [1.7801320111582375, -0.7844420268663381, 0.004310015708100662],
    [-0.24377094860030846, 2.4432181685707977, -1.1994472199704893],
    [-0.4715762768203783, -0.7105721829898775, 2.182148459810256]
])
wb_params={
    "white_balance_gains": {
        "b_gain": 2.168214315103357,
        "g_gain": 1.0,
        "r_gain": 1.3014453071420942
    }
}
gamma = 2.2

def psnr(a: np.ndarray, b: np.ndarray, maxv: float) -> float:
    a = np.clip(a.astype(np.float64), 0.0, maxv)
    b = np.clip(b.astype(np.float64), 0.0, maxv)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10((maxv ** 2) / mse)


def main():
    raw_data =read_raw_image(Input_path).astype(np.float64)
    mask = raw_data > 100
    raw_data = raw_data * mask/5.0

    rgb_img = demosaic_easy(raw_data)
    rgb  = rgb_img.astype(np.float64)

    rgb_ccm1 = np.dot(rgb_img.reshape(-1, 3),ccm_matrix.T).reshape(rgb_img.shape)

    inv_ccm = np.linalg.inv(ccm_matrix)
    rgb_ccm2 = np.dot(rgb_ccm1.reshape(-1, 3),inv_ccm.T).reshape(rgb_img.shape)

    rgb_img = apply_ccm_16bit(rgb_img,ccm_matrix,'linear3x3')



    srgb_img = apply_gamma_correction_16bit(rgb_img,gamma)

    inver_rgb = inverse_gamma_correction(srgb_img,gamma)

    inver_rgb = inverse_ccm_correction(rgb_img,ccm_matrix,'linear3x3')


    re_raw = inverse_demosaic(inver_rgb)

    re_rgb = demosaic_easy(re_raw)

    re_srgb = apply_gamma_correction_16bit(re_rgb,gamma)


    print(f"raw_psnr:{psnr(raw_data, re_raw, 4095)}")
    print(f"rgb_psnr:{psnr(rgb, inver_rgb, 4095)}")
    print(f"rgb_psnr:{psnr(rgb_img, re_rgb, 4095)}")
    print(f"srgb_psnr:{psnr(srgb_img,re_srgb,4095)}")


if __name__ == "__main__":
    exit(main())