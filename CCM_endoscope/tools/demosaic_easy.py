#!/usr/bin/env python3
"""
最简单的去马赛克函数
使用2x2块的方式，4个像素共享相同的RGB值
"""

import numpy as np
import cv2
from typing import Optional

def demosaic_easy(raw_data: np.ndarray, bayer_pattern: str = 'rggb') -> Optional[np.ndarray]:
    """
    最简单的去马赛克函数
    使用2x2块的方式，4个像素共享相同的RGB值
    
    Args:
        raw_data: RAW数据数组 (H, W)
        bayer_pattern: Bayer模式 ('rggb', 'bggr', 'grbg', 'gbrg')
        
    Returns:
        去马赛克后的彩色图像 (H, W, 3)，失败时返回None
    """
    print(f"Applying easy demosaicing for {bayer_pattern} pattern...")
    
    try:
        height, width = raw_data.shape
        print(f"  Input: {raw_data.shape}, dtype: {raw_data.dtype}")
        
        # 确保尺寸是偶数（2x2块需要）
        if height % 2 != 0 or width % 2 != 0:
            print(f"  Warning: Image size ({height}x{width}) is not even, cropping to fit 2x2 blocks")
            height = (height // 2) * 2
            width = (width // 2) * 2
            raw_data = raw_data[:height, :width]
            print(f"  Cropped to: {raw_data.shape}")
        
        # 创建输出图像
        color_image = np.zeros((height, width, 3), dtype=raw_data.dtype)
        
        if bayer_pattern == 'rggb':
            # RGGB模式
            # 2x2块布局:
            # R G
            # G B
            for y in range(0, height, 2):
                for x in range(0, width, 2):
                    # 获取2x2块的值
                    r = raw_data[y, x]           # 左上角 R
                    g1 = raw_data[y, x+1]        # 右上角 G
                    g2 = raw_data[y+1, x]        # 左下角 G
                    b = raw_data[y+1, x+1]       # 右下角 B
                    
                    # 计算G的平均值
                    g_avg = (g1 + g2) // 2
                    
                    # 为2x2块的所有像素设置相同的RGB值
                    # 左上角 (y, x)
                    color_image[y, x, 0] = b     # B通道
                    color_image[y, x, 1] = g_avg # G通道
                    color_image[y, x, 2] = r     # R通道
                    
                    # 右上角 (y, x+1)
                    color_image[y, x+1, 0] = b
                    color_image[y, x+1, 1] = g_avg
                    color_image[y, x+1, 2] = r
                    
                    # 左下角 (y+1, x)
                    color_image[y+1, x, 0] = b
                    color_image[y+1, x, 1] = g_avg
                    color_image[y+1, x, 2] = r
                    
                    # 右下角 (y+1, x+1)
                    color_image[y+1, x+1, 0] = b
                    color_image[y+1, x+1, 1] = g_avg
                    color_image[y+1, x+1, 2] = r
        
        elif bayer_pattern == 'bggr':
            # BGGR模式
            # 2x2块布局:
            # B G
            # G R
            for y in range(0, height, 2):
                for x in range(0, width, 2):
                    b = raw_data[y, x]           # 左上角 B
                    g1 = raw_data[y, x+1]        # 右上角 G
                    g2 = raw_data[y+1, x]        # 左下角 G
                    r = raw_data[y+1, x+1]       # 右下角 R
                    
                    g_avg = (g1 + g2) // 2
                    
                    # 为2x2块的所有像素设置相同的RGB值
                    for dy in range(2):
                        for dx in range(2):
                            color_image[y+dy, x+dx, 0] = b     # B通道
                            color_image[y+dy, x+dx, 1] = g_avg # G通道
                            color_image[y+dy, x+dx, 2] = r     # R通道
        
        elif bayer_pattern == 'grbg':
            # GRBG模式
            # 2x2块布局:
            # G R
            # B G
            for y in range(0, height, 2):
                for x in range(0, width, 2):
                    g1 = raw_data[y, x]          # 左上角 G
                    r = raw_data[y, x+1]         # 右上角 R
                    b = raw_data[y+1, x]         # 左下角 B
                    g2 = raw_data[y+1, x+1]      # 右下角 G
                    
                    g_avg = (g1 + g2) // 2
                    
                    # 为2x2块的所有像素设置相同的RGB值
                    for dy in range(2):
                        for dx in range(2):
                            color_image[y+dy, x+dx, 0] = b     # B通道
                            color_image[y+dy, x+dx, 1] = g_avg # G通道
                            color_image[y+dy, x+dx, 2] = r     # R通道
        
        elif bayer_pattern == 'gbrg':
            # GBRG模式
            # 2x2块布局:
            # G B
            # R G
            for y in range(0, height, 2):
                for x in range(0, width, 2):
                    g1 = raw_data[y, x]          # 左上角 G
                    b = raw_data[y, x+1]         # 右上角 B
                    r = raw_data[y+1, x]         # 左下角 R
                    g2 = raw_data[y+1, x+1]      # 右下角 G
                    
                    g_avg = (g1 + g2) // 2
                    
                    # 为2x2块的所有像素设置相同的RGB值
                    for dy in range(2):
                        for dx in range(2):
                            color_image[y+dy, x+dx, 0] = b     # B通道
                            color_image[y+dy, x+dx, 1] = g_avg # G通道
                            color_image[y+dy, x+dx, 2] = r     # R通道
        
        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
        
        print(f"  Demosaicing completed: {color_image.shape}, range: {np.min(color_image)}-{np.max(color_image)}")
        return color_image
        
    except Exception as e:
        print(f"  Error in easy demosaicing: {e}")
        return None

def test_demosaic_easy():
    """测试简单去马赛克函数"""
    print("Testing Easy Demosaicing")
    print("=" * 40)
    
    # 创建测试数据
    height, width = 8, 8
    raw_data = np.zeros((height, width), dtype=np.uint16)
    
    # 填充测试数据 (RGGB模式)
    # R G R G
    # G B G B
    # R G R G
    # G B G B
    raw_data[0, 0] = 1000  # R
    raw_data[0, 1] = 2000  # G
    raw_data[0, 2] = 1000  # R
    raw_data[0, 3] = 2000  # G
    raw_data[1, 0] = 2000  # G
    raw_data[1, 1] = 3000  # B
    raw_data[1, 2] = 2000  # G
    raw_data[1, 3] = 3000  # B
    raw_data[2, 0] = 1000  # R
    raw_data[2, 1] = 2000  # G
    raw_data[2, 2] = 1000  # R
    raw_data[2, 3] = 2000  # G
    raw_data[3, 0] = 2000  # G
    raw_data[3, 1] = 3000  # B
    raw_data[3, 2] = 2000  # G
    raw_data[3, 3] = 3000  # B
    
    print(f"Test RAW data:")
    print(raw_data)
    
    # 测试去马赛克
    result = demosaic_easy(raw_data, 'rggb')
    
    if result is not None:
        print(f"\nDemosaiced result:")
        print(f"Shape: {result.shape}")
        print(f"Range: {np.min(result)}-{np.max(result)}")
        
        # 显示第一个2x2块的结果
        print(f"\nFirst 2x2 block RGB values:")
        for y in range(2):
            for x in range(2):
                rgb = result[y, x]
                print(f"  Pixel ({y},{x}): R={rgb[2]}, G={rgb[1]}, B={rgb[0]}")
        
        # 验证所有像素的RGB值是否相同
        first_rgb = result[0, 0]
        all_same = np.all(result == first_rgb)
        print(f"\nAll pixels have same RGB values: {all_same}")
        
        if all_same:
            print(f"RGB values: R={first_rgb[2]}, G={first_rgb[1]}, B={first_rgb[0]}")
            print(f"Expected: R=1000, G=2000, B=3000")
    else:
        print("Demosaicing failed!")

if __name__ == "__main__":
    test_demosaic_easy()
