#!/usr/bin/env python3
"""
增强的去马赛克模块
支持多种去马赛克算法选择
"""

import numpy as np
import cv2
from typing import Optional, Literal

# 去马赛克算法类型
DemosaicMethod = Literal[
    'opencv_bilinear',      # OpenCV双线性插值（默认）
    'opencv_vng',          # OpenCV VNG算法
    'opencv_ea',           # OpenCV EA算法
    'opencv_ahd',          # OpenCV AHD算法
    'opencv_malvar',       # OpenCV Malvar算法
    'opencv_dcbi',         # OpenCV DCBI算法
    'opencv_hqlinear',     # OpenCV HQ线性算法
    'opencv_edgeaware',    # OpenCV边缘感知算法
    'opencv_ccm',          # OpenCV CCM算法
    'opencv_rccm',         # OpenCV RCCM算法
    'opencv_rccm_simple',  # OpenCV RCCM简单算法
    'opencv_rccm_advanced' # OpenCV RCCM高级算法
]

def enhanced_demosaic(raw_data: np.ndarray, 
                     bayer_pattern: str = 'rggb',
                     method: DemosaicMethod = 'opencv_bilinear',
                     apply_rb_swap: bool = True) -> Optional[np.ndarray]:
    """
    增强的去马赛克函数，支持多种算法
    
    Args:
        raw_data: RAW数据数组
        bayer_pattern: Bayer模式 ('rggb', 'bggr', 'grbg', 'gbrg')
        method: 去马赛克算法
        apply_rb_swap: 是否应用R/B通道交换（针对RGGB模式）
        
    Returns:
        去马赛克后的彩色图像，失败时返回None
    """
    print(f"Applying enhanced demosaicing: {method} for {bayer_pattern} pattern")
    
    try:
        # 转换为uint16
        raw_data_uint16 = raw_data.astype(np.uint16)
        print(f"  Input: {raw_data_uint16.shape}, dtype: {raw_data_uint16.dtype}")
        
        # 根据Bayer模式选择OpenCV常量
        bayer_constants = {
            'rggb': cv2.COLOR_BayerRG2BGR,
            'bggr': cv2.COLOR_BayerBG2BGR,
            'grbg': cv2.COLOR_BayerGR2BGR,
            'gbrg': cv2.COLOR_BayerGB2BGR
        }
        
        if bayer_pattern not in bayer_constants:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
        
        bayer_constant = bayer_constants[bayer_pattern]
        
        # 根据算法选择不同的处理方式
        if method == 'opencv_bilinear':
            # 标准双线性插值
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant)
            
        elif method == 'opencv_vng':
            # VNG (Variable Number of Gradients) 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_VNG)
            
        elif method == 'opencv_ea':
            # EA (Edge-Aware) 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_EA)
            
        elif method == 'opencv_ahd':
            # AHD (Adaptive Homogeneity-Directed) 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_AHD)
            
        elif method == 'opencv_malvar':
            # Malvar 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_MALVAR)
            
        elif method == 'opencv_dcbi':
            # DCBI (Directional Color Bilinear Interpolation) 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_DCBI)
            
        elif method == 'opencv_hqlinear':
            # HQ Linear 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_HQ_LINEAR)
            
        elif method == 'opencv_edgeaware':
            # Edge-Aware 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_EDGE_AWARE)
            
        elif method == 'opencv_ccm':
            # CCM (Color Correction Matrix) 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_CCM)
            
        elif method == 'opencv_rccm':
            # RCCM (Robust Color Correction Matrix) 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_RCCM)
            
        elif method == 'opencv_rccm_simple':
            # RCCM Simple 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_RCCM_SIMPLE)
            
        elif method == 'opencv_rccm_advanced':
            # RCCM Advanced 算法
            demosaiced = cv2.cvtColor(raw_data_uint16, bayer_constant | cv2.COLOR_BAYER_RCCM_ADVANCED)
            
        else:
            raise ValueError(f"Unsupported demosaic method: {method}")
        
        # 应用R/B通道交换（仅对RGGB模式）
        if apply_rb_swap and bayer_pattern == 'rggb':
            corrected = demosaiced.copy()
            corrected[:, :, 0] = demosaiced[:, :, 2]  # B = R
            corrected[:, :, 2] = demosaiced[:, :, 0]  # R = B
            demosaiced = corrected
            print(f"  Applied R/B channel swap for RGGB pattern")
        
        print(f"  Demosaicing completed: {demosaiced.shape}, range: {np.min(demosaiced)}-{np.max(demosaiced)}")
        return demosaiced
        
    except Exception as e:
        print(f"  Error in enhanced demosaicing: {e}")
        return None

def compare_demosaic_methods(raw_data: np.ndarray, 
                           bayer_pattern: str = 'rggb',
                           methods: list = None) -> dict:
    """
    比较不同去马赛克算法的结果
    
    Args:
        raw_data: RAW数据数组
        bayer_pattern: Bayer模式
        methods: 要比较的算法列表
        
    Returns:
        包含各算法结果的字典
    """
    if methods is None:
        methods = [
            'opencv_bilinear',
            'opencv_vng', 
            'opencv_ea',
            'opencv_ahd',
            'opencv_malvar'
        ]
    
    print(f"Comparing demosaic methods for {bayer_pattern} pattern...")
    results = {}
    
    for method in methods:
        print(f"\nTesting method: {method}")
        result = enhanced_demosaic(raw_data, bayer_pattern, method)
        if result is not None:
            results[method] = result
            print(f"  ✅ {method} completed successfully")
        else:
            print(f"  ❌ {method} failed")
    
    return results

def get_available_methods() -> list:
    """
    获取可用的去马赛克算法列表
    
    Returns:
        可用算法列表
    """
    return [
        'opencv_bilinear',      # 双线性插值（最快，质量一般）
        'opencv_vng',          # VNG算法（质量好，速度中等）
        'opencv_ea',           # EA算法（边缘感知）
        'opencv_ahd',          # AHD算法（高质量，速度慢）
        'opencv_malvar',       # Malvar算法（质量好）
        'opencv_dcbi',         # DCBI算法（方向性插值）
        'opencv_hqlinear',     # HQ线性算法（高质量线性）
        'opencv_edgeaware',    # 边缘感知算法
        'opencv_ccm',          # CCM算法（颜色校正）
        'opencv_rccm',         # RCCM算法（鲁棒颜色校正）
        'opencv_rccm_simple',  # RCCM简单算法
        'opencv_rccm_advanced' # RCCM高级算法
    ]

def get_method_description(method: DemosaicMethod) -> str:
    """
    获取算法描述
    
    Args:
        method: 算法名称
        
    Returns:
        算法描述
    """
    descriptions = {
        'opencv_bilinear': '双线性插值 - 最快，质量一般，适合实时处理',
        'opencv_vng': 'VNG算法 - 质量好，速度中等，适合一般用途',
        'opencv_ea': 'EA算法 - 边缘感知，保持边缘细节',
        'opencv_ahd': 'AHD算法 - 高质量，速度慢，适合高质量输出',
        'opencv_malvar': 'Malvar算法 - 质量好，平衡速度和质量',
        'opencv_dcbi': 'DCBI算法 - 方向性插值，减少伪影',
        'opencv_hqlinear': 'HQ线性算法 - 高质量线性插值',
        'opencv_edgeaware': '边缘感知算法 - 保持边缘锐度',
        'opencv_ccm': 'CCM算法 - 包含颜色校正',
        'opencv_rccm': 'RCCM算法 - 鲁棒颜色校正',
        'opencv_rccm_simple': 'RCCM简单算法 - 简化版本',
        'opencv_rccm_advanced': 'RCCM高级算法 - 完整版本'
    }
    
    return descriptions.get(method, '未知算法')

if __name__ == "__main__":
    # 测试代码
    print("Enhanced Demosaic Module")
    print("=" * 40)
    
    print("\nAvailable methods:")
    methods = get_available_methods()
    for method in methods:
        print(f"  {method}: {get_method_description(method)}")
    
    print(f"\nTotal methods: {len(methods)}")
