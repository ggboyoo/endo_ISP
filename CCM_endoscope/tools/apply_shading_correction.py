#!/usr/bin/env python3
"""
Apply Lens Shading Correction Script
使用已保存的矫正参数对新的RAW图像进行镜头阴影矫正
"""

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
import json
from datetime import datetime

# Import functions from lens_shading.py
try:
    from lens_shading import (
        load_correction_parameters, 
        shading_correct, 
        batch_shading_correct,
        IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE
    )
except ImportError:
    print("Error: lens_shading.py not found in the same directory!")
    print("Please ensure lens_shading.py is in the same directory as this script.")
    exit(1)

# ============================================================================
# 配置文件 - 直接在这里修改，无需交互输入
# ============================================================================

# 矫正参数路径配置
CORRECTION_PARAMS_DIR = r"F:\ZJU\Picture\lens shading"  # 矫正参数目录

# 输入输出配置
INPUT_IMAGE_PATH = r"F:\ZJU\Picture\lens shading\1000.raw"  # 单张图片路径
INPUT_DIRECTORY = r"F:\ZJU\Picture\lens shading\new_images"      # 批量处理目录
OUTPUT_DIRECTORY = r"F:\ZJU\Picture\lens shading\corrected"      # 输出目录

# 暗电流配置
DARK_IMAGE_PATH = r"F:\ZJU\Picture\dark\g8\average_dark.raw"    # 暗电流图像路径（可选）
ENABLE_DARK_CORRECTION = True                                     # 是否启用暗电流矫正

# 输出格式配置
SAVE_FORMATS = ['raw', 'png', 'jpg']  # 保存格式：raw, png, jpg

# 处理模式
PROCESS_MODE = 'single'  # 'single' 或 'batch'

# 直方图显示配置
SHOW_HISTOGRAMS = False    # 是否显示矫正前后直方图对比
SAVE_HISTOGRAMS = False    # 是否保存直方图到文件

# ============================================================================
# 直方图显示功能
# ============================================================================

def create_histogram_comparison(original_image: np.ndarray, corrected_image: np.ndarray, 
                               dark_corrected_image: np.ndarray = None, 
                               output_dir: str = None, filename: str = "histogram_comparison") -> None:
    """
    创建矫正前后直方图对比
    
    Args:
        original_image: 原始图像数据
        corrected_image: 矫正后图像数据
        dark_corrected_image: 暗电流矫正后图像数据（可选）
        output_dir: 输出目录
        filename: 文件名前缀
    """
    if not SHOW_HISTOGRAMS:
        return
    
    print(f"\nCreating histogram comparison...")
    
    # 创建子图
    if dark_corrected_image is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Image Histogram Comparison', fontsize=16)
        
        # 原始图像直方图
        axes[0, 0].hist(original_image.flatten(), bins=100, alpha=0.7, color='blue', 
                       label='Original', density=True)
        axes[0, 0].set_title('Original Image Histogram')
        axes[0, 0].set_xlabel('Pixel Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 暗电流矫正后直方图
        axes[0, 1].hist(dark_corrected_image.flatten(), bins=100, alpha=0.7, color='green', 
                       label='Dark Corrected', density=True)
        axes[0, 1].set_title('Dark Corrected Image Histogram')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 完全矫正后直方图
        axes[1, 0].hist(corrected_image.flatten(), bins=100, alpha=0.7, color='red', 
                       label='Fully Corrected', density=True)
        axes[1, 0].set_title('Fully Corrected Image Histogram')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 对比直方图
        axes[1, 1].hist(original_image.flatten(), bins=100, alpha=0.5, color='blue', 
                       label='Original', density=True)
        axes[1, 1].hist(corrected_image.flatten(), bins=100, alpha=0.5, color='red', 
                       label='Fully Corrected', density=True)
        axes[1, 1].set_title('Comparison: Original vs Fully Corrected')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Image Histogram Comparison', fontsize=16)
        
        # 原始图像直方图
        axes[0].hist(original_image.flatten(), bins=100, alpha=0.7, color='blue', 
                    label='Original', density=True)
        axes[0].set_title('Original Image Histogram')
        axes[0].set_xlabel('Pixel Value')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 矫正后直方图
        axes[1].hist(original_image.flatten(), bins=100, alpha=0.5, color='blue', 
                    label='Original', density=True)
        axes[1].hist(corrected_image.flatten(), bins=100, alpha=0.5, color='red', 
                    label='Corrected', density=True)
        axes[1].set_title('Comparison: Original vs Corrected')
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 显示直方图
    if SHOW_HISTOGRAMS:
        plt.show()
    
    # 保存直方图
    if SAVE_HISTOGRAMS and output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        histogram_path = output_path / f"{filename}_histogram_comparison.png"
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"Histogram comparison saved: {histogram_path}")
    
    plt.close()


def create_statistics_summary(original_image: np.ndarray, corrected_image: np.ndarray, 
                             dark_corrected_image: np.ndarray = None) -> Dict:
    """
    创建图像统计信息摘要
    
    Args:
        original_image: 原始图像数据
        corrected_image: 矫正后图像数据
        dark_corrected_image: 暗电流矫正后图像数据（可选）
        
    Returns:
        统计信息字典
    """
    stats = {
        'original': {
            'mean': float(np.mean(original_image)),
            'std': float(np.std(original_image)),
            'min': float(np.min(original_image)),
            'max': float(np.max(original_image)),
            'median': float(np.median(original_image))
        },
        'corrected': {
            'mean': float(np.mean(corrected_image)),
            'std': float(np.std(corrected_image)),
            'min': float(np.min(corrected_image)),
            'max': float(np.max(corrected_image)),
            'median': float(np.median(corrected_image))
        }
    }
    
    if dark_corrected_image is not None:
        stats['dark_corrected'] = {
            'mean': float(np.mean(dark_corrected_image)),
            'std': float(np.std(dark_corrected_image)),
            'min': float(np.min(dark_corrected_image)),
            'max': float(np.max(dark_corrected_image)),
            'median': float(np.median(dark_corrected_image))
        }
    
    return stats


def print_statistics_summary(stats: Dict) -> None:
    """
    打印统计信息摘要
    
    Args:
        stats: 统计信息字典
    """
    print(f"\n=== Image Statistics Summary ===")
    
    print(f"Original Image:")
    print(f"  Mean: {stats['original']['mean']:.2f}")
    print(f"  Std:  {stats['original']['std']:.2f}")
    print(f"  Min:  {stats['original']['min']:.2f}")
    print(f"  Max:  {stats['original']['max']:.2f}")
    print(f"  Median: {stats['original']['median']:.2f}")
    
    if 'dark_corrected' in stats:
        print(f"Dark Corrected Image:")
        print(f"  Mean: {stats['dark_corrected']['mean']:.2f}")
        print(f"  Std:  {stats['dark_corrected']['std']:.2f}")
        print(f"  Min:  {stats['dark_corrected']['min']:.2f}")
        print(f"  Max:  {stats['dark_corrected']['max']:.2f}")
        print(f"  Median: {stats['dark_corrected']['median']:.2f}")
    
    print(f"Fully Corrected Image:")
    print(f"  Mean: {stats['corrected']['mean']:.2f}")
    print(f"  Std:  {stats['corrected']['std']:.2f}")
    print(f"  Min:  {stats['corrected']['min']:.2f}")
    print(f"  Max:  {stats['corrected']['max']:.2f}")
    print(f"  Median: {stats['corrected']['median']:.2f}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    print("=== Apply Lens Shading Correction ===")
    print(f"Correction parameters: {CORRECTION_PARAMS_DIR}")
    print(f"Process mode: {PROCESS_MODE}")
    print(f"Dark correction: {ENABLE_DARK_CORRECTION}")
    if ENABLE_DARK_CORRECTION:
        print(f"Dark image: {DARK_IMAGE_PATH}")
    print(f"Output formats: {SAVE_FORMATS}")
    print()
    
    try:
        # 1. 加载矫正参数
        print("Loading correction parameters...")
        correction_params = load_correction_parameters(CORRECTION_PARAMS_DIR)
        print(f"Successfully loaded correction parameters for channels: {list(correction_params.keys())}")
        print()
        
        # 2. 根据模式处理图片
        if PROCESS_MODE == 'single':
            # 单张图片处理
            if not Path(INPUT_IMAGE_PATH).exists():
                print(f"Error: Input image not found: {INPUT_IMAGE_PATH}")
                return
            
            print(f"Processing single image: {INPUT_IMAGE_PATH}")
            result = shading_correct(
                input_image_path=INPUT_IMAGE_PATH,
                correction_params=correction_params,
                dark_image_path=DARK_IMAGE_PATH if ENABLE_DARK_CORRECTION else None,
                output_dir=OUTPUT_DIRECTORY,
                save_formats=SAVE_FORMATS
            )
            
            print(f"\nSingle image correction completed!")
            print(f"Output files:")
            for format_name, file_path in result['saved_files'].items():
                print(f"  {format_name.upper()}: {file_path}")
            
            # 创建直方图对比
            base_filename = Path(INPUT_IMAGE_PATH).stem
            create_histogram_comparison(
                original_image=result['original_image'],
                corrected_image=result['corrected_image'],
                dark_corrected_image=result['dark_corrected_image'],
                output_dir=OUTPUT_DIRECTORY,
                filename=base_filename
            )
            
            # 创建并打印统计信息摘要
            stats = create_statistics_summary(
                original_image=result['original_image'],
                corrected_image=result['corrected_image'],
                dark_corrected_image=result['dark_corrected_image']
            )
            print_statistics_summary(stats)
        
        elif PROCESS_MODE == 'batch':
            # 批量处理
            if not Path(INPUT_DIRECTORY).exists():
                print(f"Error: Input directory not found: {INPUT_DIRECTORY}")
                return
            
            print(f"Processing batch images from: {INPUT_DIRECTORY}")
            batch_results = batch_shading_correct(
                input_dir=INPUT_DIRECTORY,
                correction_params=correction_params,
                dark_image_path=DARK_IMAGE_PATH if ENABLE_DARK_CORRECTION else None,
                output_dir=OUTPUT_DIRECTORY,
                save_formats=SAVE_FORMATS
            )
            
            print(f"\nBatch processing completed!")
            print(f"Total files processed: {len(batch_results)}")
            successful = len([r for r in batch_results if r['success']])
            failed = len([r for r in batch_results if not r['success']])
            print(f"Successful: {successful}, Failed: {failed}")
            
            # 为批量处理创建汇总直方图（仅显示第一个成功处理的文件）
            if successful > 0 and SHOW_HISTOGRAMS:
                first_successful = next((r for r in batch_results if r['success']), None)
                if first_successful:
                    print(f"\nNote: Histogram comparison is only shown for the first processed file.")
                    print(f"For detailed histograms of each file, use single mode processing.")
        
        else:
            print(f"Error: Invalid process mode: {PROCESS_MODE}")
            print("Please set PROCESS_MODE to 'single' or 'batch'")
            return
        
        print(f"\n=== All Done! ===")
        print(f"Results saved to: {OUTPUT_DIRECTORY}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# ============================================================================
# 高级用法示例
# ============================================================================

def advanced_usage_examples():
    """
    高级用法示例
    """
    print("=== Advanced Usage Examples ===")
    
    # 示例1: 自定义矫正参数
    # custom_params = {
    #     'R': np.ones((1080, 1920)) * 1.1,    # R通道轻微增强
    #     'G1': np.ones((1080, 1920)) * 1.0,   # G1通道不变
    #     'G2': np.ones((1080, 1920)) * 1.0,   # G2通道不变
    #     'B': np.ones((1080, 1920)) * 0.9     # B通道轻微减弱
    # }
    
    # 示例2: 只保存特定格式
    # result = shading_correct(
    #     input_image_path="image.raw",
    #     correction_params=correction_params,
    #     save_formats=['png']  # 只保存PNG
    # )
    
    # 示例3: 使用不同的输出目录
    # result = shading_correct(
    #     input_image_path="image.raw",
    #     correction_params=correction_params,
    #     output_dir="custom_output"  # 自定义输出目录
    # )
    
    print("请取消注释上述代码来使用这些高级功能")


# 如果要使用高级示例，取消注释下面这行
# advanced_usage_examples()
