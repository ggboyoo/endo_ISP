#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dark RAW Image Analysis Tool
读取暗场RAW图像，绘制直方图，并去除异常大值
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 配置参数
DARK_RAW_PATH = r"F:\ZJU\Picture\dark\g5\average_dark.raw" # 暗场RAW图像路径
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
DATA_TYPE = np.uint16
OUTPUT_DIRECTORY = "dark_analysis_output"

# 异常值阈值
OUTLIER_THRESHOLD = 100

def read_raw_image(file_path, width, height, dtype=np.uint16):
    """读取RAW图像文件"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=dtype)
        
        if len(raw_data) != width * height:
            print(f"警告: 数据长度 {len(raw_data)} 与预期 {width * height} 不匹配")
            # 尝试调整尺寸
            if len(raw_data) == width * height * 2:  # 可能是16位数据
                raw_data = raw_data.reshape(height, width)
            else:
                raw_data = raw_data[:width * height].reshape(height, width)
        else:
            raw_data = raw_data.reshape(height, width)
        
        return raw_data
    except Exception as e:
        print(f"读取RAW文件失败: {e}")
        return None

def analyze_dark_image(raw_data, threshold=100):
    """分析暗场图像"""
    print(f"图像尺寸: {raw_data.shape}")
    print(f"数据类型: {raw_data.dtype}")
    print(f"数值范围: {raw_data.min()} - {raw_data.max()}")
    print(f"均值: {raw_data.mean():.2f}")
    print(f"标准差: {raw_data.std():.2f}")
    
    # 统计异常值
    outliers = raw_data > threshold
    outlier_count = np.sum(outliers)
    outlier_percentage = (outlier_count / raw_data.size) * 100
    
    print(f"异常值统计 (> {threshold}):")
    print(f"  数量: {outlier_count}")
    print(f"  百分比: {outlier_percentage:.2f}%")
    
    # 去除异常值后的统计
    cleaned_data = raw_data.copy()
    cleaned_data[outliers] = 0  # 将异常值设为0
    
    print(f"去除异常值后:")
    print(f"  均值: {cleaned_data.mean():.2f}")
    print(f"  标准差: {cleaned_data.std():.2f}")
    
    return cleaned_data, outliers

def plot_histogram(raw_data, outliers, threshold=100, save_path=None):
    """绘制直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Dark RAW Image Analysis - Normal Pixels Only', fontsize=16)
    
    # 只对正常像素绘制直方图（剔除异常值）
    normal_pixels = raw_data[~outliers]  # 使用~outliers获取非异常值
    
    # 计算直方图数据
    hist, bin_edges = np.histogram(normal_pixels.flatten(), bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 绘制直方图
    bars = axes[0].bar(bin_centers, hist, alpha=0.7, color='green', edgecolor='black', width=bin_edges[1]-bin_edges[0])
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
    axes[0].set_title('Normal Pixels Histogram (Outliers Excluded)')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Pixel Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
     
    # 在横坐标上标注像素值
    # 只标注有像素的bin的像素值
    for i, (center, count) in enumerate(zip(bin_centers, hist)):
        if count > 0:  # 只标注有像素的bin
            # 在横坐标下方标注像素值
            axes[0].text(center, -max(hist) * 0.05, f'{int(center)}', 
                        ha='center', va='top', fontsize=8, color='blue', rotation=45)
     
    # 对正常像素进行归一化显示
    normal_pixels = raw_data[~outliers]  # 获取正常像素
    if len(normal_pixels) > 0:
        # 计算正常像素的最大值和最小值
        normal_min = normal_pixels.min()
        normal_max = 10
        
        # 归一化正常像素到0-1范围
        if normal_max > normal_min:
            normalized_image = (raw_data - normal_min) / (normal_max - normal_min)
            normalized_image = np.clip(normalized_image, 0, 1)  # 确保在0-1范围内
        else:
            normalized_image = np.zeros_like(raw_data, dtype=np.float64)
        
        im1 = axes[1].imshow(normalized_image, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Normalized Dark Image (Normal: {normal_min}-{normal_max})')
    else:
        im1 = axes[1].imshow(raw_data, cmap='gray', vmin=0, vmax=raw_data.max())
        axes[1].set_title('Original Dark Image')
    
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 添加统计信息
    stats_text = f'Normal Pixels: {len(normal_pixels):,}\n'
    stats_text += f'Outliers: {np.sum(outliers):,}\n'
    stats_text += f'Mean: {normal_pixels.mean():.2f}\n'
    stats_text += f'Std: {normal_pixels.std():.2f}\n'
    stats_text += f'Min: {normal_pixels.min()}\n'
    stats_text += f'Max: {normal_pixels.max()}'
    
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"直方图已保存到: {save_path}")
    
    plt.show()

def plot_outlier_distribution(raw_data, outliers, threshold=100, save_path=None):
    """绘制异常值分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Outlier Distribution Analysis', fontsize=16)
    
    # 异常值位置图
    axes[0].imshow(outliers, cmap='Reds', alpha=0.8)
    axes[0].set_title(f'Outlier Locations (> {threshold})')
    axes[0].axis('off')
    
    # 异常值统计
    outlier_values = raw_data[outliers]
    if len(outlier_values) > 0:
        axes[1].hist(outlier_values, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1].set_title('Outlier Value Distribution')
        axes[1].set_xlabel('Outlier Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # 添加统计信息
        axes[1].text(0.7, 0.8, f'Count: {len(outlier_values)}\nMin: {outlier_values.min()}\nMax: {outlier_values.max()}\nMean: {outlier_values.mean():.2f}', 
                    transform=axes[1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    else:
        axes[1].text(0.5, 0.5, 'No outliers found', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Outlier Value Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"异常值分布图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    print("Dark RAW Image Analysis Tool")
    print("=" * 50)
    
    # 检查输入文件
    if not os.path.exists(DARK_RAW_PATH):
        print(f"错误: 找不到文件 {DARK_RAW_PATH}")
        return
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIRECTORY)
    output_dir.mkdir(exist_ok=True)
    
    # 读取RAW图像
    print(f"读取RAW图像: {DARK_RAW_PATH}")
    raw_data = read_raw_image(DARK_RAW_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, DATA_TYPE)
    
    if raw_data is None:
        print("读取失败，程序退出")
        return
    
    # 分析暗场图像
    print("\n分析暗场图像...")
    cleaned_data, outliers = analyze_dark_image(raw_data, OUTLIER_THRESHOLD)
    
    # 绘制直方图
    print("\n绘制直方图...")
    histogram_path = output_dir / f"dark_histogram_analysis_{OUTLIER_THRESHOLD}.png"
    plot_histogram(raw_data, outliers, OUTLIER_THRESHOLD, histogram_path)
    
    # 保存清理后的数据
    cleaned_path = output_dir / "cleaned_dark.raw"
    with open(cleaned_path, 'wb') as f:
        f.write(cleaned_data.astype(DATA_TYPE).tobytes())
    print(f"\n清理后的数据已保存到: {cleaned_path}")
    
    # 保存异常值掩码
    mask_path = output_dir / "outlier_mask.npy"
    np.save(mask_path, outliers)
    print(f"异常值掩码已保存到: {mask_path}")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()