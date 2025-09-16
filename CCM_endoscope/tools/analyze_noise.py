#!/usr/bin/env python3
"""
Noise Analysis Program
读取Excel文件中的RGGB分析数据，绘制方差-均值图表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend for plot display
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 默认配置
DEFAULT_EXCEL_PATH = r"F:\ZJU\Picture\denoise\g1/rggb_average_results.xlsx"
DEFAULT_OUTPUT_DIR = r"F:\ZJU\Picture\denoise\g1/noise_analysis_output"

def detect_outliers_iqr(x, y, factor=1.5):
    """
    使用IQR方法检测异常点
    
    Args:
        x: x轴数据
        y: y轴数据
        factor: IQR倍数因子，默认1.5
        
    Returns:
        tuple: (inlier_indices, outlier_indices)
    """
    # 计算残差
    if len(x) < 2:
        return np.arange(len(x)), np.array([])
    
    # 简单线性拟合
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = np.abs(y - y_pred)
    
    # 计算残差的IQR
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1
    
    # 异常点阈值
    threshold = q3 + factor * iqr
    
    # 识别异常点
    outlier_mask = residuals > threshold
    inlier_indices = np.where(~outlier_mask)[0]
    outlier_indices = np.where(outlier_mask)[0]
    
    return inlier_indices, outlier_indices

def range_filtered_fitting(x, y, mean_range=(0, 1000), target_r2=0.99, max_iterations=10, outlier_factor=1.5):
    """
    基于均值范围过滤的拟合，只使用指定范围内的数据点
    
    Args:
        x: x轴数据（均值）
        y: y轴数据（方差）
        mean_range: 均值范围，默认(0, 500)
        target_r2: 目标R²值，默认0.99
        max_iterations: 最大迭代次数，默认10
        outlier_factor: 异常点检测因子，默认1.5
        
    Returns:
        dict: 包含拟合结果的字典
    """
    if len(x) < 2:
        return {
            'slope': 0, 'intercept': 0, 'r2': 0, 'iterations': 0,
            'final_inliers': np.arange(len(x)), 'outliers_removed': [],
            'range_filtered_points': 0, 'original_points': len(x)
        }
    
    # 过滤均值范围内的点
    range_mask = (x >= mean_range[0]) & (x <= mean_range[1])
    filtered_x = x[range_mask]
    filtered_y = y[range_mask]
    filtered_indices = np.where(range_mask)[0]
    
    print(f"范围过滤: 原始数据点 {len(x)} 个，过滤后 {len(filtered_x)} 个 (均值范围: {mean_range[0]}-{mean_range[1]})")
    
    if len(filtered_x) < 2:
        print(f"过滤后数据点不足，无法拟合")
        return {
            'slope': 0, 'intercept': 0, 'r2': 0, 'iterations': 0,
            'final_inliers': np.arange(len(x)), 'outliers_removed': [],
            'range_filtered_points': len(filtered_x), 'original_points': len(x)
        }
    
    # 对过滤后的数据进行迭代拟合
    current_x = filtered_x.copy()
    current_y = filtered_y.copy()
    current_indices = filtered_indices.copy()
    all_outliers = []
    iteration = 0
    
    print(f"开始范围过滤后的迭代拟合，目标R² = {target_r2}")
    
    for iteration in range(max_iterations):
        # 线性拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(current_x, current_y)
        r2 = r_value ** 2
        
        print(f"  迭代 {iteration + 1}: R² = {r2:.6f}, 数据点 = {len(current_x)}")
        
        # 检查是否达到目标R²
        if r2 >= target_r2:
            print(f"  达到目标R² = {target_r2:.6f}，停止迭代")
            break
        
        # 检测异常点
        inlier_indices, outlier_indices = detect_outliers_iqr(current_x, current_y, outlier_factor)
        
        if len(outlier_indices) == 0:
            print(f"  未发现异常点，停止迭代")
            break
        
        # 记录被剔除的异常点（相对于原始数据的索引）
        current_outliers = current_indices[outlier_indices]
        all_outliers.extend(current_outliers)
        
        # 更新数据（只保留内点）
        current_x = current_x[inlier_indices]
        current_y = current_y[inlier_indices]
        current_indices = current_indices[inlier_indices]
        
        if len(current_x) < 2:
            print(f"  数据点不足，停止迭代")
            break
    
    # 最终拟合
    if len(current_x) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(current_x, current_y)
        r2 = r_value ** 2
    else:
        slope, intercept, r2 = 0, 0, 0
    
    print(f"范围过滤拟合最终结果: R² = {r2:.6f}, 迭代次数 = {iteration + 1}, 剔除异常点 = {len(all_outliers)}")
    print(f"最终使用数据点: {len(current_x)} 个 (原始 {len(x)} 个)")
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'iterations': iteration + 1,
        'final_inliers': current_indices,
        'outliers_removed': all_outliers,
        'final_x': current_x,
        'final_y': current_y,
        'range_filtered_points': len(filtered_x),
        'original_points': len(x),
        'mean_range': mean_range
    }

def iterative_fitting_with_outlier_removal(x, y, target_r2=0.99, max_iterations=10, outlier_factor=1.5):
    """
    迭代拟合，剔除异常点直到达到目标R²
    
    Args:
        x: x轴数据
        y: y轴数据
        target_r2: 目标R²值，默认0.99
        max_iterations: 最大迭代次数，默认10
        outlier_factor: 异常点检测因子，默认1.5
        
    Returns:
        dict: 包含拟合结果的字典
    """
    if len(x) < 2:
        return {
            'slope': 0, 'intercept': 0, 'r2': 0, 'iterations': 0,
            'final_inliers': np.arange(len(x)), 'outliers_removed': []
        }
    
    current_x = x.copy()
    current_y = y.copy()
    all_outliers = []
    iteration = 0
    
    print(f"开始迭代拟合，目标R² = {target_r2}")
    
    for iteration in range(max_iterations):
        # 线性拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(current_x, current_y)
        r2 = r_value ** 2
        
        print(f"  迭代 {iteration + 1}: R² = {r2:.6f}, 数据点 = {len(current_x)}")
        
        # 检查是否达到目标R²
        if r2 >= target_r2:
            print(f"  达到目标R² = {target_r2:.6f}，停止迭代")
            break
        
        # 检测异常点
        inlier_indices, outlier_indices = detect_outliers_iqr(current_x, current_y, outlier_factor)
        
        if len(outlier_indices) == 0:
            print(f"  未发现异常点，停止迭代")
            break
        
        # 记录被剔除的异常点
        current_outliers = np.arange(len(current_x))[outlier_indices]
        all_outliers.extend(current_outliers)
        
        # 更新数据（只保留内点）
        current_x = current_x[inlier_indices]
        current_y = current_y[inlier_indices]
        
        if len(current_x) < 2:
            print(f"  数据点不足，停止迭代")
            break
    
    # 最终拟合
    if len(current_x) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(current_x, current_y)
        r2 = r_value ** 2
    else:
        slope, intercept, r2 = 0, 0, 0
    
    print(f"最终结果: R² = {r2:.6f}, 迭代次数 = {iteration + 1}, 剔除异常点 = {len(all_outliers)}")
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'iterations': iteration + 1,
        'final_inliers': np.arange(len(current_x)),
        'outliers_removed': all_outliers,
        'final_x': current_x,
        'final_y': current_y
    }

def load_excel_data(excel_path: str) -> pd.DataFrame:
    """
    加载Excel文件中的RGGB分析数据
    
    Args:
        excel_path: Excel文件路径
        
    Returns:
        DataFrame containing the data
    """
    try:
        df = pd.read_excel(excel_path)
        print(f"成功加载Excel文件: {excel_path}")
        print(f"数据行数: {len(df)}")
        print(f"数据列数: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"加载Excel文件失败: {e}")
        return None

def dual_fitting_analysis(x, y, mean_range=(0, 1000), target_r2=0.99, max_iterations=10, outlier_factor=1.5):
    """
    同时进行全数据拟合和范围过滤拟合
    
    Args:
        x: x轴数据（均值）
        y: y轴数据（方差）
        mean_range: 均值范围，默认(0, 1000)
        target_r2: 目标R²值，默认0.99
        max_iterations: 最大迭代次数，默认10
        outlier_factor: 异常点检测因子，默认1.5
        
    Returns:
        dict: 包含两种拟合结果的字典
    """
    print(f"开始双拟合分析...")
    print(f"数据点总数: {len(x)}")
    print(f"均值范围: {x.min():.2f} - {x.max():.2f}")
    
    # 1. 全数据拟合
    print(f"\n=== 全数据拟合 ===")
    full_result = iterative_fitting_with_outlier_removal(x, y, target_r2, max_iterations, outlier_factor)
    
    # 2. 范围过滤拟合
    print(f"\n=== 范围过滤拟合 (均值范围: {mean_range[0]}-{mean_range[1]}) ===")
    range_result = range_filtered_fitting(x, y, mean_range, target_r2, max_iterations, outlier_factor)
    
    # 3. 比较结果
    print(f"\n=== 拟合结果比较 ===")
    print(f"全数据拟合:")
    print(f"  R² = {full_result['r2']:.6f}")
    print(f"  斜率 = {full_result['slope']:.6f}")
    print(f"  截距 = {full_result['intercept']:.6f}")
    print(f"  使用数据点 = {len(full_result['final_x'])}")
    print(f"  剔除异常点 = {len(full_result['outliers_removed'])}")
    
    print(f"范围过滤拟合:")
    print(f"  R² = {range_result['r2']:.6f}")
    print(f"  斜率 = {range_result['slope']:.6f}")
    print(f"  截距 = {range_result['intercept']:.6f}")
    print(f"  使用数据点 = {len(range_result['final_x'])}")
    print(f"  剔除异常点 = {len(range_result['outliers_removed'])}")
    print(f"  范围过滤点 = {range_result['range_filtered_points']}")
    
    return {
        'full_data': full_result,
        'range_filtered': range_result,
        'comparison': {
            'slope_diff': abs(full_result['slope'] - range_result['slope']),
            'intercept_diff': abs(full_result['intercept'] - range_result['intercept']),
            'r2_diff': abs(full_result['r2'] - range_result['r2'])
        }
    }

def create_variance_mean_plot(df: pd.DataFrame, output_dir: Path, target_r2=0.99, use_range_filter=True, mean_range=(0, 500)):
    """
    创建方差-均值图表，包含迭代拟合直线（剔除异常点）
    
    Args:
        df: 包含RGGB数据的DataFrame
        output_dir: 输出目录
        target_r2: 目标R²值，默认0.99
        use_range_filter: 是否使用均值范围过滤，默认True
        mean_range: 均值范围，默认(0, 500)
    """
    print("创建方差-均值图表（带异常点剔除）...")
    
    # 定义通道和颜色
    channels = ['R', 'G1', 'G2', 'B']
    colors = ['red', 'green', 'lightgreen', 'blue']
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 收集所有数据点用于整体拟合
    all_means = []
    all_variances = []
    
    # 为每个通道绘制散点图和拟合直线
    for channel, color in zip(channels, colors):
        mean_col = f'{channel}_Mean'
        var_col = f'{channel}_Variance'
        
        # 提取均值和方差数据
        means = df[mean_col].values
        variances = df[var_col].values
        
        # 收集所有数据点
        all_means.extend(means)
        all_variances.extend(variances)
        
        # 绘制散点图
        plt.scatter(means, variances, label=f'{channel} Channel', 
                   color=color, alpha=0.7, s=50)
        
        # 为每个通道绘制拟合直线（使用迭代拟合）
        if len(means) > 1:  # 至少需要2个点才能拟合直线
            print(f"\n{channel} 通道拟合:")
            
            # 选择拟合方法
            if use_range_filter:
                fit_result = range_filtered_fitting(means, variances, mean_range, target_r2)
                fit_type = "范围过滤拟合"
            else:
                fit_result = iterative_fitting_with_outlier_removal(means, variances, target_r2)
                fit_type = "全数据拟合"
            
            if fit_result['r2'] > 0:
                # 计算拟合直线
                x_fit = np.linspace(min(means), max(means), 100)
                y_fit = fit_result['slope'] * x_fit + fit_result['intercept']
            
                # 绘制拟合直线
                plt.plot(x_fit, y_fit, color=color, linestyle='--', alpha=0.8, linewidth=2)
                
                # 标记被剔除的异常点
                if len(fit_result['outliers_removed']) > 0:
                    outlier_means = means[fit_result['outliers_removed']]
                    outlier_variances = variances[fit_result['outliers_removed']]
                    plt.scatter(outlier_means, outlier_variances, 
                              color=color, marker='x', s=100, alpha=0.8, 
                              label=f'{channel} Outliers ({len(fit_result["outliers_removed"])})')
                
                # 如果使用范围过滤，标记范围外的点
                if use_range_filter and 'range_filtered_points' in fit_result:
                    range_outside_mask = (means < mean_range[0]) | (means > mean_range[1])
                    if np.any(range_outside_mask):
                        range_outside_means = means[range_outside_mask]
                        range_outside_variances = variances[range_outside_mask]
                        plt.scatter(range_outside_means, range_outside_variances, 
                                  color=color, marker='o', s=30, alpha=0.3, 
                                  label=f'{channel} Range Outside ({np.sum(range_outside_mask)})')
                
                print(f"  {channel} 通道 ({fit_type}): {len(means)} 个数据点, 最终R² = {fit_result['r2']:.4f}, 剔除异常点 = {len(fit_result['outliers_removed'])}")
                if use_range_filter and 'range_filtered_points' in fit_result:
                    print(f"    范围过滤: 使用 {fit_result['range_filtered_points']} 个点 (范围: {mean_range[0]}-{mean_range[1]})")
            else:
                print(f"  {channel} 通道: 拟合失败")
        else:
            print(f"  {channel} 通道: {len(means)} 个数据点 (数据点不足，无法拟合)")
    
    # 绘制整体拟合直线（使用迭代拟合）
    if len(all_means) > 1:
        all_means = np.array(all_means)
        all_variances = np.array(all_variances)
        
        print(f"\n整体拟合:")
        
        # 选择拟合方法
        if use_range_filter:
            fit_result_all = range_filtered_fitting(all_means, all_variances, mean_range, target_r2)
            fit_type = "范围过滤拟合"
        else:
            fit_result_all = iterative_fitting_with_outlier_removal(all_means, all_variances, target_r2)
            fit_type = "全数据拟合"
        
        if fit_result_all['r2'] > 0:
            # 计算整体拟合直线
            x_fit_all = np.linspace(min(all_means), max(all_means), 100)
            y_fit_all = fit_result_all['slope'] * x_fit_all + fit_result_all['intercept']
            
            # 绘制整体拟合直线（粗黑线）
            plt.plot(x_fit_all, y_fit_all, color='black', linestyle='-', 
                    linewidth=3, alpha=0.9, 
                    label=f'Overall Fit (R² = {fit_result_all["r2"]:.4f}, 剔除{len(fit_result_all["outliers_removed"])}个异常点)')
            
            # 标记被剔除的异常点
            if len(fit_result_all['outliers_removed']) > 0:
                outlier_means = all_means[fit_result_all['outliers_removed']]
                outlier_variances = all_variances[fit_result_all['outliers_removed']]
                plt.scatter(outlier_means, outlier_variances, 
                          color='black', marker='x', s=150, alpha=0.8, 
                          label=f'Overall Outliers ({len(fit_result_all["outliers_removed"])})')
            
            # 如果使用范围过滤，标记范围外的点
            if use_range_filter and 'range_filtered_points' in fit_result_all:
                range_outside_mask = (all_means < mean_range[0]) | (all_means > mean_range[1])
                if np.any(range_outside_mask):
                    range_outside_means = all_means[range_outside_mask]
                    range_outside_variances = all_variances[range_outside_mask]
                    plt.scatter(range_outside_means, range_outside_variances, 
                              color='black', marker='o', s=30, alpha=0.3, 
                              label=f'Overall Range Outside ({np.sum(range_outside_mask)})')
            
            print(f"  整体拟合 ({fit_type}): R² = {fit_result_all['r2']:.4f}, 斜率 = {fit_result_all['slope']:.4f}, 截距 = {fit_result_all['intercept']:.4f}")
            print(f"  剔除异常点: {len(fit_result_all['outliers_removed'])} 个")
            if use_range_filter and 'range_filtered_points' in fit_result_all:
                print(f"  范围过滤: 使用 {fit_result_all['range_filtered_points']} 个点 (范围: {mean_range[0]}-{mean_range[1]})")
        else:
            print(f"  整体拟合失败")
    
    # 设置图表属性
    plt.xlabel('Mean Value', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.title(f'Variance vs Mean Plot with Iterative Fitting (Target R² ≥ {target_r2})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 自适应坐标轴范围
    mean_range = max(all_means) - min(all_means)
    var_range = max(all_variances) - min(all_variances)
    
    plt.xlim(min(all_means) - mean_range * 0.05, max(all_means) + mean_range * 0.05)
    plt.ylim(min(all_variances) - var_range * 0.05, max(all_variances) + var_range * 0.05)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = output_dir / "variance_mean_plot_with_outlier_removal.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {plot_path}")
    
    # 显示图表
    plt.show()

def create_dual_fitting_plots(df: pd.DataFrame, output_dir: Path, target_r2=0.99, mean_range=(0, 1000)):
    """
    创建双拟合图表，分别显示全数据拟合和范围过滤拟合的结果
    
    Args:
        df: 包含RGGB数据的DataFrame
        output_dir: 输出目录
        target_r2: 目标R²值，默认0.99
        mean_range: 均值范围，默认(0, 1000)
    """
    print("创建双拟合图表...")
    
    # 定义通道和颜色
    channels = ['R', 'G1', 'G2', 'B']
    colors = ['red', 'green', 'lightgreen', 'blue']
    
    # 收集所有数据点
    all_means = []
    all_variances = []
    
    for channel in channels:
        mean_col = f'{channel}_Mean'
        var_col = f'{channel}_Variance'
        means = df[mean_col].values
        variances = df[var_col].values
        all_means.extend(means)
        all_variances.extend(variances)
    
    all_means = np.array(all_means)
    all_variances = np.array(all_variances)
    
    # 进行双拟合分析
    dual_results = dual_fitting_analysis(all_means, all_variances, mean_range, target_r2)
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 子图1：全数据拟合
    ax1.set_title('Full Data Fitting (All Points)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mean Value', fontsize=12)
    ax1.set_ylabel('Variance', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 绘制散点图
    ax1.scatter(all_means, all_variances, color='blue', alpha=0.6, s=30, label='Data Points')
    
    # 绘制全数据拟合直线
    full_result = dual_results['full_data']
    if full_result['r2'] > 0:
        x_fit = np.linspace(min(all_means), max(all_means), 100)
        y_fit = full_result['slope'] * x_fit + full_result['intercept']
        ax1.plot(x_fit, y_fit, color='red', linewidth=3, alpha=0.8, 
                label=f'Full Data Fit (R² = {full_result["r2"]:.4f})')
        
        # 标记异常点
        if len(full_result['outliers_removed']) > 0:
            outlier_means = all_means[full_result['outliers_removed']]
            outlier_variances = all_variances[full_result['outliers_removed']]
            ax1.scatter(outlier_means, outlier_variances, color='red', marker='x', s=100, alpha=0.8, 
                       label=f'Outliers ({len(full_result["outliers_removed"])})')
    
    ax1.legend()
    
    # 子图2：范围过滤拟合
    ax2.set_title(f'Range Filtered Fitting (Mean: {mean_range[0]}-{mean_range[1]})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean Value', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 绘制散点图
    ax2.scatter(all_means, all_variances, color='blue', alpha=0.6, s=30, label='Data Points')
    
    # 标记范围外的点
    range_outside_mask = (all_means < mean_range[0]) | (all_means > mean_range[1])
    if np.any(range_outside_mask):
        range_outside_means = all_means[range_outside_mask]
        range_outside_variances = all_variances[range_outside_mask]
        ax2.scatter(range_outside_means, range_outside_variances, color='gray', alpha=0.3, s=20, 
                   label=f'Outside Range ({np.sum(range_outside_mask)})')
    
    # 绘制范围过滤拟合直线
    range_result = dual_results['range_filtered']
    if range_result['r2'] > 0:
        x_fit = np.linspace(min(all_means), max(all_means), 100)
        y_fit = range_result['slope'] * x_fit + range_result['intercept']
        ax2.plot(x_fit, y_fit, color='green', linewidth=3, alpha=0.8, 
                label=f'Range Filtered Fit (R² = {range_result["r2"]:.4f})')
        
        # 标记异常点
        if len(range_result['outliers_removed']) > 0:
            outlier_means = all_means[range_result['outliers_removed']]
            outlier_variances = all_variances[range_result['outliers_removed']]
            ax2.scatter(outlier_means, outlier_variances, color='green', marker='x', s=100, alpha=0.8, 
                       label=f'Outliers ({len(range_result["outliers_removed"])})')
    
    ax2.legend()
    
    # 设置相同的坐标轴范围
    mean_range_plot = max(all_means) - min(all_means)
    var_range_plot = max(all_variances) - min(all_variances)
    
    xlim = (min(all_means) - mean_range_plot * 0.05, max(all_means) + mean_range_plot * 0.05)
    ylim = (min(all_variances) - var_range_plot * 0.05, max(all_variances) + var_range_plot * 0.05)
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = output_dir / "dual_fitting_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"双拟合对比图表已保存: {plot_path}")
    
    # 显示图表
    plt.show()
    
    # 保存拟合结果到文件
    save_fitting_results(dual_results, output_dir, mean_range)
    
    return dual_results

def save_fitting_results(dual_results, output_dir: Path, mean_range):
    """
    保存拟合结果到JSON文件
    
    Args:
        dual_results: 双拟合结果
        output_dir: 输出目录
        mean_range: 均值范围
    """
    import json
    from datetime import datetime
    
    # 准备保存的数据
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'mean_range': mean_range,
        'full_data_fitting': {
            'slope': float(dual_results['full_data']['slope']),
            'intercept': float(dual_results['full_data']['intercept']),
            'r2': float(dual_results['full_data']['r2']),
            'iterations': int(dual_results['full_data']['iterations']),
            'data_points_used': len(dual_results['full_data']['final_x']),
            'outliers_removed': len(dual_results['full_data']['outliers_removed'])
        },
        'range_filtered_fitting': {
            'slope': float(dual_results['range_filtered']['slope']),
            'intercept': float(dual_results['range_filtered']['intercept']),
            'r2': float(dual_results['range_filtered']['r2']),
            'iterations': int(dual_results['range_filtered']['iterations']),
            'data_points_used': len(dual_results['range_filtered']['final_x']),
            'outliers_removed': len(dual_results['range_filtered']['outliers_removed']),
            'range_filtered_points': int(dual_results['range_filtered']['range_filtered_points']),
            'original_points': int(dual_results['range_filtered']['original_points'])
        },
        'comparison': {
            'slope_difference': float(dual_results['comparison']['slope_diff']),
            'intercept_difference': float(dual_results['comparison']['intercept_diff']),
            'r2_difference': float(dual_results['comparison']['r2_diff'])
        }
    }
    
    # 保存到JSON文件
    results_path = output_dir / "dual_fitting_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"拟合结果已保存: {results_path}")
    
    # 打印结果摘要
    print(f"\n=== 拟合结果摘要 ===")
    print(f"全数据拟合: 斜率={results_data['full_data_fitting']['slope']:.6f}, 截距={results_data['full_data_fitting']['intercept']:.6f}, R²={results_data['full_data_fitting']['r2']:.6f}")
    print(f"范围过滤拟合: 斜率={results_data['range_filtered_fitting']['slope']:.6f}, 截距={results_data['range_filtered_fitting']['intercept']:.6f}, R²={results_data['range_filtered_fitting']['r2']:.6f}")
    print(f"差异: 斜率={results_data['comparison']['slope_difference']:.6f}, 截距={results_data['comparison']['intercept_difference']:.6f}")

def print_data_summary(df: pd.DataFrame):
    """
    打印数据摘要
    
    Args:
        df: 包含RGGB数据的DataFrame
    """
    print("\n=== 数据摘要 ===")
    print(f"总分析次数: {len(df)}")
    print(f"数据时间范围: {df['Timestamp'].min()} 到 {df['Timestamp'].max()}")
    
    print("\n=== 各通道数据点统计 ===")
    channels = ['R', 'G1', 'G2', 'B']
    
    total_points = 0
    for channel in channels:
        mean_col = f'{channel}_Mean'
        var_col = f'{channel}_Variance'
        
        points = len(df[mean_col])
        total_points += points
        
        print(f"{channel} 通道: {points} 个数据点")
        print(f"  均值范围: {df[mean_col].min():.1f} - {df[mean_col].max():.1f}")
        print(f"  方差范围: {df[var_col].min():.1f} - {df[var_col].max():.1f}")
    
    print(f"\n总数据点: {total_points}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Analyze RGGB noise data from Excel file with outlier removal")
    parser.add_argument("--excel", default=DEFAULT_EXCEL_PATH, 
                       help=f"Path to Excel file (default: {DEFAULT_EXCEL_PATH})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                       help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--target-r2", type=float, default=0.99,
                       help="Target R² value for iterative fitting (default: 0.99)")
    parser.add_argument("--outlier-factor", type=float, default=1.5,
                       help="Outlier detection factor for IQR method (default: 1.5)")
    parser.add_argument("--use-range-filter", action="store_true", default=True,
                       help="Use mean range filtering for better intercept accuracy (default: True)")
    parser.add_argument("--mean-range-min", type=float, default=0,
                       help="Minimum mean value for range filtering (default: 0)")
    parser.add_argument("--mean-range-max", type=float, default=1000,
                       help="Maximum mean value for range filtering (default: 1000)")
    parser.add_argument("--dual-fitting", action="store_true", default=True,
                       help="Perform both full data and range filtered fitting (default: True)")
    
    args = parser.parse_args()
    
    # 检查Excel文件是否存在
    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"错误: Excel文件不存在: {excel_path}")
        print(f"请确保已经运行过noise_cali.py程序并生成了Excel文件")
        return 1
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据
    df = load_excel_data(str(excel_path))
    if df is None:
        return 1
    
    # 打印数据摘要
    print_data_summary(df)
    
    # 生成方差-均值图表
    try:
        mean_range = (args.mean_range_min, args.mean_range_max)
        print(f"使用目标R² = {args.target_r2}, 异常点检测因子 = {args.outlier_factor}")
        print(f"范围过滤: {'启用' if args.use_range_filter else '禁用'}, 均值范围 = {mean_range[0]}-{mean_range[1]}")
        
        if args.dual_fitting:
            print("执行双拟合分析...")
            create_dual_fitting_plots(df, output_dir, target_r2=args.target_r2, mean_range=mean_range)
        else:
            print("执行单拟合分析...")
            create_variance_mean_plot(df, output_dir, target_r2=args.target_r2, 
                                    use_range_filter=args.use_range_filter, mean_range=mean_range)
    except Exception as e:
        print(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n分析完成! 结果保存在: {output_dir}")
    return 0

if __name__ == "__main__":
    exit(main())