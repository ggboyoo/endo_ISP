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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 默认配置
DEFAULT_EXCEL_PATH = r"F:\ZJU\Picture\denoise\g7/rggb_analysis_results.xlsx"
DEFAULT_OUTPUT_DIR = r"F:\ZJU\Picture\denoise\g7/noise_analysis_output"

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

def create_variance_mean_plot(df: pd.DataFrame, output_dir: Path):
    """
    创建方差-均值图表，包含拟合直线
    
    Args:
        df: 包含RGGB数据的DataFrame
        output_dir: 输出目录
    """
    print("创建方差-均值图表...")
    
    # 定义通道和颜色
    channels = ['R', 'G1', 'G2', 'B']
    colors = ['red', 'green', 'lightgreen', 'blue']
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
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
        
        # 为每个通道绘制拟合直线
        if len(means) > 1:  # 至少需要2个点才能拟合直线
            slope, intercept, r_value, p_value, std_err = stats.linregress(means, variances)
            
            # 计算拟合直线
            x_fit = np.linspace(min(means), max(means), 100)
            y_fit = slope * x_fit + intercept
            
            # 绘制拟合直线
            plt.plot(x_fit, y_fit, color=color, linestyle='--', alpha=0.8, linewidth=2)
            
            print(f"  {channel} 通道: {len(means)} 个数据点, R² = {r_value**2:.4f}")
        else:
            print(f"  {channel} 通道: {len(means)} 个数据点 (数据点不足，无法拟合)")
    
    # 绘制整体拟合直线
    if len(all_means) > 1:
        all_means = np.array(all_means)
        all_variances = np.array(all_variances)
        
        # 整体线性拟合
        slope_all, intercept_all, r_value_all, p_value_all, std_err_all = stats.linregress(all_means, all_variances)
        
        # 计算整体拟合直线
        x_fit_all = np.linspace(min(all_means), max(all_means), 100)
        y_fit_all = slope_all * x_fit_all + intercept_all
        
        # 绘制整体拟合直线（粗黑线）
        plt.plot(x_fit_all, y_fit_all, color='black', linestyle='-', 
                linewidth=3, alpha=0.9, label=f'Overall Fit (R² = {r_value_all**2:.4f})')
        
        print(f"  整体拟合: R² = {r_value_all**2:.4f}, 斜率 = {slope_all:.4f}, 截距 = {intercept_all:.4f}")
    
    # 设置图表属性
    plt.xlabel('Mean Value', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.title('Variance vs Mean Plot with Fitted Lines (All RGGB Channels)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 自适应坐标轴范围
    mean_range = max(all_means) - min(all_means)
    var_range = max(all_variances) - min(all_variances)
    
    plt.xlim(min(all_means) - mean_range * 0.05, max(all_means) + mean_range * 0.05)
    plt.ylim(min(all_variances) - var_range * 0.05, max(all_variances) + var_range * 0.05)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = output_dir / "variance_mean_plot_with_fit.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {plot_path}")
    
    # 显示图表
    plt.show()

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
    parser = argparse.ArgumentParser(description="Analyze RGGB noise data from Excel file")
    parser.add_argument("--excel", default=DEFAULT_EXCEL_PATH, 
                       help=f"Path to Excel file (default: {DEFAULT_EXCEL_PATH})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                       help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    
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
        create_variance_mean_plot(df, output_dir)
    except Exception as e:
        print(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n分析完成! 结果保存在: {output_dir}")
    return 0

if __name__ == "__main__":
    exit(main())