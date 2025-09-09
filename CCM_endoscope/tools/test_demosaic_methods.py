#!/usr/bin/env python3
"""
测试不同去马赛克算法的脚本
"""

import numpy as np
import cv2
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_demosaic_methods():
    """测试不同去马赛克算法"""
    print("=" * 60)
    print("Testing Different Demosaic Methods")
    print("=" * 60)
    
    try:
        from enhanced_demosaic import get_available_methods, get_method_description, compare_demosaic_methods
        from raw_reader import read_raw_image
        from ISP import process_single_image, load_dark_reference, load_correction_parameters, load_white_balance_parameters
        
        # 配置参数
        raw_file = r"F:\ZJU\Picture\ccm\ccm_1\25-09-05 101447.raw"
        output_dir = Path(r"F:\ZJU\Picture\demosaic_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            'IMAGE_WIDTH': 3840,
            'IMAGE_HEIGHT': 2160,
            'DATA_TYPE': 'uint16',
            'DARK_RAW_PATH': r"F:\ZJU\Picture\dark\g3\average_dark.raw",
            'LENS_SHADING_PARAMS_DIR': r"F:\ZJU\Picture\lens shading\new",
            'WB_PARAMS_PATH': r"F:\ZJU\Picture\wb\wb_output",
            'ccm_matrix': np.array([
                [1.7801320111582375, -0.7844420268663381, 0.004310015708100662],
                [-0.24377094860030846, 2.4432181685707977, -1.1994472199704893],
                [-0.4715762768203783, -0.7105721829898775, 2.182148459810256]
            ]),
            'wb_params': {
                "white_balance_gains": {
                    "b_gain": 2.168214315103357,
                    "g_gain": 1.0,
                    "r_gain": 1.3014453071420942
                }
            }
        }
        
        # 加载RAW数据
        print("\n1. Loading RAW data...")
        raw_data = read_raw_image(raw_file, config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'], config['DATA_TYPE'])
        if raw_data is None:
            raise ValueError(f"Failed to load RAW image: {raw_file}")
        
        print(f"  RAW data loaded: {raw_data.shape}, range: {np.min(raw_data)}-{np.max(raw_data)}")
        
        # 加载ISP参数
        print("\n2. Loading ISP parameters...")
        dark_data = load_dark_reference(config['DARK_RAW_PATH'], config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'], config['DATA_TYPE'])
        lens_shading_params = load_correction_parameters(config['LENS_SHADING_PARAMS_DIR'])
        wb_params = load_white_balance_parameters(config['WB_PARAMS_PATH'])
        
        # 获取可用的去马赛克算法
        available_methods = get_available_methods()
        print(f"\n3. Available demosaic methods: {len(available_methods)}")
        for method in available_methods:
            print(f"  - {method}: {get_method_description(method)}")
        
        # 选择要测试的算法（选择前5个进行测试）
        test_methods = available_methods[:5]
        print(f"\n4. Testing methods: {test_methods}")
        
        # 测试不同算法
        results = {}
        for method in test_methods:
            print(f"\n5. Testing method: {method}")
            try:
                result = process_single_image(
                    raw_file=raw_file,
                    dark_data=dark_data,
                    lens_shading_params=lens_shading_params,
                    width=config['IMAGE_WIDTH'],
                    height=config['IMAGE_HEIGHT'],
                    data_type=config['DATA_TYPE'],
                    wb_params=wb_params,
                    ccm_matrix=config['ccm_matrix'],
                    demosaic_method=method
                )
                
                if result['processing_success'] and result['color_img'] is not None:
                    results[method] = result['color_img']
                    print(f"  ✅ {method} completed successfully")
                    
                    # 保存结果
                    output_file = output_dir / f"isp_result_{method}.jpg"
                    cv2.imwrite(str(output_file), result['color_img'])
                    print(f"  Saved: {output_file}")
                else:
                    print(f"  ❌ {method} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"  ❌ {method} failed with exception: {e}")
        
        # 创建对比图
        if len(results) > 1:
            print(f"\n6. Creating comparison plot...")
            create_comparison_plot(results, output_dir)
        
        print(f"\n✅ Test completed!")
        print(f"   Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def create_comparison_plot(results: dict, output_dir: Path):
    """创建对比图"""
    n_methods = len(results)
    if n_methods == 0:
        return
    
    # 计算子图布局
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    fig.suptitle('Demosaic Methods Comparison', fontsize=16, fontweight='bold')
    
    for i, (method, img) in enumerate(results.items()):
        if i < len(axes):
            # 转换为RGB显示
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(f'{method}', fontsize=12)
            axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    comparison_path = output_dir / "demosaic_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  Comparison plot saved: {comparison_path}")

if __name__ == "__main__":
    test_demosaic_methods()
