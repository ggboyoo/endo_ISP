#!/usr/bin/env python3
"""
测试JSON序列化修复
"""

import numpy as np
import json
from datetime import datetime

def test_json_serialization():
    """测试JSON序列化修复"""
    print("=== 测试JSON序列化修复 ===")
    
    # 模拟包含numpy数组的配置
    config = {
        'ccm_matrix': np.array([
            [1.7801320111582375, -0.7844420268663381, 0.004310015708100662],
            [-0.24377094860030846, 2.4432181685707977, -1.1994472199704893],
            [-0.4715762768203783, -0.7105721829898775, 2.182148459810256]
        ]),
        'dark_data': np.random.randint(0, 100, (1000, 1000), dtype=np.uint16),
        'wb_params': {
            "white_balance_gains": {
                "b_gain": 2.168214315103357,
                "g_gain": 1.0,
                "r_gain": 1.3014453071420942
            }
        },
        'IMAGE_WIDTH': 3840,
        'IMAGE_HEIGHT': 2160,
        'DATA_TYPE': 'uint16'
    }
    
    print("原始配置包含numpy数组:")
    print(f"  ccm_matrix: {type(config['ccm_matrix'])} {config['ccm_matrix'].shape}")
    print(f"  dark_data: {type(config['dark_data'])} {config['dark_data'].shape}")
    
    # 测试直接JSON序列化（应该失败）
    print("\n--- 测试直接JSON序列化（应该失败）---")
    try:
        json.dumps(config)
        print("✗ 直接序列化应该失败但没有失败")
    except TypeError as e:
        print(f"✓ 直接序列化失败（预期）: {e}")
    
    # 测试修复后的序列化
    print("\n--- 测试修复后的序列化 ---")
    safe_config = {}
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            # 将numpy数组转换为列表或形状信息
            if value.size < 100:  # 小数组转换为列表
                safe_config[key] = value.tolist()
            else:  # 大数组只保存形状和类型信息
                safe_config[key] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'size': int(value.size)
                }
        elif isinstance(value, dict):
            # 递归处理字典中的numpy数组
            safe_value = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    if sub_value.size < 100:
                        safe_value[sub_key] = sub_value.tolist()
                    else:
                        safe_value[sub_key] = {
                            'shape': list(sub_value.shape),
                            'dtype': str(sub_value.dtype),
                            'size': int(sub_value.size)
                        }
                else:
                    safe_value[sub_key] = sub_value
            safe_config[key] = safe_value
        else:
            safe_config[key] = value
    
    print("修复后的配置:")
    print(f"  ccm_matrix: {type(safe_config['ccm_matrix'])}")
    print(f"  dark_data: {type(safe_config['dark_data'])}")
    print(f"  wb_params: {type(safe_config['wb_params'])}")
    
    # 测试JSON序列化
    try:
        json_str = json.dumps(safe_config, indent=2)
        print("✓ JSON序列化成功")
        print(f"JSON长度: {len(json_str)} 字符")
        
        # 测试反序列化
        parsed_config = json.loads(json_str)
        print("✓ JSON反序列化成功")
        
        # 验证关键信息
        assert safe_config['ccm_matrix'] == parsed_config['ccm_matrix']
        assert safe_config['dark_data']['shape'] == parsed_config['dark_data']['shape']
        assert safe_config['dark_data']['dtype'] == parsed_config['dark_data']['dtype']
        print("✓ 关键信息验证通过")
        
    except Exception as e:
        print(f"✗ JSON序列化失败: {e}")
        return False
    
    return True

def test_report_creation():
    """测试报告创建"""
    print("\n=== 测试报告创建 ===")
    
    # 模拟图像数据
    img_12bit = np.random.randint(0, 4095, (2160, 3840, 3), dtype=np.uint16)
    raw_data = np.random.randint(0, 4095, (2160, 3840), dtype=np.uint16)
    actual_width, actual_height = 3840, 2160
    
    # 模拟配置
    config = {
        'ccm_matrix': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'dark_data': np.random.randint(0, 100, (100, 100), dtype=np.uint16),
        'OUTPUT_RAW_PATH': 'test_output.raw'
    }
    
    # 创建报告
    safe_config = {}
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            if value.size < 100:
                safe_config[key] = value.tolist()
            else:
                safe_config[key] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'size': int(value.size)
                }
        else:
            safe_config[key] = value
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_image': 'test_input.png',
        'output_raw': config['OUTPUT_RAW_PATH'],
        'config': safe_config,
        'processing_success': True,
        'image_info': {
            'original_shape': list(img_12bit.shape),
            'actual_dimensions': f"{actual_width}x{actual_height}",
            'raw_shape': list(raw_data.shape),
            'raw_range': [int(np.min(raw_data)), int(np.max(raw_data))],
            'raw_dtype': str(raw_data.dtype)
        }
    }
    
    # 测试JSON序列化
    try:
        json_str = json.dumps(report, indent=2)
        print("✓ 报告JSON序列化成功")
        
        # 保存到文件测试
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("✓ 报告文件保存成功")
        
        # 清理测试文件
        import os
        if os.path.exists('test_report.json'):
            os.remove('test_report.json')
            print("✓ 测试文件清理完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 报告创建失败: {e}")
        return False

if __name__ == "__main__":
    success1 = test_json_serialization()
    success2 = test_report_creation()
    
    if success1 and success2:
        print("\n🎉 所有JSON序列化测试通过！")
    else:
        print("\n❌ 部分测试失败")
