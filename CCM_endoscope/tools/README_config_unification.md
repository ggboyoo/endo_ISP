# ISP和逆ISP配置统一化

本次修改将ISP和逆ISP处理使用相同的配置参数，确保处理过程的一致性和可逆性。

## 主要修改

### 1. 配置参数统一

#### 修改前的问题
- ISP使用`config`参数
- 逆ISP使用`INVERT_CONFIG`（来自`invert_ISP.py`的`DEFAULT_CONFIG`）
- 两个配置系统独立，参数不一致

#### 修改后的解决方案
- 逆ISP现在使用与ISP相同的配置参数
- 通过`invert_config.update()`将ISP配置传递给逆ISP
- 确保所有处理开关和参数路径一致

### 2. 配置传递机制

#### 在`example_invert_ISP.py`中
```python
# 3. 逆ISP处理
print("\n3. Inverse ISP processing...")
invert_config = INVERT_CONFIG.copy()
invert_config.update({
    'INPUT_IMAGE_PATH': str(output_dir / "temp_isp_result.png"),
    'OUTPUT_RAW_PATH': str(output_dir / "reconstructed.raw"),
    'IMAGE_WIDTH': config['IMAGE_WIDTH'],
    'IMAGE_HEIGHT': config['IMAGE_HEIGHT'],
    'DATA_TYPE': config['DATA_TYPE'],
    'BAYER_PATTERN': 'rggb',
    'SAVE_INTERMEDIATE': False,
    'DISPLAY_RAW_GRAYSCALE': False,
    'SAVE_RAW_GRAYSCALE': False,
    'CREATE_COMPARISON_PLOT': False,
    # 使用与ISP相同的处理开关
    'DARK_SUBTRACTION_ENABLED': config.get('DARK_SUBTRACTION_ENABLED', True),
    'LENS_SHADING_ENABLED': config.get('LENS_SHADING_ENABLED', True),
    'WHITE_BALANCE_ENABLED': config.get('WHITE_BALANCE_ENABLED', True),
    'CCM_ENABLED': config.get('CCM_ENABLED', True),
    'GAMMA_CORRECTION_ENABLED': config.get('GAMMA_CORRECTION_ENABLED', True),
    'GAMMA_VALUE': config.get('GAMMA_VALUE', 2.2),
    # 使用与ISP相同的参数路径
    'DARK_RAW_PATH': config.get('DARK_RAW_PATH'),
    'LENS_SHADING_PARAMS_DIR': config.get('LENS_SHADING_PARAMS_DIR'),
    'WB_PARAMS_PATH': config.get('WB_PARAMS_PATH'),
    'CCM_MATRIX_PATH': config.get('CCM_MATRIX_PATH'),
    # 使用与ISP相同的直接参数
    'ccm_matrix': config.get('ccm_matrix'),
    'wb_params': config.get('wb_params'),
    'dark_data': config.get('dark_data'),
    'lens_shading_params': config.get('lens_shading_params')
})
```

### 3. 逆ISP参数使用优化

#### CCM矩阵参数
```python
# 修改前
if config.get('CCM_MATRIX') is not None:
    ccm_matrix = np.array(config['CCM_MATRIX'])

# 修改后
if config.get('ccm_matrix') is not None:
    print("  Using provided CCM matrix")
    ccm_matrix = np.array(config['ccm_matrix'])
elif config.get('CCM_MATRIX') is not None:
    print("  Using provided CCM matrix (legacy)")
    ccm_matrix = np.array(config['CCM_MATRIX'])
elif config['CCM_MATRIX_PATH']:
    ccm_matrix, matrix_type = load_ccm_matrix(config['CCM_MATRIX_PATH'])
```

#### 白平衡参数
```python
# 修改前
if config.get('WB_PARAMS') is not None:
    wb_params = config['WB_PARAMS']['white_balance_gains']

# 修改后
if config.get('wb_params') is not None:
    print("  Using provided white balance parameters")
    wb_params = config['wb_params']['white_balance_gains']
elif config.get('WB_PARAMS') is not None:
    print("  Using provided white balance parameters (legacy)")
    wb_params = config['WB_PARAMS']['white_balance_gains']
elif config['WB_PARAMS_PATH']:
    wb_params = load_white_balance_parameters(config['WB_PARAMS_PATH'])
```

#### 镜头阴影参数
```python
# 修改前
if config.get('LENS_SHADING_PARAMS_DIR'):
    lens_shading_params = load_correction_parameters(config['LENS_SHADING_PARAMS_DIR'])

# 修改后
if config.get('lens_shading_params') is not None:
    print("  Using provided lens shading parameters")
    lens_shading_params = config['lens_shading_params']
elif config.get('LENS_SHADING_PARAMS_DIR'):
    lens_shading_params = load_correction_parameters(config['LENS_SHADING_PARAMS_DIR'])
else:
    print("  No lens shading parameters provided, skipping...")
    lens_shading_params = None
```

#### 暗电流参数
```python
# 修改前
if config['DARK_SUBTRACTION_ENABLED'] and config.get('DARK_RAW_PATH'):
    dark_data = load_dark_reference(config['DARK_RAW_PATH'], ...)

# 修改后
if config['DARK_SUBTRACTION_ENABLED']:
    if config.get('dark_data') is not None:
        print("  Using provided dark data")
        dark_data = config['dark_data']
    elif config.get('DARK_RAW_PATH'):
        dark_data = load_dark_reference(config['DARK_RAW_PATH'], ...)
    else:
        print("  No dark data provided, skipping...")
        dark_data = None
```

## 配置参数映射

### 1. 处理开关参数
| ISP参数 | 逆ISP参数 | 说明 |
|---------|-----------|------|
| `DARK_SUBTRACTION_ENABLED` | `DARK_SUBTRACTION_ENABLED` | 暗电流校正开关 |
| `LENS_SHADING_ENABLED` | `LENS_SHADING_ENABLED` | 镜头阴影校正开关 |
| `WHITE_BALANCE_ENABLED` | `WHITE_BALANCE_ENABLED` | 白平衡校正开关 |
| `CCM_ENABLED` | `CCM_ENABLED` | CCM校正开关 |
| `GAMMA_CORRECTION_ENABLED` | `GAMMA_CORRECTION_ENABLED` | 伽马校正开关 |

### 2. 参数路径
| ISP参数 | 逆ISP参数 | 说明 |
|---------|-----------|------|
| `DARK_RAW_PATH` | `DARK_RAW_PATH` | 暗电流文件路径 |
| `LENS_SHADING_PARAMS_DIR` | `LENS_SHADING_PARAMS_DIR` | 镜头阴影参数目录 |
| `WB_PARAMS_PATH` | `WB_PARAMS_PATH` | 白平衡参数文件路径 |
| `CCM_MATRIX_PATH` | `CCM_MATRIX_PATH` | CCM矩阵文件路径 |

### 3. 直接参数
| ISP参数 | 逆ISP参数 | 说明 |
|---------|-----------|------|
| `dark_data` | `dark_data` | 暗电流数据数组 |
| `lens_shading_params` | `lens_shading_params` | 镜头阴影参数字典 |
| `wb_params` | `wb_params` | 白平衡参数字典 |
| `ccm_matrix` | `ccm_matrix` | CCM矩阵数组 |

### 4. 图像参数
| ISP参数 | 逆ISP参数 | 说明 |
|---------|-----------|------|
| `IMAGE_WIDTH` | `IMAGE_WIDTH` | 图像宽度 |
| `IMAGE_HEIGHT` | `IMAGE_HEIGHT` | 图像高度 |
| `DATA_TYPE` | `DATA_TYPE` | 数据类型 |
| `GAMMA_VALUE` | `GAMMA_VALUE` | 伽马值 |

## 优势特点

### 1. 一致性
- **参数统一**：ISP和逆ISP使用相同的配置参数
- **处理一致**：确保处理开关和参数路径一致
- **结果可逆**：保证ISP→逆ISP→ISP的循环处理

### 2. 灵活性
- **向后兼容**：支持legacy参数名称
- **多种方式**：支持直接参数和文件路径两种方式
- **优先级明确**：直接参数优先于文件路径

### 3. 可维护性
- **配置集中**：所有配置在一个地方管理
- **参数映射**：清晰的参数映射关系
- **错误处理**：完善的错误处理和日志输出

## 使用示例

### 基本配置
```python
config = {
    # 图像参数
    'IMAGE_WIDTH': 3840,
    'IMAGE_HEIGHT': 2160,
    'DATA_TYPE': 'uint16',
    
    # 处理开关
    'DARK_SUBTRACTION_ENABLED': True,
    'LENS_SHADING_ENABLED': False,
    'WHITE_BALANCE_ENABLED': True,
    'CCM_ENABLED': True,
    'GAMMA_CORRECTION_ENABLED': True,
    'GAMMA_VALUE': 2.2,
    
    # 参数路径
    'DARK_RAW_PATH': "dark.raw",
    'LENS_SHADING_PARAMS_DIR': "lens_shading",
    'WB_PARAMS_PATH': "wb_params.json",
    'CCM_MATRIX_PATH': "ccm_matrix.json",
    
    # 直接参数（优先使用）
    'dark_data': dark_array,
    'lens_shading_params': lens_params_dict,
    'wb_params': wb_params_dict,
    'ccm_matrix': ccm_matrix_array
}
```

### ISP处理
```python
isp_result = process_single_image(
    raw_file=raw_data,
    dark_data=config.get('dark_data'),
    lens_shading_params=config.get('lens_shading_params'),
    width=config['IMAGE_WIDTH'],
    height=config['IMAGE_HEIGHT'],
    data_type=config['DATA_TYPE'],
    wb_params=config.get('wb_params'),
    dark_subtraction_enabled=config.get('DARK_SUBTRACTION_ENABLED', True),
    lens_shading_enabled=config.get('LENS_SHADING_ENABLED', True),
    white_balance_enabled=config.get('WHITE_BALANCE_ENABLED', True),
    ccm_enabled=config.get('CCM_ENABLED', True),
    ccm_matrix_path=config.get('CCM_MATRIX_PATH'),
    ccm_matrix=config.get('ccm_matrix'),
    gamma_correction_enabled=config.get('GAMMA_CORRECTION_ENABLED', True),
    gamma_value=config.get('GAMMA_VALUE', 2.2)
)
```

### 逆ISP处理
```python
invert_config = INVERT_CONFIG.copy()
invert_config.update({
    'INPUT_IMAGE_PATH': "temp_isp_result.png",
    'OUTPUT_RAW_PATH': "reconstructed.raw",
    'IMAGE_WIDTH': config['IMAGE_WIDTH'],
    'IMAGE_HEIGHT': config['IMAGE_HEIGHT'],
    'DATA_TYPE': config['DATA_TYPE'],
    'BAYER_PATTERN': 'rggb',
    # 使用与ISP相同的处理开关和参数
    'DARK_SUBTRACTION_ENABLED': config.get('DARK_SUBTRACTION_ENABLED', True),
    'LENS_SHADING_ENABLED': config.get('LENS_SHADING_ENABLED', True),
    'WHITE_BALANCE_ENABLED': config.get('WHITE_BALANCE_ENABLED', True),
    'CCM_ENABLED': config.get('CCM_ENABLED', True),
    'GAMMA_CORRECTION_ENABLED': config.get('GAMMA_CORRECTION_ENABLED', True),
    'GAMMA_VALUE': config.get('GAMMA_VALUE', 2.2),
    'dark_data': config.get('dark_data'),
    'lens_shading_params': config.get('lens_shading_params'),
    'wb_params': config.get('wb_params'),
    'ccm_matrix': config.get('ccm_matrix')
})

invert_result = invert_isp_pipeline("temp_isp_result.png", invert_config)
```

## 参数优先级

### 1. 直接参数优先
- `ccm_matrix` > `CCM_MATRIX` > `CCM_MATRIX_PATH`
- `wb_params` > `WB_PARAMS` > `WB_PARAMS_PATH`
- `lens_shading_params` > `LENS_SHADING_PARAMS_DIR`
- `dark_data` > `DARK_RAW_PATH`

### 2. 处理逻辑
```python
# CCM矩阵
if config.get('ccm_matrix') is not None:
    # 使用直接提供的CCM矩阵
    ccm_matrix = np.array(config['ccm_matrix'])
elif config.get('CCM_MATRIX') is not None:
    # 使用legacy CCM矩阵
    ccm_matrix = np.array(config['CCM_MATRIX'])
elif config['CCM_MATRIX_PATH']:
    # 从文件加载CCM矩阵
    ccm_matrix, matrix_type = load_ccm_matrix(config['CCM_MATRIX_PATH'])
else:
    # 没有CCM矩阵，跳过
    ccm_matrix = None
```

## 错误处理

### 1. 参数缺失
- 提供清晰的错误信息
- 自动跳过缺失的参数
- 保持处理流程的连续性

### 2. 参数格式错误
- 验证参数格式
- 提供格式转换
- 记录错误日志

### 3. 文件加载失败
- 尝试多种加载方式
- 提供fallback选项
- 记录加载状态

## 测试验证

### 1. 配置一致性测试
```python
def test_config_consistency():
    # 测试ISP和逆ISP使用相同配置
    config = create_test_config()
    
    # ISP处理
    isp_result = process_single_image(..., **config)
    
    # 逆ISP处理
    invert_config = create_invert_config(config)
    invert_result = invert_isp_pipeline(..., invert_config)
    
    # 验证配置一致性
    assert invert_config['CCM_ENABLED'] == config['CCM_ENABLED']
    assert invert_config['WHITE_BALANCE_ENABLED'] == config['WHITE_BALANCE_ENABLED']
    # ... 其他参数验证
```

### 2. 参数优先级测试
```python
def test_parameter_priority():
    # 测试直接参数优先于文件路径
    config = {
        'ccm_matrix': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'CCM_MATRIX_PATH': "other_ccm.json"
    }
    
    # 应该使用ccm_matrix而不是CCM_MATRIX_PATH
    result = process_with_config(config)
    assert result['ccm_source'] == 'direct_matrix'
```

## 总结

配置统一化实现了：

1. **参数一致性**：ISP和逆ISP使用相同的配置参数
2. **处理一致性**：确保处理开关和参数路径一致
3. **可逆性保证**：保证ISP→逆ISP→ISP的循环处理
4. **向后兼容**：支持legacy参数名称
5. **灵活性**：支持直接参数和文件路径两种方式
6. **可维护性**：配置集中管理，参数映射清晰

这些改进确保了ISP和逆ISP处理的一致性和可逆性，为高质量图像处理提供了更好的基础。
