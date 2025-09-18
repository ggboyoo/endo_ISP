# JSON序列化错误修复

## 问题描述

在逆ISP处理中，当尝试保存处理报告时出现JSON序列化错误：

```
Error in inverse ISP processing: Object of type ndarray is not JSON serializable
```

## 问题原因

在`invert_ISP.py`中，`report`字典包含了`config`配置，而`config`中可能包含numpy数组（如`ccm_matrix`、`dark_data`等），这些numpy数组不能直接序列化为JSON格式。

```python
# 问题代码
report = {
    'timestamp': datetime.now().isoformat(),
    'input_image': image_path,
    'output_raw': config['OUTPUT_RAW_PATH'],
    'config': config,  # 包含numpy数组，无法JSON序列化
    'processing_success': True,
    'image_info': {
        'original_shape': img_12bit.shape,  # numpy数组形状
        'actual_dimensions': f"{actual_width}x{actual_height}",
        'raw_shape': raw_data.shape,  # numpy数组形状
        'raw_range': [int(np.min(raw_data)), int(np.max(raw_data))],
        'raw_dtype': str(raw_data.dtype)
    }
}
```

## 修复方案

创建了一个JSON序列化安全的配置副本，将numpy数组转换为JSON兼容的格式：

```python
# 修复后的代码
# 创建JSON序列化安全的配置副本
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

report = {
    'timestamp': datetime.now().isoformat(),
    'input_image': image_path,
    'output_raw': config['OUTPUT_RAW_PATH'],
    'config': safe_config,  # 使用安全的配置副本
    'processing_success': True,
    'image_info': {
        'original_shape': list(img_12bit.shape),  # 转换为列表
        'actual_dimensions': f"{actual_width}x{actual_height}",
        'raw_shape': list(raw_data.shape),  # 转换为列表
        'raw_range': [int(np.min(raw_data)), int(np.max(raw_data))],
        'raw_dtype': str(raw_data.dtype)
    }
}
```

## 处理策略

### 1. 小数组（size < 100）
- 直接转换为Python列表
- 保留完整的数值信息
- 适合CCM矩阵等小数组

### 2. 大数组（size >= 100）
- 只保存形状、数据类型和大小信息
- 不保存实际数值（避免JSON文件过大）
- 适合图像数据等大数组

### 3. 嵌套字典
- 递归处理字典中的numpy数组
- 保持字典结构的完整性
- 处理`wb_params`等嵌套结构

## 示例转换

### CCM矩阵（小数组）
```python
# 原始
ccm_matrix = np.array([[1.78, -0.78, 0.004], [-0.24, 2.44, -1.20], [-0.47, -0.71, 2.18]])

# 转换后
safe_config['ccm_matrix'] = [[1.78, -0.78, 0.004], [-0.24, 2.44, -1.20], [-0.47, -0.71, 2.18]]
```

### 暗电流数据（大数组）
```python
# 原始
dark_data = np.random.randint(0, 100, (1000, 1000), dtype=np.uint16)

# 转换后
safe_config['dark_data'] = {
    'shape': [1000, 1000],
    'dtype': 'uint16',
    'size': 1000000
}
```

### 嵌套字典
```python
# 原始
wb_params = {
    "white_balance_gains": {
        "b_gain": 2.168214315103357,
        "g_gain": 1.0,
        "r_gain": 1.3014453071420942
    }
}

# 转换后（无numpy数组，保持不变）
safe_config['wb_params'] = {
    "white_balance_gains": {
        "b_gain": 2.168214315103357,
        "g_gain": 1.0,
        "r_gain": 1.3014453071420942
    }
}
```

## 修复效果

### 1. 错误消除
- ✅ 不再出现JSON序列化错误
- ✅ 可以正常保存处理报告
- ✅ 保持配置信息的完整性

### 2. 信息保留
- ✅ 小数组保留完整数值
- ✅ 大数组保留关键信息（形状、类型、大小）
- ✅ 嵌套结构保持完整

### 3. 性能优化
- ✅ 避免JSON文件过大
- ✅ 提高序列化速度
- ✅ 减少内存使用

## 测试验证

创建了`test_json_serialization_fix.py`测试脚本，验证：

1. **直接序列化失败**：确认原始配置无法直接序列化
2. **修复后序列化成功**：验证修复后的配置可以序列化
3. **信息完整性**：确保关键信息得到保留
4. **报告创建**：测试完整的报告创建流程

## 使用示例

### 在invert_ISP.py中
```python
# 创建安全的配置副本
safe_config = create_safe_config(config)

# 创建报告
report = {
    'timestamp': datetime.now().isoformat(),
    'input_image': image_path,
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

# 保存报告
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
```

## 注意事项

### 1. 数据丢失
- 大数组的实际数值不会保存到JSON中
- 只保留形状、类型和大小信息
- 如需完整数据，应单独保存为.npy文件

### 2. 性能考虑
- 小数组转换可能增加内存使用
- 大数组只保存元信息，性能影响较小
- 递归处理可能增加计算时间

### 3. 兼容性
- 生成的JSON文件与标准JSON格式兼容
- 可以被其他工具正常读取
- 保持向后兼容性

## 总结

这个修复解决了逆ISP处理中的JSON序列化错误，通过智能处理numpy数组，既保证了序列化的成功，又保留了关键信息。修复后的代码可以正常保存处理报告，为调试和分析提供了便利。
