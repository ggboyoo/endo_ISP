# 白平衡参数访问错误修复

## 问题描述

在逆ISP处理中，当访问白平衡参数时出现`'white_balance_gains'`键不存在的错误：

```
Error in inverse ISP processing: 'white_balance_gains'
```

## 问题原因

在`invert_ISP.py`中，代码直接访问`config['wb_params']['white_balance_gains']`，但没有检查`white_balance_gains`键是否存在：

```python
# 问题代码
if config.get('wb_params') is not None:
    wb_params = config['wb_params']['white_balance_gains']  # 可能出错
```

## 修复方案

修改了`invert_ISP.py`中的白平衡参数访问逻辑，添加了键存在性检查：

```python
# 修复后的代码
if config.get('wb_params') is not None:
    print("  Using provided white balance parameters")
    wb_params_dict = config['wb_params']
    if 'white_balance_gains' in wb_params_dict:
        wb_params = wb_params_dict['white_balance_gains']
    else:
        # 如果wb_params直接包含增益值
        wb_params = wb_params_dict
elif config.get('WB_PARAMS') is not None:
    print("  Using provided white balance parameters (legacy)")
    wb_params_dict = config['WB_PARAMS']
    if 'white_balance_gains' in wb_params_dict:
        wb_params = wb_params_dict['white_balance_gains']
    else:
        wb_params = wb_params_dict
elif config['WB_PARAMS_PATH']:
    wb_params = load_white_balance_parameters(config['WB_PARAMS_PATH'])
else:
    print("  No white balance parameters provided, skipping...")
    wb_params = None
```

## 支持的参数格式

### 格式1：包含white_balance_gains的结构
```python
wb_params = {
    "white_balance_gains": {
        "b_gain": 2.168214315103357,
        "g_gain": 1.0,
        "r_gain": 1.3014453071420942
    }
}
```

### 格式2：直接包含增益值的结构
```python
wb_params = {
    "b_gain": 2.168214315103357,
    "g_gain": 1.0,
    "r_gain": 1.3014453071420942
}
```

## 修复效果

1. **错误消除**：不再出现`'white_balance_gains'`键不存在的错误
2. **格式兼容**：支持两种不同的白平衡参数格式
3. **向后兼容**：保持对legacy参数的支持
4. **错误处理**：提供清晰的错误信息和日志输出

## 测试验证

创建了`test_wb_params_fix.py`测试脚本，验证：

1. **参数解析**：测试不同格式的白平衡参数解析
2. **配置结构**：验证配置结构的正确性
3. **逆ISP逻辑**：测试逆ISP白平衡处理逻辑

## 使用示例

### 在example_invert_ISP.py中
```python
config = {
    'wb_params': {
        "white_balance_gains": {
            "b_gain": 2.168214315103357,
            "g_gain": 1.0,
            "r_gain": 1.3014453071420942
        }
    }
}
```

### 在invert_ISP.py中
```python
# 现在可以安全地访问白平衡参数
if config.get('wb_params') is not None:
    wb_params_dict = config['wb_params']
    if 'white_balance_gains' in wb_params_dict:
        wb_params = wb_params_dict['white_balance_gains']
    else:
        wb_params = wb_params_dict
```

## 总结

这个修复解决了白平衡参数访问时的键不存在错误，提高了代码的健壮性和兼容性。现在逆ISP处理可以正确处理不同格式的白平衡参数，确保ISP→逆ISP→ISP的循环处理能够正常进行。
