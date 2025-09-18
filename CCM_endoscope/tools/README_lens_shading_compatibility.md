# 镜头阴影尺寸兼容性检查功能

修改后的ISP.py和invert_ISP.py现在包含智能的镜头阴影尺寸兼容性检查，当尺寸不匹配时会自动跳过镜头阴影矫正。

## 主要修改

### 1. 新增兼容性检查函数

#### `check_lens_shading_compatibility(raw_data, lens_shading_params)`
- 检查镜头阴影参数是否与RAW图像尺寸兼容
- 支持R/G1/G2/B四个通道的独立检查
- 允许±10%的尺寸容差
- 当尺寸严重不匹配时返回False

### 2. ISP.py修改

#### 镜头阴影处理逻辑
```python
# 3. Lens shading correction
if lens_shading_enabled and lens_shading_params is not None:
    print(f"  3. Applying lens shading correction (Bayer per-channel map)...")
    # 检查镜头阴影参数是否与图像尺寸匹配
    if check_lens_shading_compatibility(dark_corrected, lens_shading_params):
        # lens_shading_params 为一个包含 R/G1/G2/B 小图的字典
        lens_corrected = lensshading_correction_bayer(dark_corrected, lens_shading_params, 'rggb')
        print(f"  3. Lens shading correction applied")
    else:
        lens_corrected = dark_corrected.copy()
        print(f"  3. Lens shading correction skipped due to dimension mismatch")
else:
    lens_corrected = dark_corrected.copy()
    print(f"  3. Lens shading correction skipped")
```

### 3. invert_ISP.py修改

#### 逆镜头阴影处理逻辑
```python
# 6. 逆镜头阴影校正
if config['LENS_SHADING_ENABLED'] and config.get('LENS_SHADING_PARAMS_DIR'):
    print("\n6. Applying inverse lens shading correction...")
    lens_shading_params = load_correction_parameters(config['LENS_SHADING_PARAMS_DIR'])
    if lens_shading_params is not None:
        # 检查镜头阴影参数是否与图像尺寸匹配
        if check_lens_shading_compatibility(raw_data, lens_shading_params):
            raw_data = inverse_lens_shading_correction(raw_data, lens_shading_params)
            results['step6_inverse_lens_shading'] = raw_data
        else:
            print("  Skipping lens shading correction due to dimension mismatch")
            results['step6_inverse_lens_shading'] = raw_data  # 保持原始数据
    else:
        print("  Failed to load lens shading parameters, skipping...")
        results['step6_inverse_lens_shading'] = raw_data  # 保持原始数据
else:
    print("\n6. Skipping inverse lens shading correction")
    results['step6_inverse_lens_shading'] = raw_data  # 保持原始数据
```

## 功能特点

### 1. 严格尺寸检查
- 自动计算期望的小图尺寸（RAW尺寸的一半）
- 检查每个通道的尺寸是否完全匹配
- 不允许任何尺寸误差，必须完全匹配

### 2. 严格匹配机制
```python
# 直接判断尺寸是否完全匹配，不允许误差
if actual_h != expected_h or actual_w != expected_w:
    # 尺寸不匹配，跳过矫正
    return False
```

### 3. 详细日志输出
- 显示期望尺寸和实际尺寸
- 明确说明跳过原因
- 提供清晰的错误信息

### 4. 安全处理
- 当尺寸不匹配时，保持原始数据不变
- 不会因为尺寸问题导致程序崩溃
- 提供清晰的错误信息

## 使用场景

### 1. 分辨率切换
- 从1K切换到4K时，镜头阴影参数可能不匹配
- 系统会自动跳过不匹配的矫正
- 确保处理流程正常进行

### 2. 参数更新
- 镜头阴影参数更新后尺寸可能变化
- 系统会检查兼容性并做出相应处理
- 避免因参数不匹配导致的问题

### 3. 批量处理
- 处理不同分辨率的图像时
- 自动适应不同的参数尺寸
- 提高处理成功率

## 测试和验证

### 运行测试脚本
```bash
python test_lens_shading_compatibility.py
```

### 测试内容
1. **完全匹配测试**: 尺寸完全匹配的情况
2. **轻微不匹配测试**: 在容差范围内的差异
3. **严重不匹配测试**: 超出容差范围的差异
4. **空参数测试**: 参数为None的情况
5. **部分通道缺失测试**: 缺少某些通道的情况
6. **4K分辨率测试**: 4K图像的处理

### 预期结果
- 完全匹配：返回True，执行矫正
- 任何不匹配和空参数：返回False，跳过矫正
- 部分通道缺失：返回True，执行矫正（缺失通道使用默认值）

## 配置选项

### 严格匹配设置
```python
# 直接判断尺寸是否完全匹配，不允许误差
if actual_h != expected_h or actual_w != expected_w:
    return False
```

### 通道检查
```python
channels = ['R', 'G1', 'G2', 'B']
for channel in channels:
    if channel in lens_shading_params:
        # 检查该通道的尺寸
```

## 错误处理

### 常见错误情况
1. **参数为None**: 直接返回False
2. **通道缺失**: 跳过缺失的通道
3. **尺寸严重不匹配**: 返回False并输出警告
4. **数据类型错误**: 捕获异常并返回False

### 错误信息示例
```
Warning: lens shading R channel dimension mismatch!
  Expected: 960x540
  Actual: 1920x1080
```

## 性能优化

### 1. 早期检查
- 在处理前进行尺寸检查
- 避免不必要的计算开销
- 提高处理效率

### 2. 严格匹配机制
- 要求完全匹配，确保精度
- 避免因尺寸不匹配导致的问题
- 提供更可靠的处理结果

### 3. 内存优化
- 不匹配时直接跳过，不创建额外数据
- 保持原始数据不变
- 减少内存使用

## 扩展功能

### 1. 严格匹配
- 要求完全匹配，确保精度
- 适应高精度要求
- 提供更可靠的控制

### 2. 更多通道支持
- 可以扩展支持更多通道
- 适应不同的镜头阴影模型
- 提供更好的兼容性

### 3. 自动调整
- 可以自动调整参数尺寸
- 使用插值方法适应不同尺寸
- 提供更智能的处理

## 故障排除

### 常见问题
1. **总是跳过矫正**: 检查参数尺寸是否完全匹配
2. **尺寸计算错误**: 确认RAW图像尺寸正确
3. **参数加载失败**: 检查参数文件路径和格式

### 调试建议
1. 使用测试脚本验证功能
2. 检查日志输出中的尺寸信息
3. 确认参数文件格式正确
4. 验证RAW图像尺寸
5. 确保参数尺寸与RAW图像尺寸完全匹配

## 最佳实践

### 1. 参数管理
- 为不同分辨率准备完全匹配的参数
- 定期检查参数尺寸兼容性
- 使用版本控制管理参数

### 2. 错误处理
- 监控跳过矫正的情况
- 及时更新尺寸不匹配的参数
- 记录处理日志

### 3. 性能监控
- 监控处理时间
- 检查跳过率
- 优化参数管理

## 总结

镜头阴影尺寸兼容性检查功能提供了：

1. **严格检查**: 自动检查参数尺寸完全匹配
2. **安全处理**: 不匹配时安全跳过
3. **详细日志**: 提供清晰的错误信息
4. **严格匹配**: 要求完全匹配，确保精度
5. **性能优化**: 避免不必要的计算

这些功能确保了ISP和逆ISP处理流程的稳定性和可靠性，特别是在处理不同分辨率图像或使用不同参数时。严格的尺寸检查确保了镜头阴影矫正的精度和可靠性。
