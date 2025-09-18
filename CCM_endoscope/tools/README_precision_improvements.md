# 精度改进和过程可逆性优化

本次修改主要针对ISP和逆ISP处理流程，移除了中间过程中的clip操作，确保精度和过程的可逆性。

## 主要修改

### 1. ROI功能Enable参数

#### 新增ROI_ENABLED参数
```python
# 在配置中添加ROI enable参数
'ROI_ENABLED': True,  # 是否启用ROI检测
```

#### 修改PSNR计算函数
```python
def calculate_psnr_circular_roi(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0, 
                               threshold: float = 0.1, roi_enabled: bool = True) -> Tuple[float, Tuple[int, int, int]]:
    # 检测圆形ROI（如果启用）
    if roi_enabled:
        center_x, center_y, radius = detect_circular_roi(img1, threshold)
    else:
        # 如果禁用ROI，使用整个图像
        h, w = img1.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y, w - center_x, h - center_y)
        print(f"ROI detection disabled, using full image: center=({center_x}, {center_y}), radius={radius}")
```

### 2. ISP中间过程精度优化

#### 移除中间clip操作
所有ISP中间处理函数都移除了`np.clip(corrected_data, 0, 4095).astype(np.uint16)`操作，改为只裁剪负值：

```python
# 修改前
corrected = np.clip(corrected, 0, 4095).astype(np.uint16)

# 修改后
corrected = np.clip(corrected, 0, None)  # 只裁剪负值，保持精度
```

#### 涉及的函数
1. **`apply_white_balance_correction_16bit`**
2. **`apply_white_balance_bayer`**
3. **`apply_ccm_16bit`**
4. **`lensshading_correction`**
5. **`lensshading_correction_bayer`**

#### CCM矫正后的clip操作
在CCM矫正后添加了专门的clip操作，然后进行gamma矫正：

```python
# 在CCM矫正后进行clip，然后进行gamma矫正
if ccm_enabled and (ccm_matrix is not None or ccm_matrix_path is not None):
    print(f"  6.5. Clipping CCM corrected data to 12-bit range...")
    color_img_16bit = np.clip(color_img_16bit, 0, 4095)
    print(f"  6.5. CCM corrected data clipped: range {np.min(color_img_16bit)}-{np.max(color_img_16bit)}")
```

### 3. 逆ISP中间过程精度优化

#### 移除中间clip操作
所有逆ISP中间处理函数都移除了clip操作，改为只裁剪负值：

```python
# 修改前
corrected = np.clip(corrected, 0, 4095).astype(np.uint16)

# 修改后
corrected = np.clip(corrected, 0, None)  # 只裁剪负值，保持精度
```

#### 涉及的函数
1. **`inverse_ccm_correction`**
2. **`inverse_white_balance_correction`**
3. **`inverse_white_balance_bayer`**
4. **`inverse_dark_subtraction`**
5. **`inverse_lens_shading_correction`**

#### 最后保存时的clip操作
在`save_raw_data`函数中，在最后保存为raw图时进行clip和uint16转换：

```python
# 在最后保存时进行clip和uint16转换
print(f"  Clipping RAW data to 12-bit range and converting to uint16...")
raw_data_clipped = np.clip(raw_data, 0, 4095).astype(np.uint16)

# 保存为二进制文件
raw_data_clipped.tofile(output_path)
```

## 技术优势

### 1. 精度保持
- **中间过程不量化**：所有中间计算都保持float64精度
- **避免精度损失**：只在必要时进行clip操作
- **保持动态范围**：允许超出12-bit范围的值在中间过程中存在

### 2. 过程可逆性
- **逆ISP可逆性**：中间过程不进行clip，确保逆操作的有效性
- **数据完整性**：保持原始数据的完整信息
- **处理链完整性**：确保ISP→逆ISP→ISP的循环处理

### 3. 灵活性
- **ROI可选**：通过ROI_ENABLED参数控制是否使用ROI检测
- **配置灵活**：可以根据需要调整clip策略
- **向后兼容**：保持原有接口的兼容性

## 处理流程

### ISP处理流程
```
RAW数据 → 暗电流矫正 → 镜头阴影矫正 → 白平衡 → 去马赛克 → CCM矫正 → [CLIP] → 伽马矫正 → 输出
```

### 逆ISP处理流程
```
RGB图像 → 逆伽马矫正 → 逆CCM矫正 → 逆马赛克 → 逆白平衡 → 逆镜头阴影矫正 → 逆暗电流矫正 → [CLIP+uint16] → RAW输出
```

## 配置参数

### ROI控制
```python
'ROI_ENABLED': True,  # 是否启用ROI检测
```

### 处理开关
```python
'LENS_SHADING_ENABLED': False,
'WHITE_BALANCE_ENABLED': True,
'CCM_ENABLED': True,
'GAMMA_CORRECTION_ENABLED': True,
```

## 使用示例

### 启用ROI检测
```python
config = {
    'ROI_ENABLED': True,
    # 其他配置...
}

# PSNR计算会使用ROI检测
psnr, roi_info = calculate_psnr_circular_roi(img1, img2, 255.0, roi_enabled=True)
```

### 禁用ROI检测
```python
config = {
    'ROI_ENABLED': False,
    # 其他配置...
}

# PSNR计算会使用整个图像
psnr, roi_info = calculate_psnr_circular_roi(img1, img2, 255.0, roi_enabled=False)
```

## 性能影响

### 1. 内存使用
- **增加内存使用**：float64比uint16占用更多内存
- **精度提升**：计算精度显著提高
- **可接受范围**：对于大多数应用场景是可接受的

### 2. 计算性能
- **计算精度**：float64计算比uint16更精确
- **处理时间**：可能略微增加，但影响很小
- **质量提升**：显著提高处理质量

### 3. 存储优化
- **最终输出**：仍然保存为uint16格式
- **中间数据**：保持高精度用于计算
- **存储效率**：最终文件大小不变

## 测试验证

### 精度测试
```python
# 测试中间过程精度
def test_precision():
    # 创建测试数据
    test_data = np.random.uniform(0, 5000, (100, 100, 3))
    
    # 测试ISP处理
    result = process_single_image(test_data, ...)
    
    # 验证精度保持
    assert result['color_img'].dtype == np.float64
    assert np.max(result['color_img']) > 4095  # 允许超出12-bit范围
```

### 可逆性测试
```python
# 测试ISP→逆ISP→ISP的循环
def test_reversibility():
    # 原始数据
    original = load_raw_data("test.raw")
    
    # ISP处理
    isp_result = process_single_image(original, ...)
    
    # 逆ISP处理
    invert_result = invert_isp_pipeline(isp_result['color_img'], ...)
    
    # 再次ISP处理
    final_result = process_single_image(invert_result['raw_data'], ...)
    
    # 验证可逆性
    assert np.allclose(isp_result['color_img'], final_result['color_img'], atol=1e-6)
```

## 注意事项

### 1. 数据范围
- **中间过程**：允许超出12-bit范围的值
- **最终输出**：clip到12-bit范围
- **精度保持**：确保计算精度

### 2. 兼容性
- **向后兼容**：保持原有接口
- **配置灵活**：通过参数控制行为
- **错误处理**：保持原有的错误处理机制

### 3. 性能考虑
- **内存使用**：注意内存使用情况
- **计算精度**：权衡精度和性能
- **存储优化**：最终输出仍然优化

## 总结

本次修改实现了：

1. **ROI功能可选**：通过ROI_ENABLED参数控制
2. **精度优化**：移除中间过程的clip操作
3. **可逆性保证**：确保逆ISP过程的有效性
4. **流程优化**：在合适的位置进行clip操作
5. **向后兼容**：保持原有接口和功能

这些改进显著提高了ISP和逆ISP处理的质量和可逆性，为高质量图像处理提供了更好的基础。
