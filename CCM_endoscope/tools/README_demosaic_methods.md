# 去马赛克算法选择指南

## 概述

现在ISP.py支持多种去马赛克算法，您可以根据需要选择最适合的算法。

## 可用的去马赛克算法

### 1. 基本算法
- **`opencv_bilinear`** (默认) - 双线性插值
  - 速度：最快
  - 质量：一般
  - 适用：实时处理，快速预览

### 2. 高质量算法
- **`opencv_vng`** - VNG算法
  - 速度：中等
  - 质量：好
  - 适用：一般用途，平衡速度和质量

- **`opencv_ea`** - EA算法
  - 速度：中等
  - 质量：好
  - 适用：边缘感知，保持边缘细节

- **`opencv_ahd`** - AHD算法
  - 速度：慢
  - 质量：高
  - 适用：高质量输出，最终处理

- **`opencv_malvar`** - Malvar算法
  - 速度：中等
  - 质量：好
  - 适用：平衡速度和质量

### 3. 高级算法
- **`opencv_dcbi`** - DCBI算法
  - 速度：中等
  - 质量：好
  - 适用：方向性插值，减少伪影

- **`opencv_hqlinear`** - HQ线性算法
  - 速度：中等
  - 质量：高
  - 适用：高质量线性插值

- **`opencv_edgeaware`** - 边缘感知算法
  - 速度：中等
  - 质量：好
  - 适用：保持边缘锐度

### 4. 颜色校正算法
- **`opencv_ccm`** - CCM算法
  - 速度：中等
  - 质量：好
  - 适用：包含颜色校正

- **`opencv_rccm`** - RCCM算法
  - 速度：中等
  - 质量：好
  - 适用：鲁棒颜色校正

- **`opencv_rccm_simple`** - RCCM简单算法
  - 速度：快
  - 质量：好
  - 适用：简化版本

- **`opencv_rccm_advanced`** - RCCM高级算法
  - 速度：慢
  - 质量：高
  - 适用：完整版本

## 使用方法

### 1. 在ISP.py中修改配置

```python
# 在ISP.py的配置部分修改
DEMOSAIC_METHOD = 'opencv_vng'  # 选择您想要的算法
```

### 2. 在函数调用中指定

```python
# 在process_single_image函数调用中
result = process_single_image(
    raw_file=raw_file,
    dark_data=dark_data,
    lens_shading_params=lens_shading_params,
    width=width,
    height=height,
    data_type=data_type,
    wb_params=wb_params,
    ccm_matrix=ccm_matrix,
    # ... 其他参数 ...
    demosaic_output=True  # 启用去马赛克输出
)
```

### 3. 直接使用enhanced_demosaic模块

```python
from enhanced_demosaic import enhanced_demosaic

# 直接调用
result = enhanced_demosaic(
    raw_data=raw_data,
    bayer_pattern='rggb',
    method='opencv_vng',
    apply_rb_swap=True
)
```

## 算法选择建议

### 根据用途选择

1. **实时处理/预览**：
   - `opencv_bilinear` - 最快速度
   - `opencv_rccm_simple` - 快速且质量好

2. **一般用途**：
   - `opencv_vng` - 平衡速度和质量
   - `opencv_malvar` - 质量好
   - `opencv_ea` - 边缘感知

3. **高质量输出**：
   - `opencv_ahd` - 最高质量
   - `opencv_hqlinear` - 高质量线性
   - `opencv_rccm_advanced` - 完整颜色校正

4. **特殊需求**：
   - `opencv_dcbi` - 减少伪影
   - `opencv_edgeaware` - 保持边缘锐度
   - `opencv_ccm` - 包含颜色校正

### 根据图像类型选择

1. **高对比度图像**：
   - `opencv_ahd`
   - `opencv_edgeaware`

2. **低光图像**：
   - `opencv_vng`
   - `opencv_malvar`

3. **彩色图像**：
   - `opencv_rccm`
   - `opencv_ccm`

4. **纹理丰富图像**：
   - `opencv_dcbi`
   - `opencv_hqlinear`

## 性能对比

| 算法 | 速度 | 质量 | 内存使用 | 推荐用途 |
|------|------|------|----------|----------|
| opencv_bilinear | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 实时处理 |
| opencv_vng | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 一般用途 |
| opencv_ea | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 边缘感知 |
| opencv_ahd | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 高质量输出 |
| opencv_malvar | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 平衡选择 |
| opencv_dcbi | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 减少伪影 |
| opencv_hqlinear | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 高质量线性 |
| opencv_edgeaware | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 边缘锐度 |
| opencv_ccm | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 颜色校正 |
| opencv_rccm | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 鲁棒校正 |

## 测试脚本

使用 `test_demosaic_methods.py` 来测试不同算法：

```bash
python test_demosaic_methods.py
```

这将：
1. 加载RAW图像
2. 使用不同算法进行处理
3. 生成对比图像
4. 保存结果到输出目录

## 注意事项

1. **兼容性**：某些算法可能在某些OpenCV版本中不可用
2. **性能**：高质量算法需要更多计算时间
3. **内存**：复杂算法可能需要更多内存
4. **质量**：不同算法在不同类型的图像上表现可能不同

## 建议

1. 先使用 `opencv_bilinear` 进行快速测试
2. 根据图像特点选择合适的算法
3. 使用测试脚本比较不同算法的效果
4. 根据实际需求平衡速度和质量
