# 简单去马赛克算法 (demosaic_easy)

## 概述

`demosaic_easy` 是一个最简单的去马赛克函数，使用2x2块的方式处理Bayer模式图像。每个2x2块内的4个像素共享相同的RGB值。

## 算法原理

### 2x2块处理方式

对于RGGB模式的Bayer图像：
```
R G
G B
```

算法步骤：
1. 取R和B的原始Bayer值
2. 取G1和G2的平均值作为G值
3. 将计算出的RGB值赋给2x2块内的所有4个像素

### 支持的Bayer模式

- **RGGB**: `R G` / `G B`
- **BGGR**: `B G` / `G R`  
- **GRBG**: `G R` / `B G`
- **GBRG**: `G B` / `R G`

## 使用方法

### 1. 基本使用

```python
from demosaic_easy import demosaic_easy
import numpy as np

# 加载RAW数据
raw_data = np.load('image.raw')  # 假设是 (H, W) 的数组

# 去马赛克
color_image = demosaic_easy(raw_data, bayer_pattern='rggb')

if color_image is not None:
    print(f"去马赛克完成: {color_image.shape}")
    # color_image 是 (H, W, 3) 的BGR格式图像
```

### 2. 集成到ISP处理

```python
from ISP_easy_demosaic import process_single_image_easy

# 处理单个图像
result = process_single_image_easy(
    raw_file="path/to/image.raw",
    dark_data=dark_data,
    lens_shading_params=lens_params,
    width=3840,
    height=2160,
    data_type='uint16',
    wb_params=wb_params,
    ccm_matrix=ccm_matrix,
    bayer_pattern='rggb'
)

if result['processing_success']:
    final_image = result['color_img']  # 最终处理结果
```

### 3. 测试函数

```python
from demosaic_easy import test_demosaic_easy

# 运行测试
test_demosaic_easy()
```

## 函数参数

### demosaic_easy()

```python
def demosaic_easy(raw_data: np.ndarray, bayer_pattern: str = 'rggb') -> Optional[np.ndarray]:
```

**参数:**
- `raw_data`: RAW数据数组 (H, W)
- `bayer_pattern`: Bayer模式 ('rggb', 'bggr', 'grbg', 'gbrg')

**返回:**
- 成功: 去马赛克后的彩色图像 (H, W, 3)
- 失败: None

## 算法特点

### 优点
1. **简单快速**: 算法简单，处理速度快
2. **内存友好**: 不需要复杂的插值计算
3. **易于理解**: 逻辑清晰，便于调试
4. **稳定可靠**: 不会产生复杂的伪影

### 缺点
1. **分辨率减半**: 2x2块共享RGB值，实际分辨率减半
2. **细节丢失**: 没有插值，细节信息可能丢失
3. **边缘效应**: 块状效应可能在某些图像中明显

## 适用场景

### 适合使用
1. **快速预览**: 需要快速查看图像内容
2. **低计算资源**: 计算能力有限的环境
3. **调试测试**: 算法开发和调试阶段
4. **简单应用**: 对图像质量要求不高的场景

### 不适合使用
1. **高质量输出**: 需要高质量图像的场景
2. **细节保留**: 需要保留图像细节的场景
3. **专业应用**: 专业图像处理应用

## 性能对比

| 特性 | demosaic_easy | OpenCV双线性 | OpenCV VNG |
|------|---------------|--------------|------------|
| 速度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 质量 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 内存使用 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 分辨率 | 1/4 | 1:1 | 1:1 |
| 复杂度 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## 示例代码

### 完整处理流程

```python
#!/usr/bin/env python3
import numpy as np
import cv2
from demosaic_easy import demosaic_easy

def process_raw_image(raw_file, width, height):
    """完整的RAW图像处理流程"""
    
    # 1. 加载RAW数据
    raw_data = np.fromfile(raw_file, dtype=np.uint16).reshape(height, width)
    
    # 2. 去马赛克
    color_image = demosaic_easy(raw_data, 'rggb')
    
    if color_image is not None:
        # 3. 转换为8位显示
        display_image = (color_image / 16).astype(np.uint8)
        
        # 4. 保存结果
        cv2.imwrite('result.jpg', display_image)
        print("处理完成!")
        
        return display_image
    
    return None

# 使用示例
if __name__ == "__main__":
    result = process_raw_image('image.raw', 3840, 2160)
    if result is not None:
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

## 注意事项

1. **图像尺寸**: 确保图像尺寸是偶数，否则会被裁剪
2. **数据类型**: 支持uint16格式的RAW数据
3. **内存使用**: 输出图像是输入图像的3倍大小（RGB）
4. **Bayer模式**: 确保Bayer模式参数正确

## 文件说明

- `demosaic_easy.py`: 核心去马赛克函数
- `test_easy_demosaic.py`: 测试脚本
- `ISP_easy_demosaic.py`: 集成到ISP处理的完整脚本
- `README_easy_demosaic.md`: 使用说明文档
