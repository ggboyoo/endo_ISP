# Inverse ISP Processing Script

## 概述

`invert_ISP.py` 是一个逆ISP（图像信号处理）脚本，用于将处理过的sRGB图像逆向转换为RAW数据。该脚本实现了完整的逆ISP处理流程，包括逆伽马校正、逆CCM变换、逆白平衡和逆马赛克等步骤。

## 功能特性

- **sRGB到12bit转换**: 将任意格式的sRGB图像（jpg、png等）转换为12bit数据，存储在16bit容器中
- **逆伽马校正**: 应用逆伽马变换，将非线性sRGB值转换为线性值
- **逆CCM变换**: 根据提供的CCM矩阵参数进行逆颜色校正
- **逆白平衡**: 应用逆白平衡校正
- **逆马赛克**: 将RGB图像转换为Bayer RAW格式
- **RAW输出**: 保存为.raw格式的RAW数据文件

## 使用方法

### 基本用法

```bash
python invert_ISP.py --input input_image.jpg --output output.raw
```

### 完整参数

```bash
python invert_ISP.py \
    --input input_image.jpg \
    --output output.raw \
    --width 3840 \
    --height 2160 \
    --bayer rggb \
    --ccm ccm_matrix.json \
    --wb wb_parameters.json \
    --gamma 2.2
```

### 参数说明

- `--input, -i`: 输入sRGB图像路径（必需）
- `--output, -o`: 输出RAW文件路径（必需）
- `--width, -w`: 图像宽度（默认：3840）
- `--height, -h`: 图像高度（默认：2160）
- `--bayer, -b`: Bayer模式，可选值：rggb, bggr, grbg, gbrg（默认：rggb）
- `--ccm`: CCM矩阵文件路径（可选）
- `--wb`: 白平衡参数文件路径（可选）
- `--gamma`: 伽马值（默认：2.2）
- `--no-save-intermediate`: 不保存中间结果
- `--verbose`: 详细输出

## 处理流程

1. **图像加载**: 读取sRGB图像并调整到目标尺寸
2. **12bit转换**: 将8bit图像转换为12bit数据（存储在16bit容器中）
3. **逆伽马校正**: 应用逆伽马变换
4. **逆CCM校正**: 根据CCM矩阵进行逆颜色校正（如果提供）
5. **逆白平衡**: 应用逆白平衡校正（如果提供参数）
6. **逆马赛克**: 将RGB图像转换为Bayer RAW格式
7. **RAW保存**: 保存为.raw格式文件

## 输入文件格式

### CCM矩阵文件（JSON格式）

```json
{
    "ccm_matrix": [
        [1.2, -0.1, 0.05],
        [-0.05, 1.1, -0.02],
        [0.01, -0.08, 1.15]
    ],
    "ccm_type": "linear3x3"
}
```

### 白平衡参数文件（JSON格式）

```json
{
    "white_balance_gains": {
        "r_gain": 1.2,
        "g_gain": 1.0,
        "b_gain": 0.9
    }
}
```

## 输出文件

- **RAW文件**: 主要的输出文件，包含Bayer RAW数据
- **中间结果图像**: 如果启用，会保存每个处理步骤的中间结果
- **处理报告**: JSON格式的处理报告，包含处理参数和结果信息

## 示例

### 基本转换

```bash
# 将JPG图像转换为RAW
python invert_ISP.py --input image.jpg --output image.raw
```

### 完整处理流程

```bash
# 使用CCM和白平衡参数进行完整转换
python invert_ISP.py \
    --input processed_image.jpg \
    --output reconstructed.raw \
    --width 3840 \
    --height 2160 \
    --bayer rggb \
    --ccm ccm_output/ccm_matrix.json \
    --wb wb_output/wb_parameters.json \
    --gamma 2.2
```

### 批量处理

```bash
# 处理多个图像
for img in *.jpg; do
    python invert_ISP.py --input "$img" --output "${img%.jpg}.raw"
done
```

## 注意事项

1. **图像尺寸**: 确保输入图像尺寸与指定的宽度和高度匹配，否则会自动调整
2. **Bayer模式**: 选择正确的Bayer模式以确保逆马赛克的正确性
3. **参数文件**: CCM矩阵和白平衡参数文件必须符合指定的JSON格式
4. **数据范围**: 输出RAW数据为12bit，存储在16bit容器中，范围为0-4095
5. **文件格式**: 输出RAW文件为二进制格式，使用uint16数据类型

## 错误处理

脚本包含完整的错误处理机制：

- 文件不存在或无法读取
- 参数文件格式错误
- 矩阵运算错误（如奇异矩阵）
- 图像处理错误

所有错误都会在控制台输出，并记录在处理报告中。

## 依赖项

- numpy
- opencv-python
- pathlib
- json
- datetime
- argparse

## 版本信息

- 版本: 1.0
- 创建日期: 2025年1月
- 兼容性: Python 3.6+
