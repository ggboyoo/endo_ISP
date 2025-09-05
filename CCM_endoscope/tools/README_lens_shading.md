# Lens Shading Analysis Tool

这个脚本用于分析RAW图像，计算相机的镜头阴影校正矩阵。使用网格法分别分析RGGB四个通道，然后融合得到完整的矫正矩阵。

## 功能特点

- 📁 **RAW文件读取**: 调用 `raw_reader.py` 功能读取RAW图像
- 🌑 **暗点平矫正**: 支持暗电流校正，提高图像质量
- 🔴🟢🔵 **RGGB通道分离**: 自动分离红、绿1、绿2、蓝四个通道
- 📊 **网格法分析**: 使用可配置的网格大小进行亮度分析
- 🎯 **镜头阴影检测**: 识别图像边缘的亮度衰减
- 📈 **矫正矩阵计算**: 生成每个通道的校正系数矩阵
- 🔄 **插值扩展**: 将网格矫正矩阵插值到全图像尺寸
- 🖼️ **图像矫正**: 显示和保存矫正后的完整图像
- 📊 **可视化分析**: 生成详细的分析图表和对比图
- 💾 **多格式输出**: 保存矫正矩阵、分析结果和矫正图像

## 使用方法

### 1. 配置参数

在脚本顶部修改配置区域：

```python
# 输入路径配置
INPUT_PATH = r"F:\ZJU\Picture\lens_shading"  # 待分析的RAW图像文件夹路径
# 或者单个文件路径，例如: r"F:\ZJU\Picture\lens_shading\test_image.raw"

# 暗点平矫正配置
DARK_RAW_PATH = r"F:\ZJU\Picture\dark\g9\average_dark_10frames_20241201_143022.raw"  # 暗电流图像路径
ENABLE_DARK_CORRECTION = True  # 是否启用暗点平矫正

# 图像参数配置
IMAGE_WIDTH = 3840      # 图像宽度
IMAGE_HEIGHT = 2160     # 图像高度
DATA_TYPE = 'uint16'    # 数据类型

# Lens Shading分析配置
GRID_SIZE = 32         # 网格大小（建议32x32或64x64）
CHANNEL_NAMES = ['R', 'G1', 'G2', 'B']  # RGGB四个通道名称
MIN_VALID_VALUE = 100  # 最小有效像素值（避免过暗区域）
MAX_VALID_VALUE = 65000  # 最大有效像素值（避免过亮区域）

# 输出配置
OUTPUT_DIRECTORY = None  # 输出目录（None为自动生成）
GENERATE_PLOTS = True   # 是否生成分析图表
SAVE_PLOTS = True       # 是否保存图表文件
SAVE_CORRECTION_MATRIX = True  # 是否保存矫正矩阵
SAVE_CORRECTED_IMAGES = True   # 是否保存矫正后的图像
```

### 2. 运行脚本

```bash
python lens_shading.py
```

## 核心功能

### 1. 暗点平矫正

脚本支持暗电流校正，在镜头阴影分析之前先进行暗点平矫正：

```python
def apply_dark_correction(raw_data, dark_data):
    # 减去暗电流参考图像
    corrected_data = raw_data.astype(np.float64) - dark_data.astype(np.float64)
    # 裁剪负值为0
    corrected_data = np.clip(corrected_data, 0, None)
    return corrected_data
```

**暗点平矫正特点：**
- **暗电流参考**: 使用指定的暗电流图像作为参考
- **像素级校正**: 逐像素减去暗电流值
- **负值处理**: 自动将负值裁剪为0，避免无效数据

### 2. RGGB通道分离

脚本自动将RAW图像分离为四个通道：

```python
# RGGB pattern layout:
# R  G  R  G  R  G ...
# G  B  G  B  G  B ...
# R  G  R  G  R  G ...
# G  B  G  B  G  B ...

# Extract channels
R_channel = raw_data[0::2, 0::2]      # Even rows, even columns
G1_channel = raw_data[0::2, 1::2]     # Even rows, odd columns  
G2_channel = raw_data[1::2, 0::2]     # Odd rows, even columns
B_channel = raw_data[1::2, 1::2]      # Odd rows, odd columns
```

**通道分离结果：**
- **R通道**: 红色像素，尺寸为 (H/2, W/2)
- **G1通道**: 绿色像素1，尺寸为 (H/2, W/2)
- **G2通道**: 绿色像素2，尺寸为 (H/2, W/2)
- **B通道**: 蓝色像素，尺寸为 (H/2, W/2)

### 2. 网格法分析

对每个通道使用网格法进行亮度分析：

```python
def create_grid_analysis(channel_data, grid_size, min_valid, max_valid):
    # 将图像分割为 grid_size x grid_size 的网格
    # 计算每个网格的平均亮度
    # 过滤无效像素（过暗或过亮）
    # 返回网格平均亮度和有效像素计数
```

**网格分析特点：**
- **可配置网格大小**: 建议32x32或64x64像素
- **智能像素过滤**: 避免过暗和过亮区域的影响
- **统计信息**: 每个网格的平均亮度和有效像素数量

### 3. 镜头阴影校正计算

基于网格分析结果计算校正系数：

```python
def calculate_lens_shading_correction(grid_means, reference_value=None):
    # 使用中心区域作为参考亮度
    # 计算每个网格的校正系数: correction = reference / grid_mean
    # 限制校正系数在合理范围内 (0.5 - 2.0)
```

**校正原理：**
- **参考值选择**: 默认使用图像中心作为参考亮度
- **校正系数**: 每个像素的校正系数 = 参考亮度 / 当前亮度
- **系数限制**: 避免过度校正，限制在0.5-2.0范围内

### 4. 插值扩展

将网格矫正矩阵插值到全图像尺寸：

```python
def interpolate_correction_matrix(correction_matrix, target_height, target_width):
    # 使用三次样条插值
    # 从网格尺寸扩展到全图像尺寸
    # 保持校正的平滑性
```

**插值方法：**
- **三次样条插值**: 使用 `scipy.interpolate.RectBivariateSpline`
- **平滑过渡**: 确保网格间的平滑过渡
- **全尺寸输出**: 生成与原始图像相同尺寸的校正矩阵

### 5. 图像矫正与重建

应用镜头阴影矫正并重建完整的矫正图像：

```python
def apply_lens_shading_correction(channel_data, correction_matrix):
    # 应用矫正系数
    corrected_data = channel_data * correction_matrix
    return corrected_data

def reconstruct_corrected_image(channels, corrections):
    # 将矫正后的通道重新组合为完整图像
    # 恢复RGGB拜尔模式
```

**图像矫正特点：**
- **通道级矫正**: 每个通道独立应用矫正系数
- **完整重建**: 将矫正后的通道重新组合为完整图像
- **拜尔模式保持**: 保持原始的RGGB拜尔模式结构

## 输出结果

### 1. 分析图表

脚本生成多种分析图表：

#### 通道分析图表（4x5）
- **第1列**: 原始通道数据（灰度图）
- **第2列**: 网格平均亮度（viridis色彩图）
- **第3列**: 网格矫正矩阵（plasma色彩图）
- **第4列**: 全尺寸矫正矩阵（plasma色彩图）
- **第5列**: 矫正后的通道数据（灰度图）

#### 图像对比图表（2x3）
- **第1行**: 原始RAW图像、暗电流矫正后图像
- **第2行**: 完全矫正图像、矫正统计信息、直方图对比

### 2. 保存的文件

#### 矫正矩阵文件
- **网格矫正矩阵**: `{原文件名}_{通道名}_grid_correction.npy`
- **全尺寸矫正矩阵**: `{原文件名}_{通道名}_full_correction.npy`

#### 分析结果文件
- **分析摘要**: `analysis_summary.json`
- **综合矫正数据**: `combined_lens_shading_correction.json`
- **分析图表**: `lens_shading_analysis_{文件名}_{序号}.png`
- **图像对比图**: `image_comparison_{文件名}_{序号}.png`

#### 矫正图像文件
- **暗电流矫正RAW**: `{原文件名}_dark_corrected.raw`
- **完全矫正RAW**: `{原文件名}_fully_corrected.raw`
- **完全矫正PNG**: `{原文件名}_fully_corrected.png`

### 3. 输出示例

```
=== Lens Shading Analysis Tool ===
Input path: F:\ZJU\Picture\lens_shading
Dark correction: True
Dark reference: F:\ZJU\Picture\dark\g9\average_dark_10frames_20241201_143022.raw
Dimensions: 3840 x 2160
Data type: uint16
Grid size: 32
Channel names: ['R', 'G1', 'G2', 'B']
Valid value range: 100 - 65000

Loading dark reference image: F:\ZJU\Picture\dark\g9\average_dark_10frames_20241201_143022.raw
  Dark image loaded: (2160, 3840), dtype: uint16
  Dark image range: 1234 - 1236

Output directory: F:\ZJU\Picture\lens_shading\lens_shading_analysis_20241201_143022

Found 3 RAW files

Analyzing: test_image1.raw
  Image loaded: (2160, 3840), dtype: uint16
  Data range: 1200 - 1300
  Applying dark current correction...
    Original range: 1200 - 1300
    Dark range: 1234 - 1236
    Dark-corrected range: 0.0 - 66.0
  Channel separation complete:
    R channel: (1080, 1920)
    G1 channel: (1080, 1920)
    G2 channel: (1080, 1920)
    B channel: (1080, 1920)
  Analyzing R channel...
    Grid analysis: 33x60 grids of size 32x32
    Using center reference value: 1250.45
    Correction matrix range: 0.850 - 1.150
    Interpolated to full size: (1080, 1920)
    Applying lens shading correction...
      Original range: 1200 - 1300
      Corrected range: 1020.0 - 1495.0
  Analyzing G1 channel...
    ...
  Analyzing G2 channel...
    ...
  Analyzing B channel...
    ...
    Reconstructed corrected image: (2160, 3840)
    Corrected image range: 1020.0 - 1495.0

Generating lens shading analysis plots...
  Plot saved: lens_shading_analysis_test_image1_1.png
  Comparison plot saved: image_comparison_test_image1_1.png

Saving correction matrices...
  R correction matrices saved for test_image1
  G1 correction matrices saved for test_image1
  G2 correction matrices saved for test_image1
  B correction matrices saved for test_image1
  Combined correction data saved: combined_lens_shading_correction.json

Saving corrected images...
  Dark-corrected RAW saved: test_image1_dark_corrected.raw
  Fully corrected RAW saved: test_image1_fully_corrected.raw
  Fully corrected PNG saved: test_image1_fully_corrected.png

Analysis complete!
Results saved to: F:\ZJU\Picture\lens_shading\lens_shading_analysis_20241201_143022
Summary saved to: F:\ZJU\Picture\lens_shading\lens_shading_analysis_20241201_143022\analysis_summary.json
```

## 配置选项详解

### 输入路径配置

- **INPUT_PATH**: 待分析的RAW图像路径
  - 可以是单个文件：`r"F:\ZJU\Picture\lens_shading\test_image.raw"`
  - 可以是文件夹：`r"F:\ZJU\Picture\lens_shading"`

### 暗点平矫正配置

- **DARK_RAW_PATH**: 暗电流参考图像路径
  - 建议使用 `dark_flow.py` 生成的平均暗电流图像
  - 必须与待分析图像具有相同的尺寸和数据类型

- **ENABLE_DARK_CORRECTION**: 是否启用暗点平矫正
  - `True`: 启用暗电流校正
  - `False`: 跳过暗电流校正

### 图像参数配置

- **IMAGE_WIDTH/HEIGHT**: 图像尺寸
  - 必须与所有输入图像匹配
  - 常见尺寸：3840x2160, 1920x1080

- **DATA_TYPE**: 数据类型
  - `uint16`: 12位数据存储在16位容器中（0-4095）
  - `uint8`: 8位无符号整数（0-255）

### Lens Shading分析配置

- **GRID_SIZE**: 网格大小
  - 建议值：32, 64, 128
  - 较小的网格提供更精细的分析
  - 较大的网格提供更稳定的统计

- **MIN_VALID_VALUE/MAX_VALID_VALUE**: 有效像素值范围
  - 过滤过暗和过亮的像素
  - 避免极端值对分析的影响

### 输出配置

- **OUTPUT_DIRECTORY**: 输出目录
  - `None`: 自动在输入目录下创建 `lens_shading_analysis_时间戳` 文件夹
  - 自定义路径：`r"F:\ZJU\Picture\output"`

- **GENERATE_PLOTS**: 是否生成分析图表
- **SAVE_PLOTS**: 是否保存图表文件
- **SAVE_CORRECTION_MATRIX**: 是否保存矫正矩阵
- **SAVE_CORRECTED_IMAGES**: 是否保存矫正后的图像

## 应用场景

### 1. 相机校准
- 检测镜头的亮度均匀性
- 生成镜头阴影校正参数
- 提高图像质量的一致性

### 2. 图像质量提升
- 校正图像边缘的亮度衰减
- 改善整体图像的亮度均匀性
- 为后续图像处理提供更好的基础

### 3. 工业检测
- 确保检测图像的亮度一致性
- 提高检测算法的准确性
- 标准化图像质量

### 4. 科学研究
- 天文图像的亮度校正
- 医学影像的质量提升
- 科学摄影的标准化

## 技术原理

### 1. 镜头阴影现象

镜头阴影是指图像从中心到边缘亮度逐渐衰减的现象，主要由以下原因造成：

- **余弦四次方定律**: 光线入射角度的余弦四次方衰减
- **镜头设计**: 广角镜头的固有特性
- **传感器特性**: 像素对斜入射光的响应差异

### 2. 校正方法

脚本使用以下方法进行校正：

1. **网格分析**: 将图像分割为网格，计算每个网格的平均亮度
2. **参考选择**: 使用图像中心作为参考亮度
3. **系数计算**: 校正系数 = 参考亮度 / 当前亮度
4. **插值扩展**: 将网格系数插值到全图像尺寸

### 3. 校正效果

校正后的图像具有：
- **均匀亮度**: 从中心到边缘亮度一致
- **改善对比度**: 边缘区域细节更清晰
- **标准化质量**: 为后续处理提供标准基础

## 注意事项

### 1. 文件要求
- **确保 `raw_reader.py` 在同一目录下**
- **输入图像应该是有效的RAW格式**
- **图像尺寸必须与配置匹配**

### 2. 图像质量要求
- **避免过暗或过亮的图像**: 影响网格分析的准确性
- **使用均匀光照**: 理想情况下使用均匀白光照明的图像
- **避免高对比度场景**: 高对比度场景可能影响校正效果

### 3. 网格大小选择
- **小网格**: 提供更精细的分析，但可能不稳定
- **大网格**: 提供更稳定的统计，但精度较低
- **建议值**: 32x32或64x64像素

### 4. 内存使用
- 处理大分辨率图像时可能需要较多内存
- 建议单次处理不超过10个大尺寸图像
- 对于内存受限的环境，可以分批处理

## 故障排除

### 常见问题

1. **找不到 raw_reader.py**
   - 确保所有文件在同一目录下

2. **图像读取失败**
   - 检查图像尺寸设置是否正确
   - 确认文件格式和数据类型

3. **网格分析异常**
   - 检查有效像素值范围设置
   - 确认图像质量是否适合分析

4. **插值失败**
   - 确保scipy库已安装
   - 检查网格尺寸是否合理

5. **内存不足**
   - 减少同时处理的图像数量
   - 使用更大的网格尺寸
   - 分批处理大量图像

### 获取帮助

如果遇到问题，请检查：
1. 配置文件路径是否正确
2. 文件路径是否存在
3. 图像尺寸是否匹配
4. 依赖库是否安装完整
5. 内存是否足够
6. 图像质量是否适合分析

## 扩展应用

### 1. 批量处理
脚本支持批量处理多个RAW图像，并生成综合的校正矩阵。

### 2. 校正矩阵应用
生成的校正矩阵可以用于：
- 实时图像校正
- 图像处理管线
- 相机固件更新

### 3. 质量评估
通过分析校正矩阵可以评估：
- 镜头的质量
- 相机的性能
- 校正的效果

### 4. 自动化集成
脚本可以集成到：
- 相机生产线
- 质量检测系统
- 图像处理工作流
