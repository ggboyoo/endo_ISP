# ISP (Image Signal Processing) Tool

这个脚本实现了完整的图像信号处理管线，包括暗电流校正、镜头阴影矫正、去马赛克等步骤，最终输出高质量的彩色PNG图像。

## 功能特点

- 📁 **RAW文件读取**: 调用 `raw_reader.py` 功能读取RAW图像
- 🌑 **暗电流校正**: 减去暗电流参考图像，提高图像质量
- 🔍 **镜头阴影矫正**: 应用镜头阴影矫正参数，改善图像均匀性
- 🖼️ **去马赛克**: 将RAW数据转换为彩色图像
- 📊 **完整管线可视化**: 显示所有处理步骤的对比图表
- 💾 **PNG输出**: 保存最终处理后的彩色PNG图像
- 🔧 **灵活配置**: 支持多种处理参数和输出选项

## 使用方法

### 1. 配置参数

在脚本顶部修改配置区域：

```python
# 输入路径配置
INPUT_PATH = r"F:\ZJU\Picture\lens shading\1000.raw"  # 待处理的RAW图像路径
DARK_RAW_PATH = r"F:\ZJU\Picture\dark\g8\average_dark.raw"  # 暗电流图像路径
LENS_SHADING_PARAMS_DIR = r"F:\ZJU\Picture\lens shading"  # 镜头阴影矫正参数目录

# 图像参数配置
IMAGE_WIDTH = 3840      # 图像宽度
IMAGE_HEIGHT = 2160     # 图像高度
DATA_TYPE = 'uint16'    # 数据类型

# 输出配置
OUTPUT_DIRECTORY = None  # 输出目录（None为自动生成）
GENERATE_PLOTS = True   # 是否生成对比图表
SAVE_PLOTS = True       # 是否保存图表文件
DEMOSAIC_OUTPUT = True  # 是否输出去马赛克后的图像

# 暗电流校正配置
DARK_SUBTRACTION_ENABLED = True  # 是否启用暗电流校正
CLIP_NEGATIVE_VALUES = True      # 是否将负值裁剪为0

# 镜头阴影矫正配置
LENS_SHADING_ENABLED = True      # 是否启用镜头阴影矫正

# 输出配置
NORMALIZE_OUTPUT = True          # 是否将输出归一化到0-255范围
```

### 2. 运行脚本

```bash
python ISP.py
```

## 核心功能

### 1. 暗电流校正

脚本的核心功能是减去暗电流参考图像：

```python
# 暗电流校正过程
corrected_data = raw_data.astype(np.float64) - dark_data.astype(np.float64)

# 负值裁剪（可选）
if clip_negative:
    corrected_data = np.clip(corrected_data, 0, None)
```

**暗电流校正的作用：**
- 去除传感器的暗电流噪声
- 提高图像的对比度和清晰度
- 为后续图像处理提供更好的基础

### 2. 白平衡矫正

脚本支持白平衡矫正功能，可以读取WB.py生成的白平衡参数：

```python
# 白平衡矫正过程
corrected = color_image.copy().astype(np.float32)
corrected[:, :, 0] *= b_gain  # B通道
corrected[:, :, 1] *= g_gain  # G通道
corrected[:, :, 2] *= r_gain  # R通道
corrected = np.clip(corrected, 0, 255)
```

**白平衡矫正的作用：**
- 校正图像的颜色偏差
- 确保白色物体显示为真正的白色
- 提高图像的颜色准确性

### 3. 16位数据存储

脚本支持保存16位版本的数据，用于存储12位精度：

```python
# 16位PNG保存（存储12位数据）
color_16bit = result['color_img'].astype(np.uint16)
color_16bit = (color_16bit * 4095 / 255).astype(np.uint16)  # 8位转12位
cv2.imwrite(str(color_16bit_path), color_16bit)

# 16位RAW保存（镜头阴影校正后的数据）
lens_corrected_16bit = result['lens_corrected'].astype(np.uint16)
with open(raw_16bit_path, 'wb') as f:
    lens_corrected_16bit.tofile(f)
```

**16位数据存储的优势：**
- 保持12位数据精度，避免8位量化损失
- 支持后续高精度图像处理
- 提供镜头阴影校正后的中间数据

### 4. 完整ISP处理流程

```
原始RAW图像 → 暗电流校正 → 镜头阴影矫正 → 8位归一化 → 去马赛克 → 白平衡矫正 → 彩色PNG
     ↓              ↓           ↓            ↓         ↓         ↓         ↓
  读取数据     减去暗电流    应用矫正参数   归一化到    Bayer到   应用WB    保存PNG
             参考图像      改善均匀性    0-255范围   彩色转换   增益参数
```

### 3. 输出文件类型

脚本会生成以下输出文件：

1. **最终彩色图像**: `{原文件名}_processed.png` - 完整ISP处理后的彩色PNG图像（8位）
2. **16位彩色图像**: `{原文件名}_processed_16bit.png` - 16位PNG图像，存储12位数据
3. **16位RAW数据**: `{原文件名}_lens_corrected_16bit.raw` - 镜头阴影校正后的16位RAW数据
4. **处理管线图表**: `ISP_pipeline_{原文件名}_{序号}.png` - 显示所有处理步骤的对比图
5. **处理摘要**: `processing_summary.json` - 包含处理参数和结果的JSON文件

## 配置选项详解

### 输入路径配置

- **INPUT_PATH**: 待处理的RAW图像路径
  - 可以是单个文件：`r"F:\ZJU\Picture\test\image1.raw"`
  - 可以是文件夹：`r"F:\ZJU\Picture\test"`

- **DARK_RAW_PATH**: 暗电流参考图像路径
  - 建议使用 `dark_flow.py` 生成的平均暗电流图像
  - 必须与待处理图像具有相同的尺寸和数据类型

- **LENS_SHADING_PARAMS_DIR**: 镜头阴影矫正参数目录
  - 包含 `combined_lens_shading_correction.json` 文件
  - 由 `lens_shading.py` 分析生成

### 图像参数配置

- **IMAGE_WIDTH/HEIGHT**: 图像尺寸
  - 必须与所有输入图像匹配
  - 常见尺寸：3840x2160, 1920x1080

- **DATA_TYPE**: 数据类型
  - `uint16`: 12位数据存储在16位容器中（0-4095）
  - `uint8`: 8位无符号整数（0-255）

### 输出配置

- **OUTPUT_DIRECTORY**: 输出目录
  - `None`: 自动在输入目录下创建 `ISP_output_时间戳` 文件夹
  - 自定义路径：`r"F:\ZJU\Picture\output"`

- **GENERATE_PLOTS**: 是否生成对比图表
- **SAVE_PLOTS**: 是否保存图表文件
- **DEMOSAIC_OUTPUT**: 是否输出去马赛克后的彩色图像

### 暗电流校正配置

- **DARK_SUBTRACTION_ENABLED**: 是否启用暗电流校正
- **CLIP_NEGATIVE_VALUES**: 是否将负值裁剪为0

### 镜头阴影矫正配置

- **LENS_SHADING_ENABLED**: 是否启用镜头阴影矫正

### 白平衡配置

- **WHITE_BALANCE_ENABLED**: 是否启用白平衡矫正
- **WB_PARAMETERS_PATH**: 白平衡参数文件路径（JSON格式）

### 输出配置

- **NORMALIZE_OUTPUT**: 是否将输出归一化到0-255范围

## 使用示例

### 基本使用

1. **配置路径**：修改脚本顶部的路径配置
2. **运行脚本**：`python ISP.py`
3. **查看输出**：在输出目录中查看处理结果

### 带白平衡的使用

1. **生成白平衡参数**：先运行 `python WB.py` 生成白平衡参数文件
2. **配置白平衡路径**：在ISP.py中设置 `WB_PARAMETERS_PATH` 为生成的参数文件路径
3. **启用白平衡**：设置 `WHITE_BALANCE_ENABLED = True`
4. **运行ISP处理**：`python ISP.py`

### 输出示例

```
=== ISP (Image Signal Processing) Tool ===
Input path: F:\ZJU\Picture\lens shading\1000.raw
Dark reference: F:\ZJU\Picture\dark\g8\average_dark.raw
Lens shading params: F:\ZJU\Picture\lens shading
Dimensions: 3840 x 2160
Data type: uint16
Dark subtraction: True
Lens shading: True
White balance: True
Demosaic output: True

Output directory: F:\ZJU\Picture\lens shading\ISP_output_20241201_143022

Loading dark reference image: F:\ZJU\Picture\dark\g8\average_dark.raw
  Dark image loaded: (2160, 3840), dtype: uint16
  Dark image range: 1234 - 1236

Lens shading parameters loaded for channels: ['R', 'G1', 'G2', 'B']

Loading white balance parameters: F:\ZJU\Picture\wb_output\wb_parameters_manual_roi_20241201_143022.json
  White balance gains loaded:
    B gain: 1.234
    G gain: 1.000
    R gain: 0.876

Found 1 RAW files

Processing: 1000.raw
  1. RAW loaded: (2160, 3840), range: 1250-1300
  2. Dark correction applied
  3. Applying lens shading correction...
  3. Lens shading correction applied
  4. Normalized to 8-bit: (2160, 3840), range: 0-255
  5. Demosaicing...
  5. Color image: (2160, 3840, 3)
  6. Applying white balance correction...
  Applying gains: B=1.234, G=1.000, R=0.876
  White balance correction applied
  6. White balance correction applied
  Final PNG saved: F:\ZJU\Picture\lens shading\ISP_output_20241201_143022\1000_processed.png
  16-bit PNG saved: F:\ZJU\Picture\lens shading\ISP_output_20241201_143022\1000_processed_16bit.png
  16-bit RAW saved: F:\ZJU\Picture\lens shading\ISP_output_20241201_143022\1000_lens_corrected_16bit.raw

Processing complete!
Results saved to: F:\ZJU\Picture\lens shading\ISP_output_20241201_143022
Summary saved to: F:\ZJU\Picture\lens shading\ISP_output_20241201_143022\processing_summary.json
```

## 应用场景

### 1. 图像质量提升
- 去除暗电流噪声，提高图像清晰度
- 矫正镜头阴影，改善图像均匀性
- 增强图像对比度和细节表现
- 输出高质量的彩色图像

### 2. 科学图像处理
- 医学影像的暗电流校正
- 天文图像的噪声去除
- 工业检测图像的质量提升

### 3. 相机校准
- 传感器暗电流特性分析
- 图像校正参数优化
- 相机性能评估

### 4. 批量图像处理
- 批量RAW图像校正
- 自动化图像质量提升
- 标准化图像处理流程

## 注意事项

### 1. 文件要求
- **确保 `raw_reader.py` 在同一目录下**
- **暗电流图像必须与待处理图像具有相同的尺寸和数据类型**
- **输入图像应该是有效的RAW格式**

### 2. 内存使用
- 处理大分辨率图像时可能需要较多内存
- 建议单次处理不超过50个大尺寸图像
- 对于内存受限的环境，可以分批处理

### 3. 处理时间
- 暗电流校正相对较快
- 去马赛克处理可能较慢，特别是大尺寸图像
- 图表生成需要额外时间

### 4. 输出文件
- 输出目录会自动创建
- 所有输出文件都保存在输出目录中
- 文件名会自动添加后缀以区分不同类型

## 故障排除

### 常见问题

1. **找不到 raw_reader.py**
   - 确保所有文件在同一目录下

2. **暗电流图像加载失败**
   - 检查暗电流图像路径是否正确
   - 确认图像尺寸和数据类型是否匹配

3. **处理失败**
   - 检查输入图像路径是否正确
   - 确认图像尺寸设置是否匹配
   - 验证数据类型设置是否正确

4. **内存不足**
   - 减少同时处理的图像数量
   - 关闭去马赛克功能
   - 分批处理大量图像

5. **输出目录创建失败**
   - 检查磁盘空间是否充足
   - 确认目录的写入权限

### 获取帮助

如果遇到问题，请检查：
1. 配置文件路径是否正确
2. 文件路径是否存在
3. 图像尺寸是否匹配
4. 数据类型是否一致
5. 依赖库是否安装完整
6. 内存和磁盘空间是否充足

## 与其他工具的配合使用

### 推荐工作流程

1. **使用 `dark_flow.py` 生成暗电流参考图像**
   ```bash
   python dark_flow.py  # 生成平均暗电流图像
   ```

2. **使用 `lens_shading.py` 分析镜头阴影并生成矫正参数**
   ```bash
   python lens_shading.py  # 生成镜头阴影矫正参数
   ```

3. **使用 `ISP.py` 进行完整的图像信号处理**
   ```bash
   python ISP.py  # 应用暗电流校正和镜头阴影矫正
   ```

### 优势

- **完整ISP管线**: 从RAW到高质量彩色图像的完整处理流程
- **数据一致性**: 使用相同的图像读取和处理函数
- **结果可追溯**: 完整的处理记录和参数保存
- **质量保证**: 基于统计分析的暗电流参考图像和镜头阴影矫正参数
- **模块化设计**: 每个工具专注于特定功能，便于维护和扩展
