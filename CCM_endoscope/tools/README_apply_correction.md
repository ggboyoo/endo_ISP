# Apply Lens Shading Correction Tool

这个工具用于应用已保存的镜头阴影矫正参数来矫正新的RAW图像。它提供了单张图片处理和批量处理两种模式。

## 功能特点

- 🔧 **参数加载**: 自动加载之前保存的矫正参数
- 🖼️ **单张矫正**: 对单张RAW图像应用镜头阴影矫正
- 📁 **批量处理**: 批量处理多张RAW图像
- 🌑 **暗电流矫正**: 可选的暗电流校正
- 💾 **多格式输出**: 支持RAW、PNG、JPG等多种输出格式
- 📊 **详细日志**: 完整的处理过程记录和错误处理
- 📈 **直方图对比**: 显示矫正前后的直方图对比分析
- 📋 **统计摘要**: 提供详细的图像统计信息

## 文件结构

```
CCM_endoscope/tools/
├── lens_shading.py              # 主分析脚本（包含新功能）
├── apply_shading_correction.py  # 独立的矫正应用脚本
└── README_apply_correction.md   # 本说明文档
```

## 使用方法

### 方法1: 在lens_shading.py中使用

在 `lens_shading.py` 脚本末尾已经添加了使用示例，取消注释即可使用：

```python
# 取消注释下面的代码来使用

# 1. 加载矫正参数
correction_params = load_correction_parameters(r"F:\ZJU\Picture\lens shading\lens_shading_analysis_20241201_143022")

# 2. 矫正单张图片
result = shading_correct(
    input_image_path=r"F:\ZJU\Picture\lens shading\new_image.raw",
    correction_params=correction_params,
    dark_image_path=r"F:\ZJU\Picture\dark\g8\average_dark.raw",  # 可选
    output_dir=r"F:\ZJU\Picture\lens shading\corrected",
    save_formats=['raw', 'png', 'jpg']
)

# 3. 批量矫正
batch_results = batch_shading_correct(
    input_dir=r"F:\ZJU\Picture\lens shading\new_images",
    correction_params=correction_params,
    dark_image_path=r"F:\ZJU\Picture\dark\g8\average_dark.raw",  # 可选
    output_dir=r"F:\ZJU\Picture\lens shading\batch_corrected",
    save_formats=['raw', 'png']
)
```

### 方法2: 使用独立的apply_shading_correction.py脚本

1. **配置参数**: 在脚本顶部修改配置区域
2. **运行脚本**: `python apply_shading_correction.py`

## 核心函数说明

### 1. `load_correction_parameters(correction_dir)`

加载已保存的矫正参数。

**参数:**
- `correction_dir`: 包含矫正参数文件的目录路径

**返回值:**
- 包含各通道矫正参数的字典

**支持的文件格式:**
- `combined_lens_shading_correction.json` (优先)
- `*_{通道名}_full_correction.npy` 文件

### 2. `shading_correct(input_image_path, correction_params, ...)`

对单张图片应用镜头阴影矫正。

**参数:**
- `input_image_path`: 输入RAW图像路径
- `correction_params`: 矫正参数字典
- `dark_image_path`: 暗电流图像路径（可选）
- `output_dir`: 输出目录（可选，默认与输入图片同目录）
- `save_formats`: 保存格式列表，支持 `['raw', 'png', 'jpg']`

**返回值:**
- 包含输出文件路径的字典

### 3. `batch_shading_correct(input_dir, correction_params, ...)`

批量处理多张图片。

**参数:**
- `input_dir`: 输入图片目录
- `correction_params`: 矫正参数字典
- `dark_image_path`: 暗电流图像路径（可选）
- `output_dir`: 输出目录（可选）
- `save_formats`: 保存格式列表

**返回值:**
- 包含每张图片处理结果的列表

## 配置选项

### 矫正参数路径配置
```python
CORRECTION_PARAMS_DIR = r"F:\ZJU\Picture\lens shading\lens_shading_analysis_20241201_143022"
```

### 输入输出配置
```python
INPUT_IMAGE_PATH = r"F:\ZJU\Picture\lens shading\new_image.raw"  # 单张图片
INPUT_DIRECTORY = r"F:\ZJU\Picture\lens shading\new_images"      # 批量目录
OUTPUT_DIRECTORY = r"F:\ZJU\Picture\lens shading\corrected"      # 输出目录
```

### 暗电流配置
```python
DARK_IMAGE_PATH = r"F:\ZJU\Picture\dark\g8\average_dark.raw"    # 暗电流图像
ENABLE_DARK_CORRECTION = True                                     # 是否启用
```

### 输出格式配置
```python
SAVE_FORMATS = ['raw', 'png', 'jpg']  # 支持的格式
```

### 处理模式
```python
PROCESS_MODE = 'single'  # 'single' 或 'batch'
```

### 直方图显示配置
```python
SHOW_HISTOGRAMS = True    # 是否显示矫正前后直方图对比
SAVE_HISTOGRAMS = True    # 是否保存直方图到文件
```

## 工作流程

### 1. 矫正参数生成
```
原始RAW图像 → lens_shading.py分析 → 生成矫正参数文件
```

### 2. 应用矫正
```
新RAW图像 + 矫正参数 → shading_correct() → 矫正后的图像
```

### 3. 完整流程
```
暗电流矫正 → RGGB通道分离 → 应用矫正系数 → 图像重建 → 多格式输出 → 直方图分析 → 统计摘要
```

## 输出文件

### 矫正后的图像
- `{原文件名}_shading_corrected_{时间戳}.raw` - RAW格式
- `{原文件名}_shading_corrected_{时间戳}.png` - PNG格式
- `{原文件名}_shading_corrected_{时间戳}.jpg` - JPG格式

### 处理摘要
- `{原文件名}_correction_summary_{时间戳}.json` - 单张图片处理摘要
- `batch_correction_summary.json` - 批量处理摘要

### 直方图分析
- `{原文件名}_histogram_comparison.png` - 矫正前后直方图对比图
- 控制台输出 - 详细的图像统计信息摘要

## 直方图分析功能

### 功能说明
新增的直方图分析功能提供了矫正前后图像的详细对比分析：

#### 1. 直方图对比显示
- **原始图像直方图**: 显示矫正前图像的像素值分布
- **暗电流矫正后直方图**: 显示暗电流矫正后的像素值分布（如果启用）
- **完全矫正后直方图**: 显示镜头阴影矫正后的像素值分布
- **对比直方图**: 叠加显示原始和矫正后的直方图，便于直观对比

#### 2. 统计信息摘要
提供详细的数值统计信息：
- **均值 (Mean)**: 图像的平均亮度
- **标准差 (Std)**: 图像亮度的变化程度
- **最小值 (Min)**: 图像中最暗的像素值
- **最大值 (Max)**: 图像中最亮的像素值
- **中位数 (Median)**: 图像亮度的中位值

#### 3. 显示模式
- **启用暗电流矫正**: 显示4个子图（2x2布局）
  - 原始图像直方图
  - 暗电流矫正后直方图
  - 完全矫正后直方图
  - 原始vs完全矫正对比图
- **未启用暗电流矫正**: 显示2个子图（1x2布局）
  - 原始图像直方图
  - 原始vs矫正对比图

#### 4. 输出选项
- **屏幕显示**: 实时显示直方图窗口
- **文件保存**: 保存高分辨率PNG格式的直方图文件
- **统计输出**: 在控制台打印详细的统计信息

### 配置选项
```python
# 直方图显示配置
SHOW_HISTOGRAMS = True    # 是否显示矫正前后直方图对比
SAVE_HISTOGRAMS = True    # 是否保存直方图到文件
```

## 使用场景

### 1. 相机校准后应用
- 使用校准图像生成的矫正参数
- 对日常拍摄的图像进行实时矫正
- 通过直方图验证矫正效果

### 2. 批量图像处理
- 处理大量RAW图像
- 保持一致的图像质量
- 批量分析矫正效果

### 3. 生产环境集成
- 集成到图像处理管线
- 自动化图像质量提升
- 质量控制和效果验证

### 4. 研究和分析
- 分析镜头阴影矫正的效果
- 比较不同矫正参数的影响
- 研究图像质量改善情况

## 注意事项

### 1. 参数兼容性
- 矫正参数必须与目标图像具有相同的尺寸
- 脚本会自动调整参数尺寸（如果必要）

### 2. 文件路径
- 确保所有路径都是正确的
- 使用绝对路径避免相对路径问题

### 3. 内存使用
- 处理大分辨率图像时注意内存使用
- 批量处理时考虑分批进行

### 4. 错误处理
- 脚本包含完整的错误处理
- 检查输出日志了解处理状态

## 故障排除

### 常见问题

1. **找不到矫正参数文件**
   - 检查 `CORRECTION_PARAMS_DIR` 路径
   - 确认矫正参数文件存在

2. **参数尺寸不匹配**
   - 脚本会自动调整，但建议使用匹配的参数
   - 检查图像尺寸设置

3. **输出目录权限问题**
   - 确保有写入权限
   - 检查磁盘空间

4. **暗电流图像加载失败**
   - 检查暗电流图像路径
   - 确认图像尺寸和数据类型

### 获取帮助

如果遇到问题，请检查：
1. 文件路径是否正确
2. 矫正参数是否完整
3. 图像尺寸是否匹配
4. 依赖库是否安装完整

## 扩展应用

### 1. 自定义矫正参数
```python
# 创建自定义矫正参数
custom_params = {
    'R': np.ones((1080, 1920)) * 1.1,    # R通道增强
    'G1': np.ones((1080, 1920)) * 1.0,   # G1通道不变
    'G2': np.ones((1080, 1920)) * 1.0,   # G2通道不变
    'B': np.ones((1080, 1920)) * 0.9     # B通道减弱
}

# 应用自定义参数
result = shading_correct(
    input_image_path="image.raw",
    correction_params=custom_params,
    save_formats=['png']
)
```

### 2. 集成到其他脚本
```python
from lens_shading import shading_correct, load_correction_parameters

# 加载参数
params = load_correction_parameters("correction_dir")

# 应用矫正
result = shading_correct("image.raw", params)
```

### 3. 实时处理
```python
# 在图像采集循环中使用
for image_path in image_stream:
    corrected = shading_correct(image_path, params, save_formats=['png'])
    # 进一步处理矫正后的图像
```

这个工具为镜头阴影矫正提供了完整的解决方案，从参数生成到实际应用，支持单张和批量处理，可以轻松集成到现有的图像处理工作流中。
