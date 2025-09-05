# Batch RAW Image Processor

这个脚本可以批量处理RAW图像文件，将它们转换为PNG格式。

## 功能特点

- 🔍 **自动检测图像尺寸**: 根据文件大小自动推断图像分辨率
- 🎨 **智能Bayer模式检测**: 自动检测和选择正确的Bayer模式
- 📁 **批量处理**: 处理整个目录下的所有RAW文件
- 💾 **无损输出**: 输出PNG格式，保持图像质量
- 📊 **详细日志**: 记录处理过程和结果
- ⚙️ **灵活配置**: 通过配置文件自定义设置

## 文件结构

```
tools/
├── batch_raw_processor.py    # 主处理脚本
├── batch_config.py           # 配置文件
├── raw_reader.py             # RAW读取函数库
└── README_batch_processor.md # 使用说明
```

## 使用方法

### 1. 基本使用

```bash
cd CCM_endoscope/tools
python batch_raw_processor.py
```

### 2. 修改配置

编辑 `batch_config.py` 文件来修改设置：

```python
# 输入输出目录
INPUT_DIRECTORY = r"F:\ZJU\Picture"  # RAW文件所在目录
OUTPUT_DIRECTORY = "./data2raw"       # 输出PNG文件目录

# 强制指定图像尺寸（如果需要）
FORCE_DIMENSIONS = (3840, 2160)  # 4K分辨率
# FORCE_DIMENSIONS = None         # 自动检测

# 其他设置
MAX_VALUE = 4095                 # RAW数据最大值
BAYER_PATTERN = 'auto'          # Bayer模式
OUTPUT_FORMAT = 'png'           # 输出格式
```

### 3. 支持的RAW格式

- `.raw` - 标准RAW格式
- `.RAW` - 大写扩展名
- `.Raw` - 混合大小写

## 配置选项说明

### 输入输出设置
- `INPUT_DIRECTORY`: RAW文件所在目录路径
- `OUTPUT_DIRECTORY`: 输出PNG文件目录路径

### 图像处理设置
- `FORCE_DIMENSIONS`: 强制指定图像尺寸，格式为 (宽度, 高度)
- `MAX_VALUE`: RAW数据的最大值（通常4095为12位，65535为16位）
- `BAYER_PATTERN`: Bayer模式，可选 'auto', 'rggb', 'bggr', 'grbg', 'gbrg'

### 输出设置
- `OUTPUT_FORMAT`: 输出格式，支持 'png', 'jpg'
- `PNG_COMPRESSION`: PNG压缩级别（0-9，数字越大压缩率越高但处理越慢）
- `JPEG_QUALITY`: JPEG质量（1-100，数字越大质量越好但文件越大）

### 性能设置
- `SHOW_PROGRESS`: 是否显示进度信息
- `SAVE_LOGS`: 是否保存处理日志
- `LOG_FILE`: 日志文件名

## 自动尺寸检测

脚本会自动检测图像尺寸，支持以下分辨率：

- **4K**: 3840×2160
- **2K**: 2560×1440
- **Full HD**: 1920×1080
- **HD**: 1280×720
- **VGA**: 640×480
- 以及其他常见分辨率

## 输出文件

- 输出文件名与原RAW文件名相同，仅扩展名改为PNG
- 例如：`25-08-25 142238.raw` → `25-08-25 142238.png`
- 所有文件保存在 `./data2raw` 目录下

## 错误处理

- 如果某个文件处理失败，脚本会继续处理其他文件
- 详细的错误信息会记录在日志文件中
- 最终会显示成功和失败的文件数量统计

## 注意事项

1. **确保 `raw_reader.py` 在同一目录下**
2. **确保有足够的磁盘空间存储输出文件**
3. **PNG文件通常比RAW文件大，因为是无损压缩**
4. **处理大量文件时可能需要较长时间**
5. **可以随时按 Ctrl+C 中断处理**

## 示例输出

```
=== Batch RAW Image Processor ===
Input directory: F:\ZJU\Picture
Output directory: ./data2raw
Output format: png
Force dimensions: None
Bayer pattern: auto
Max value: 4095

Output directory created/verified: E:\...\CCM_endoscope\tools\data2raw

Scanning for RAW files in: F:\ZJU\Picture
Found 5 RAW files:
  - 25-08-25 142238.raw
  - 25-08-26 093045.raw
  - 25-08-27 154220.raw
  ...

Starting batch processing...

[1/5] Processing: 25-08-25 142238.raw
Detected dimensions: 3840 x 2160
Reading RAW file...
Image loaded: (2160, 3840), dtype: uint16
Normalizing to 8-bit...
Performing demosaicing...
Detected Bayer pattern: RGGB
Demosaiced image size: (2160, 3840, 3)
Saving to: ./data2raw/25-08-25 142238.png
✓ Successfully saved: 25-08-25 142238.png

...

=== Processing Complete ===
Total files: 5
Successful: 5
Failed: 0
Processing time: 45.23 seconds
Average time per file: 9.05 seconds
Output directory: E:\...\CCM_endoscope\tools\data2raw

✓ All files processed successfully!
```

## 故障排除

### 常见问题

1. **找不到 raw_reader.py**
   - 确保所有文件在同一目录下

2. **输入目录不存在**
   - 检查 `INPUT_DIRECTORY` 路径是否正确

3. **图像尺寸检测失败**
   - 在配置文件中设置 `FORCE_DIMENSIONS`

4. **内存不足**
   - 处理大分辨率图像时可能需要更多内存

5. **处理速度慢**
   - 降低PNG压缩级别
   - 使用SSD硬盘存储

### 获取帮助

如果遇到问题，请检查：
1. 日志文件中的错误信息
2. 确保所有依赖库已安装
3. 检查文件路径和权限设置


