# Files ISP - 批量RAW图像处理工具

`files_isp.py` 是一个批量处理RAW图像的脚本，调用 `ISP.py` 将指定路径下的所有RAW图像处理并保存为JPG格式。

## 功能特点

- **批量处理**: 自动处理指定目录下的所有RAW文件
- **灵活配置**: 支持自定义图像参数和处理选项
- **暗电流矫正**: 可选的暗电流减法
- **镜头阴影矫正**: 可选的镜头阴影矫正
- **白平衡和CCM**: 可选的白平衡和颜色矫正矩阵应用
- **进度跟踪**: 实时显示处理进度
- **详细报告**: 生成JSON格式的处理报告
- **错误处理**: 完善的错误处理和日志记录

## 使用方法

### 基本用法

```bash
# 最简单的用法 - 只处理RAW文件
python files_isp.py --input_dir "F:\ZJU\Picture\raw_images" --output_dir "F:\ZJU\Picture\processed"

# 带暗电流矫正
python files_isp.py --input_dir "F:\ZJU\Picture\raw_images" --output_dir "F:\ZJU\Picture\processed" --dark_file "F:\ZJU\Picture\dark\average_dark.raw"

# 带镜头阴影矫正
python files_isp.py --input_dir "F:\ZJU\Picture\raw_images" --output_dir "F:\ZJU\Picture\processed" --lens_shading_dir "F:\ZJU\Picture\lens_shading\new"

# 完整处理流程
python files_isp.py --input_dir "F:\ZJU\Picture\raw_images" --output_dir "F:\ZJU\Picture\processed" --dark_file "F:\ZJU\Picture\dark\average_dark.raw" --lens_shading_dir "F:\ZJU\Picture\lens_shading\new" --enable_wb --enable_ccm
```

### 参数说明

#### 必需参数
- `--input_dir, -i`: 包含RAW文件的输入目录
- `--output_dir, -o`: 处理后的JPG文件输出目录

#### 可选参数

**图像参数:**
- `--width`: 图像宽度 (默认: 3840)
- `--height`: 图像高度 (默认: 2160)
- `--data_type`: 数据类型 (uint8/uint16, 默认: uint16)
- `--bayer_pattern`: Bayer模式 (rggb/bggr/grbg/gbrg, 默认: rggb)

**处理选项:**
- `--dark_file, -d`: 暗电流参考文件路径
- `--lens_shading_dir, -l`: 镜头阴影矫正参数目录
- `--no_dark`: 禁用暗电流矫正
- `--no_lens_shading`: 禁用镜头阴影矫正
- `--enable_wb`: 启用白平衡矫正
- `--enable_ccm`: 启用CCM颜色矫正

**输出选项:**
- `--quality`: JPG质量 1-100 (默认: 95)
- `--save_16bit`: 同时保存16位PNG文件
- `--overwrite`: 覆盖已存在的输出文件

## 使用示例

### 示例1: 基本处理
```bash
python files_isp.py -i "F:\ZJU\Picture\test" -o "F:\ZJU\Picture\output"
```

### 示例2: 带暗电流矫正
```bash
python files_isp.py -i "F:\ZJU\Picture\test" -o "F:\ZJU\Picture\output" -d "F:\ZJU\Picture\dark\average_dark.raw"
```

### 示例3: 完整ISP流程
```bash
python files_isp.py -i "F:\ZJU\Picture\test" -o "F:\ZJU\Picture\output" -d "F:\ZJU\Picture\dark\average_dark.raw" -l "F:\ZJU\Picture\lens_shading\new" --enable_wb --enable_ccm --save_16bit
```

### 示例4: 自定义参数
```bash
python files_isp.py -i "F:\ZJU\Picture\test" -o "F:\ZJU\Picture\output" --width 1920 --height 1080 --data_type uint8 --quality 90 --overwrite
```

## 输出文件

### 处理后的图像
- `filename_processed.jpg`: 8位JPG格式的处理后图像
- `filename_processed_16bit.png`: 16位PNG格式图像 (如果启用 `--save_16bit`)

### 处理报告
- `processing_report.json`: 详细的处理报告，包含:
  - 处理时间戳
  - 输入/输出目录
  - 配置参数
  - 处理统计 (总数/成功/失败/跳过)
  - 每个文件的详细结果

## 处理流程

1. **扫描输入目录**: 查找所有 `.raw` 和 `.RAW` 文件
2. **加载参考数据**: 加载暗电流和镜头阴影矫正参数
3. **批量处理**: 对每个RAW文件执行以下步骤:
   - 读取RAW数据
   - 暗电流矫正 (如果启用)
   - 镜头阴影矫正 (如果启用)
   - 去马赛克
   - 白平衡矫正 (如果启用)
   - CCM颜色矫正 (如果启用)
   - 转换为8位并保存为JPG
4. **生成报告**: 创建详细的处理报告

## 错误处理

- **文件不存在**: 自动跳过不存在的文件
- **处理失败**: 记录错误信息并继续处理其他文件
- **参数错误**: 提供详细的错误信息和修复建议
- **权限问题**: 检查输出目录的写入权限

## 性能优化

- **内存管理**: 逐个处理文件，避免内存溢出
- **进度显示**: 实时显示处理进度
- **跳过已存在**: 默认跳过已处理的文件 (除非使用 `--overwrite`)
- **并行处理**: 可考虑后续版本添加多线程支持

## 注意事项

1. **文件格式**: 目前只支持 `.raw` 和 `.RAW` 文件
2. **内存使用**: 大图像可能需要较多内存
3. **处理时间**: 处理时间取决于图像大小和启用的矫正选项
4. **输出质量**: JPG质量设置影响文件大小和图像质量
5. **依赖文件**: 确保 `ISP.py` 和相关模块在同一目录

## 故障排除

### 常见问题

1. **"ISP.py not found"**: 确保 `ISP.py` 在相同目录
2. **"No RAW files found"**: 检查输入目录路径和文件扩展名
3. **"Permission denied"**: 检查输出目录的写入权限
4. **"Memory error"**: 尝试处理较小的图像或减少并发处理

### 调试模式

添加 `-v` 或 `--verbose` 参数可以获得更详细的输出信息 (如果实现)。

## 更新日志

- **v1.0**: 初始版本，支持基本的批量RAW图像处理
- 支持暗电流和镜头阴影矫正
- 支持白平衡和CCM矫正
- 生成详细的处理报告
