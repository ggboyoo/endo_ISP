# Dark Flow Analysis Tool

这个脚本用于分析RAW图像数据的暗电流特性，计算平均值、方差等统计信息。

## 功能特点

- 📊 **统计分析**: 计算均值、方差、标准差、分位数等
- 📈 **时序分析**: 分析多个暗帧的时间稳定性
- 🗂️ **文件夹整体统计**: 计算文件夹内所有图像的联合统计指标
- 🖼️ **平均图像生成**: 生成所有RAW图像的平均值图像
- 📋 **数据导出**: 将结果保存为JSON格式
- 📊 **可视化**: 生成统计图表
- 🔧 **灵活配置**: 支持命令行参数和直接配置

## 使用方法

### 1. 配置参数

在脚本顶部修改配置区域：

```python
# 输入路径配置（支持文件夹或单个文件）
INPUT_PATH = r"F:\ZJU\Picture\dark\g5"  # 文件夹路径，例如: r"F:\ZJU\Picture"
# 或者单个文件路径，例如: r"F:\ZJU\Picture\dark_frame1.raw"

# 图像参数配置
IMAGE_WIDTH = 3840      # 图像宽度
IMAGE_HEIGHT = 2160     # 图像高度
DATA_TYPE = 'uint16'    # 数据类型

# 输出配置
OUTPUT_FILE = 'dark_analysis.json'  # 输出JSON文件名
GENERATE_PLOTS = True               # 是否生成图表
SAVE_PLOTS = True                   # 是否保存图表文件
CALCULATE_OVERALL_STATS = True      # 是否计算文件夹整体统计信息
GENERATE_AVERAGE_RAW = True         # 是否生成平均RAW图像
AVERAGE_RAW_FILENAME = None         # 平均RAW图像文件名（None为自动生成）
```

### 2. 运行脚本

```bash
python dark_flow.py
```

## 新功能：文件夹整体统计

### 功能说明

新增了 `CALCULATE_OVERALL_STATS` 配置选项，当设置为 `True` 时，脚本会：

1. **收集所有图像数据**: 将文件夹内所有RAW图像的像素值合并
2. **计算整体统计**: 基于所有像素值计算整体的均值、方差等指标
3. **提供全局视角**: 给出整个文件夹数据的统计特征

### 整体统计指标

- **overall_mean**: 所有图像像素的整体均值
- **overall_variance**: 所有图像像素的整体方差
- **overall_std**: 所有图像像素的整体标准差
- **overall_min/max**: 所有图像像素的最小/最大值
- **overall_median**: 所有图像像素的中位数
- **overall_mad**: 所有图像像素的中位数绝对偏差
- **overall_p1/p5/p25/p75/p95/p99**: 所有图像像素的分位数
- **total_pixels**: 总像素数量
- **total_images**: 总图像数量

### 输出示例

```
=== Calculating Overall Folder Statistics ===
Processing 10 images for combined analysis...
  [ 1/10] Loading: dark_frame1.raw
       Image shape: (2160, 3840), Pixels: 8,294,400
  [ 2/10] Loading: dark_frame2.raw
       Image shape: (2160, 3840), Pixels: 8,294,400
  ...

Total pixels across all images: 82,944,000
Data range: 1200 - 1300

Overall Statistics:
  Mean:     1234.56
  Variance: 789.01
  Std Dev:  28.09
  Min-Max:  1200 - 1300
  Median:   1235.00
  MAD:      25.50
```

## 新功能：平均RAW图像生成

### 功能说明

新增了 `GENERATE_AVERAGE_RAW` 配置选项，当设置为 `True` 时，脚本会：

1. **加载所有图像**: 读取文件夹内所有RAW图像数据
2. **计算像素平均值**: 对每个像素位置计算所有图像的平均值
3. **生成新RAW文件**: 保存为新的.raw格式文件
4. **保持数据类型**: 输出文件与输入文件具有相同的数据类型和尺寸

### 配置选项

- **GENERATE_AVERAGE_RAW**: 是否启用平均图像生成功能
- **AVERAGE_RAW_FILENAME**: 自定义输出文件名（None为自动生成）

### 自动命名规则

如果不指定文件名，脚本会自动生成：
```
average_dark_{帧数}frames_{时间戳}.raw
```

例如：`average_dark_10frames_20241201_143022.raw`

### 输出示例

```
=== Generating Average RAW Image ===
Processing 10 images for averaging...
  [ 1/10] Loading: dark_frame1.raw
       Image shape: (2160, 3840), Added to accumulator
  [ 2/10] Loading: dark_frame2.raw
       Image shape: (2160, 3840), Added to accumulator
  ...

Successfully processed 10 files
Average image shape: (2160, 3840)
Average image range: 1234.12 - 1235.89

Average RAW image saved to: average_dark_10frames_20241201_143022.raw
File size: 16,588,800 bytes
```

### 应用场景

- **噪声抑制**: 通过平均多帧图像减少随机噪声
- **参考图像**: 生成稳定的暗电流参考图像
- **质量控制**: 评估多帧图像的一致性
- **校准基准**: 为图像校正提供标准参考

## 输出结果

### JSON文件结构

```json
{
  "analysis_timestamp": "2024-01-01T12:00:00",
  "total_frames": 10,
  "frame_analyses": [...],
  "overall_folder_statistics": {
    "overall_mean": 1234.56,
    "overall_variance": 789.01,
    "overall_std": 28.09,
    "overall_min": 1200,
    "overall_max": 1300,
    "overall_median": 1235,
    "overall_mad": 25.5,
    "overall_p1": 1210,
    "overall_p5": 1220,
    "overall_p25": 1225,
    "overall_p75": 1245,
    "overall_p95": 1250,
    "overall_p99": 1260,
    "total_pixels": 82944000,
    "total_images": 10
  },
  "temporal_statistics": {...}
}
```

### 生成的文件

1. **JSON分析结果**: `dark_analysis.json`
2. **统计图表**: `dark_analysis_plot_YYYYMMDD_HHMMSS.png`
3. **平均RAW图像**: `average_dark_10frames_YYYYMMDD_HHMMSS.raw`

## 应用场景

### 1. 暗电流分析
- 评估传感器的暗电流水平
- 检测暗电流的时间稳定性
- 识别异常像素或区域

### 2. 传感器校准
- 为图像校正提供暗电流参考
- 评估传感器的噪声特性
- 验证传感器的性能指标

### 3. 质量控制
- 监控传感器性能变化
- 检测传感器老化或损坏
- 确保图像质量的一致性

### 4. 整体性能评估
- 评估整个数据集的质量分布
- 识别系统性偏差或异常
- 为批量处理提供统计基准

### 5. 平均图像应用
- **新增**: 生成稳定的暗电流参考图像
- **新增**: 通过多帧平均减少噪声
- **新增**: 为图像校正提供标准模板
- **新增**: 评估多帧图像的一致性

## 注意事项

1. **确保 `raw_reader.py` 在同一目录下**
2. **RAW文件应该是暗帧数据**（无光照条件下的图像）
3. **图像尺寸必须正确**，否则会导致读取错误
4. **大量文件处理时可能需要较长时间**
5. **整体统计功能会消耗更多内存**（需要加载所有图像数据）
6. **平均图像生成会消耗更多内存**（需要同时处理所有图像）
7. **图表生成需要matplotlib支持**

## 性能优化建议

- **内存管理**: 如果处理大量大分辨率图像，考虑分批处理
- **整体统计**: 对于内存受限的环境，可以设置 `CALCULATE_OVERALL_STATS = False`
- **平均图像**: 对于内存受限的环境，可以设置 `GENERATE_AVERAGE_RAW = False`
- **文件数量**: 建议单次处理不超过100个文件，避免内存溢出

## 故障排除

### 常见问题

1. **找不到 raw_reader.py**
   - 确保所有文件在同一目录下

2. **文件读取错误**
   - 检查图像尺寸设置是否正确
   - 确认文件格式和数据类型

3. **内存不足**
   - 处理大分辨率图像时可能需要更多内存
   - 考虑分批处理大量文件
   - 关闭整体统计功能
   - 关闭平均图像生成功能

4. **图表显示问题**
   - 确保matplotlib正确安装
   - 在无GUI环境中使用 `--no-save` 参数

5. **平均图像生成失败**
   - 检查是否有足够的磁盘空间
   - 确认输出目录的写入权限
   - 验证数据类型设置是否正确

### 获取帮助

如果遇到问题，请检查：
1. 配置文件路径是否正确
2. 文件路径是否存在
3. 图像尺寸是否匹配
4. 依赖库是否安装完整
5. 内存是否足够（特别是启用整体统计和平均图像时）
6. 磁盘空间是否充足
7. 输出目录是否有写入权限
