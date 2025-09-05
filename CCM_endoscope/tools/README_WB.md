# White Balance Calculator with Manual ROI Selection

这个工具用于计算白平衡参数，支持手动框选标准白板区域来计算白平衡增益。它集成了完整的ISP流程，包括暗电流校正、镜头阴影校正、去马赛克等步骤。

## 功能特点

- 🔧 **完整ISP流程**: 集成暗电流校正、镜头阴影校正、去马赛克
- 🖱️ **手动ROI选择**: 支持鼠标拖拽选择白板区域
- 📊 **实时预览**: ROI选择时实时显示选择区域
- 🎯 **精确计算**: 基于选定区域计算白平衡增益
- 💾 **多格式输出**: 保存参数、图像和分析图表
- 📈 **详细分析**: 提供直方图对比和统计分析

## 文件结构

```
CCM_endoscope/tools/
├── WB.py                    # 主程序
├── ISP.py                   # ISP处理流程
├── lens_shading.py          # 镜头阴影校正
├── raw_reader.py            # RAW图像读取
└── README_WB.md            # 本说明文档
```

## 使用方法

### 基本用法

```bash
# 使用默认配置
python WB.py

# 指定输入图像
python WB.py --input "F:\ZJU\Picture\white_board.raw"

# 指定图像尺寸
python WB.py --input "F:\ZJU\Picture\white_board.raw" --width 3840 --height 2160

# 指定输出目录
python WB.py --input "F:\ZJU\Picture\white_board.raw" --output "F:\ZJU\Picture\wb_results"
```

### 高级用法

```bash
# 禁用暗电流校正
python WB.py --no-dark

# 禁用镜头阴影校正
python WB.py --no-lens-shading

# 指定暗电流参考图
python WB.py --dark "F:\ZJU\Picture\dark\average_dark.raw"

# 指定镜头阴影参数目录
python WB.py --lens-shading "F:\ZJU\Picture\lens_shading\params"
```

## 配置参数

### 输入配置

- **INPUT_IMAGE_PATH**: 输入RAW图像路径
- **IMAGE_WIDTH/HEIGHT**: 图像尺寸
- **DATA_TYPE**: 数据类型（uint16: 12位数据存储在16位容器中，0-4095）

### ISP配置

- **DARK_RAW_PATH**: 暗电流参考图路径
- **DARK_SUBTRACTION_ENABLED**: 是否启用暗电流校正
- **LENS_SHADING_PARAMS_DIR**: 镜头阴影参数目录
- **LENS_SHADING_ENABLED**: 是否启用镜头阴影校正
- **BAYER_PATTERN**: 拜尔模式（rggb, bggr, grbg, gbrg）

### 输出配置

- **OUTPUT_DIRECTORY**: 输出目录（None表示自动创建）
- **SAVE_RESULTS**: 是否保存结果
- **SAVE_IMAGES**: 是否保存图像
- **SAVE_PARAMETERS**: 是否保存参数

## 核心功能

### 1. ISP处理流程

程序调用ISP.py中的功能进行完整的ISP处理流程：

1. **RAW图像读取**: 读取12位RAW数据
2. **暗电流校正**: 减去暗电流参考
3. **镜头阴影校正**: 应用镜头阴影校正
4. **8位归一化**: 仅用于去马赛克
5. **去马赛克**: 转换为彩色图像

**注意**: WB.py不重复实现ISP功能，而是直接调用ISP.py中的 `process_single_image` 函数，确保处理流程的一致性。

### 2. 手动ROI选择

- **鼠标操作**: 点击拖拽选择白板区域
- **实时预览**: 选择过程中实时显示选择框
- **键盘控制**: 
  - `Enter`: 确认选择
  - `Esc`: 取消选择
  - `r`: 重置选择

### 3. 白平衡计算

基于选定ROI区域计算白平衡增益：

```python
# 计算各通道平均值
b_mean = np.mean(roi_region[:, :, 0])
g_mean = np.mean(roi_region[:, :, 1])
r_mean = np.mean(roi_region[:, :, 2])

# 计算白平衡增益（以G通道为参考）
b_gain = g_mean / b_mean
g_gain = 1.0
r_gain = g_mean / r_mean
```

### 4. 结果输出

程序会生成以下输出文件：

- **参数文件**: `wb_parameters_manual_roi_时间戳.json`
- **原始图像**: `original_image_时间戳.png`
- **校正图像**: `wb_corrected_manual_roi_时间戳.png`
- **分析图表**: `wb_analysis_manual_roi_时间戳.png`

## 输出结果

### 参数文件格式

```json
{
  "white_balance_gains": {
    "b_gain": 1.234,
    "g_gain": 1.000,
    "r_gain": 0.876
  },
  "method": "manual_roi",
  "roi_coordinates": {
    "x1": 100,
    "y1": 100,
    "x2": 200,
    "y2": 200
  },
  "timestamp": "2024-12-01T14:30:22",
  "description": "White balance parameters calculated using manual ROI selection method"
}
```

### 分析图表

程序会生成包含以下内容的分析图表：

1. **原始图像**: 显示ROI选择区域
2. **校正后图像**: 白平衡校正后的结果
3. **差异图像**: 原始与校正的差异
4. **直方图对比**: 各通道校正前后的直方图

## 应用场景

### 1. 标准白板校正

- 使用标准白板图像
- 手动选择白板区域
- 计算精确的白平衡参数

### 2. 自定义区域校正

- 选择图像中的中性区域
- 计算该区域的白平衡参数
- 适用于非标准白板场景

### 3. 批量处理准备

- 计算单张图像的白平衡参数
- 保存参数用于批量处理
- 验证校正效果

## 技术原理

### 1. ISP模块调用

程序直接调用ISP.py中的功能，确保处理流程的一致性：

```python
# 调用ISP.py中的功能
from ISP import process_single_image, load_dark_reference
from lens_shading import load_correction_parameters

# 加载参数
dark_data = load_dark_reference(dark_path, width, height, data_type)
lens_shading_params = load_correction_parameters(lens_shading_dir)

# 使用ISP处理
isp_result = process_single_image(raw_file, dark_data, lens_shading_params, width, height, data_type)
```

### 2. ROI选择机制

使用OpenCV的鼠标回调函数实现ROI选择：

```python
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 开始选择
    elif event == cv2.EVENT_MOUSEMOVE:
        # 更新预览
    elif event == cv2.EVENT_LBUTTONUP:
        # 完成选择
```

### 3. 白平衡算法

使用灰度世界法的变种，基于选定区域计算：

- **参考通道**: 以G通道为参考（增益=1.0）
- **计算方式**: 其他通道增益 = G通道均值 / 该通道均值
- **应用方式**: 像素值 × 对应通道增益

## 注意事项

### 1. 图像要求

- **RAW格式**: 支持12位RAW数据
- **拜尔模式**: 默认为RGGB，可配置
- **尺寸匹配**: 输入尺寸必须与配置一致

### 2. ROI选择

- **选择区域**: 应选择图像中的中性区域
- **区域大小**: 建议选择足够大的区域以获得稳定结果
- **避免边缘**: 避免选择图像边缘区域

### 3. 参数配置

- **暗电流**: 确保暗电流参考图与主图像条件一致
- **镜头阴影**: 确保镜头阴影参数与当前镜头匹配
- **输出目录**: 确保有足够的磁盘空间

## 故障排除

### 1. 常见错误

- **文件不存在**: 检查输入文件路径
- **尺寸不匹配**: 检查图像尺寸配置
- **ROI选择失败**: 确保正确完成ROI选择

### 2. 性能优化

- **大图像**: 对于大图像，ROI选择可能需要较长时间
- **内存使用**: 16位处理会占用更多内存
- **显示性能**: 调整显示窗口大小以改善性能

### 3. 结果验证

- **增益范围**: 白平衡增益通常在0.5-2.0范围内
- **图像质量**: 检查校正后的图像质量
- **参数保存**: 验证参数文件是否正确保存

## 扩展应用

### 1. 批量处理

可以将计算得到的白平衡参数用于批量处理其他图像：

```python
# 加载参数
with open('wb_parameters.json', 'r') as f:
    params = json.load(f)

# 应用白平衡
b_gain = params['white_balance_gains']['b_gain']
g_gain = params['white_balance_gains']['g_gain']
r_gain = params['white_balance_gains']['r_gain']
```

### 2. 参数优化

可以基于多张图像计算平均白平衡参数：

```python
# 计算多张图像的平均增益
avg_b_gain = np.mean([gain1['b_gain'], gain2['b_gain'], gain3['b_gain']])
avg_g_gain = 1.0
avg_r_gain = np.mean([gain1['r_gain'], gain2['r_gain'], gain3['r_gain']])
```

### 3. 自动化集成

可以将此工具集成到自动化处理流程中，实现批量白平衡校正。
