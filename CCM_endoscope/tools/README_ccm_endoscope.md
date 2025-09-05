# CCM Endoscope Calibration Program

这个程序用于色卡RAW图的CCM（Color Correction Matrix）标定。它集成了完整的ISP处理流程，包括暗电流校正、镜头阴影校正、白平衡校正，然后进行CCM标定。

## 功能特点

- 🔧 **完整ISP流程**: 集成暗电流校正、镜头阴影校正、白平衡校正
- 🎨 **色块提取**: 自动从色卡图像中提取24个色块
- 📊 **数据归一化**: 将16位数据归一化到浮点范围(0, 255)
- 🎯 **CCM标定**: 使用梯度优化方法计算CCM矩阵
- 📈 **详细分析**: 提供DeltaE误差分析和可视化图表
- 💾 **多格式输出**: 保存参数、图像和分析结果

## 文件结构

```
CCM_endoscope/tools/
├── ccm_endoscope.py           # 主程序
├── ISP.py                     # ISP处理流程
├── ccm_calculator.py          # CCM计算功能
├── lens_shading.py            # 镜头阴影校正
├── raw_reader.py              # RAW图像读取
└── README_ccm_endoscope.md    # 本说明文档
```

## 使用方法

### 基本用法

```bash
# 使用默认配置
python ccm_endoscope.py

# 指定输入图像
python ccm_endoscope.py --input "F:\ZJU\Picture\ccm\25-09-01 160530.raw"

# 指定图像尺寸
python ccm_endoscope.py --input "F:\ZJU\Picture\ccm\25-09-01 160530.raw" --width 3840 --height 2160

# 指定输出目录
python ccm_endoscope.py --input "F:\ZJU\Picture\ccm\25-09-01 160530.raw" --output "F:\ZJU\Picture\ccm_results"
```

### 交互式操作流程

程序运行时会进入交互模式，按以下步骤操作：

1. **色卡区域选择**：
   - 程序显示ISP处理后的彩色图像
   - 鼠标拖拽选择色卡区域
   - 按 `Enter` 确认选择
   - 按 `r` 重置选择
   - 按 `Esc` 退出程序

2. **色块预览检查**：
   - 程序显示提取的24个色块边界和编号
   - 按 `Enter` 继续标定
   - 按 `Esc` 取消操作

3. **色块网格预览**：
   - 显示4x6色块网格图
   - 打印每个色块的RGB值
   - 按任意键继续

### 高级用法

```bash
# 指定CCM计算方法
python ccm_endoscope.py --method gradient_optimization

# 禁用某些功能
python ccm_endoscope.py --no-dark --no-lens-shading --no-wb

# 指定各种参数文件
python ccm_endoscope.py --dark "F:\ZJU\Picture\dark\average_dark.raw" --lens-shading "F:\ZJU\Picture\lens_shading" --wb-params "F:\ZJU\Picture\wb\wb_parameters.json"
```

## 配置参数

### 输入配置

- **INPUT_IMAGE_PATH**: 输入色卡RAW图像路径
- **IMAGE_WIDTH/HEIGHT**: 图像尺寸
- **DATA_TYPE**: 数据类型（uint16: 12位数据存储在16位容器中，0-4095）

### ISP配置

- **DARK_RAW_PATH**: 暗电流参考图路径
- **DARK_SUBTRACTION_ENABLED**: 是否启用暗电流校正
- **LENS_SHADING_PARAMS_DIR**: 镜头阴影参数目录
- **LENS_SHADING_ENABLED**: 是否启用镜头阴影校正
- **BAYER_PATTERN**: 拜尔模式（rggb, bggr, grbg, gbrg）

### 白平衡配置

- **WHITE_BALANCE_ENABLED**: 是否启用白平衡矫正
- **WB_PARAMETERS_PATH**: 白平衡参数文件路径

### CCM配置

- **CCM_METHOD**: CCM计算方法（gradient_optimization, linear_regression）
- **WHITE_BALANCE_ENABLED_CCM**: CCM计算时是否启用白平衡
- **LUMINANCE_NORMALIZATION**: 是否启用亮度归一化
- **WHITE_PRESERVATION_CONSTRAINT**: 是否启用白色保持约束
- **PATCH_19_20_WHITE_BALANCE**: 是否使用第19、20色块进行白平衡

### 输出配置

- **OUTPUT_DIRECTORY**: 输出目录（None表示自动创建）
- **SAVE_RESULTS**: 是否保存结果
- **SAVE_IMAGES**: 是否保存图像
- **SAVE_PARAMETERS**: 是否保存参数
- **GENERATE_PLOTS**: 是否生成图表

## 核心功能

### 1. ISP处理流程

程序调用ISP.py中的功能进行完整的ISP处理：

```
原始RAW图像 → 暗电流校正 → 镜头阴影校正 → 8位归一化 → 去马赛克 → 白平衡校正 → 彩色图像
```

### 2. 交互式色块提取

程序提供交互式色块提取功能：

1. **手动框选色卡区域**：
   - 鼠标拖拽选择色卡区域
   - 支持重置和取消操作
   - 实时显示选择框

2. **色块预览和检查**：
   - 显示提取的24个色块边界和编号
   - 创建色块网格预览图
   - 打印每个色块的RGB值

3. **用户确认**：
   - 检查色块提取准确性
   - 确认无误后继续标定

```python
# 交互式色块提取
patches = extract_color_patches(color_image, patch_size=50)
```

### 3. 数据归一化

将16位数据归一化到浮点范围(0, 255)：

```python
# 16位数据归一化到浮点范围(0, 255)
normalized = data.astype(np.float32)
normalized = (normalized / 4095.0 * 255.0)  # 12位数据范围
```

### 4. CCM计算

使用梯度优化方法计算CCM矩阵：

```python
# 调用ccm_calculator.py中的函数
ccm_result = solve_ccm_gradient_optimization(
    measured_patches=measured_patches_normalized,
    reference_patches=reference_patches,
    measured_is_srgb=False,
    reference_is_srgb=True,
    luminance_normalization=True,
    white_preservation_constraint=True
)
```

### 5. 结果输出

程序会生成以下输出文件：

- **CCM参数文件**: `ccm_parameters_时间戳.json`
- **色卡图像**: `colorcheck_processed_时间戳.png`
- **色块对比图**: `patch_comparison_时间戳.png`
- **分析图表**: `ccm_analysis_时间戳.png`

## 输出结果

### 参数文件格式

```json
{
  "ccm_matrix": [[1.234, 0.123, -0.456], [0.234, 1.123, 0.456], [-0.123, 0.456, 1.234]],
  "ccm_type": "linear3x3",
  "delta_e_error": 2.345,
  "white_balance_gains": {
    "r_gain": 1.234,
    "g_gain": 1.000,
    "b_gain": 0.876
  },
  "luminance_normalization": true,
  "white_preservation_constraint": true,
  "timestamp": "2024-12-01T14:30:22",
  "description": "CCM calibration parameters calculated using gradient optimization"
}
```

### 分析图表

程序会生成包含以下内容的分析图表：

1. **色块对比散点图**: 显示测量值与参考值的相关性
2. **DeltaE误差分布**: 显示各色块的DeltaE误差分布
3. **CCM矩阵可视化**: 显示CCM矩阵的热力图
4. **标定摘要**: 显示标定参数和结果摘要

## 应用场景

### 1. 相机标定

- 使用标准24色卡进行相机CCM标定
- 获得精确的颜色校正矩阵
- 用于后续图像处理流程

### 2. 颜色精度验证

- 验证相机系统的颜色精度
- 分析DeltaE误差分布
- 优化图像处理参数

### 3. 批量标定

- 对多台相机进行CCM标定
- 比较不同相机的颜色表现
- 建立标准化的颜色处理流程

## 技术原理

### 1. 交互式色块提取算法

```python
# 1. 用户交互选择色卡区域
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        # 实时显示选择框
        cv2.rectangle(display_image, start_point, (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        selection_rect = (x, y, width, height)

# 2. 根据选择区域计算色块位置
rows, cols = 4, 6
patch_h = h_rect // rows
patch_w = w_rect // cols

# 3. 提取每个色块并计算平均RGB值
for i in range(rows):
    for j in range(cols):
        patch_x = x + j * patch_w
        patch_y = y + i * patch_h
        patch_region = color_image[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
        avg_rgb = np.mean(patch_region, axis=(0, 1))
```

### 2. 数据归一化

```python
# 12位数据归一化到(0, 255)范围
normalized = data.astype(np.float32)
normalized = np.clip(normalized, 0, 4095)
normalized = (normalized / 4095.0 * 255.0)
```

### 3. CCM计算

使用梯度优化方法在Lab颜色空间中最小化DeltaE误差：

```python
# 梯度优化目标函数
def compute_delta_e_loss(ccm_matrix, measured_patches, reference_patches):
    # 应用CCM矩阵
    corrected_patches = apply_ccm(measured_patches, ccm_matrix)
    
    # 转换到Lab空间
    corrected_lab = rgb_to_lab(corrected_patches)
    reference_lab = rgb_to_lab(reference_patches)
    
    # 计算DeltaE误差
    delta_e = compute_delta_e(corrected_lab, reference_lab)
    return np.mean(delta_e)
```

## 注意事项

### 1. 图像要求

- **RAW格式**: 支持12位RAW数据
- **色卡标准**: 使用标准24色卡
- **拍摄条件**: 确保均匀光照和正确的色卡位置

### 2. 交互式操作

- **色卡选择**: 确保选择的区域完全包含24个色块
- **色块检查**: 仔细检查提取的色块是否与标准色卡对应
- **操作确认**: 在确认色块提取无误后再进行CCM标定

### 3. 参数配置

- **暗电流**: 确保暗电流参考图与主图像条件一致
- **镜头阴影**: 确保镜头阴影参数与当前镜头匹配
- **白平衡**: 确保白平衡参数正确加载

### 4. 结果验证

- **DeltaE误差**: 通常应小于3.0
- **CCM矩阵**: 检查矩阵的合理性
- **色块对比**: 验证色块提取的准确性

## 故障排除

### 1. 常见错误

- **色块提取失败**: 检查色卡位置和大小
- **交互窗口无响应**: 确保OpenCV窗口获得焦点
- **色块选择不准确**: 重新选择色卡区域
- **CCM计算失败**: 检查输入数据质量
- **参数加载失败**: 检查文件路径和格式

### 2. 性能优化

- **大图像**: 对于大图像，色块提取可能需要较长时间
- **内存使用**: 16位处理会占用更多内存
- **计算时间**: 梯度优化可能需要较长时间

### 3. 结果分析

- **DeltaE误差**: 分析误差分布和异常值
- **色块质量**: 检查色块提取的准确性
- **CCM合理性**: 验证CCM矩阵的数值范围

## 扩展应用

### 1. 批量标定

可以将此工具用于批量标定多台相机：

```python
# 批量处理多个色卡图像
for image_path in colorcheck_images:
    result = process_colorcheck_image(image_path, config)
    # 保存结果
```

### 2. 参数优化

可以基于多张图像优化CCM参数：

```python
# 计算多张图像的平均CCM
avg_ccm = np.mean([ccm1, ccm2, ccm3], axis=0)
```

### 3. 自动化集成

可以将此工具集成到自动化标定流程中，实现批量CCM标定。
