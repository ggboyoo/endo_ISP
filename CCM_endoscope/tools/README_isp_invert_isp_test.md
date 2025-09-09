# ISP-逆ISP对比测试脚本

## 概述

这些脚本实现了完整的ISP-逆ISP-ISP测试流程，用于验证逆ISP处理的准确性和ISP处理的一致性。

## 测试流程

```
原始RAW → ISP处理 → sRGB图像 → 逆ISP处理 → 重建RAW → ISP处理 → 重建sRGB图像
```

## 脚本说明

### 1. `isp_invert_isp_comparison.py` - 完整测试脚本

**功能**：
- 完整的ISP-逆ISP-ISP测试流程
- 计算RAW图像和ISP图像的PSNR
- 生成详细的对比图像和报告
- 支持所有ISP参数配置

**使用方法**：
```bash
python isp_invert_isp_comparison.py
```

**输出**：
- `original_raw.png` - 原始RAW图像（灰度图）
- `reconstructed_raw.png` - 重建RAW图像（灰度图）
- `original_isp.jpg` - 第一次ISP处理结果
- `reconstructed_isp.jpg` - 第二次ISP处理结果
- `comparison_results.png` - 对比图
- `isp_invert_isp_report.json` - 详细测试报告

### 2. `quick_isp_test.py` - 快速测试脚本

**功能**：
- 简化的测试流程
- 快速验证功能
- 基本的PSNR计算

**使用方法**：
```bash
python quick_isp_test.py
```

## 配置参数

### 输入输出配置
```python
'INPUT_RAW_PATH': r"F:\ZJU\Picture\ccm\25-09-01 160527.raw"  # 输入RAW文件
'OUTPUT_DIRECTORY': r"F:\ZJU\Picture\isp_invert_isp_test"     # 输出目录
```

### 图像参数
```python
'IMAGE_WIDTH': 3840      # 图像宽度
'IMAGE_HEIGHT': 2160     # 图像高度
'DATA_TYPE': 'uint16'    # 数据类型
```

### ISP参数路径
```python
'DARK_RAW_PATH': r"F:\ZJU\Picture\dark\g3\average_dark.raw"
'LENS_SHADING_PARAMS_DIR': r"F:\ZJU\Picture\lens shading\new"
'WB_PARAMS_PATH': r"F:\ZJU\Picture\wb\wb_output"
'CCM_MATRIX_PATH': r"F:\ZJU\Picture\ccm\ccm_2\ccm_output_20250905_162714"
```

### 直接参数（优先使用）
```python
'ccm_matrix': np.array([...])  # CCM矩阵
'wb_params': {...}             # 白平衡参数
```

## PSNR计算

### RAW图像PSNR
- 使用16bit精度计算
- 最大值为4095（12bit数据在16bit容器中）
- 公式：`PSNR = 20*log10(4095) - 10*log10(MSE)`

### ISP图像PSNR
- 使用8bit精度计算
- 最大值为255
- 公式：`PSNR = 20*log10(255) - 10*log10(MSE)`

## 结果解读

### PSNR值含义
- **> 40 dB**: 优秀质量，几乎无差异
- **30-40 dB**: 良好质量，轻微差异
- **20-30 dB**: 可接受质量，明显差异
- **< 20 dB**: 质量较差，显著差异

### 测试目标
- **RAW PSNR**: 验证逆ISP重建RAW的准确性
- **ISP PSNR**: 验证ISP处理的一致性

## 故障排除

### 常见问题

1. **模块导入失败**
   - 确保所有依赖模块在同一目录
   - 检查Python路径设置

2. **文件路径错误**
   - 检查输入RAW文件是否存在
   - 验证参数文件路径是否正确

3. **内存不足**
   - 使用较小的测试图像
   - 关闭不必要的程序

4. **PSNR计算失败**
   - 检查图像尺寸是否匹配
   - 验证图像数据类型

### 调试建议

1. 先运行 `quick_isp_test.py` 验证基本功能
2. 检查中间结果图像的质量
3. 对比原始和重建图像的直方图
4. 验证ISP参数是否正确加载

## 扩展功能

### 自定义测试
可以修改脚本中的配置参数来测试不同的：
- 图像尺寸
- ISP参数
- 处理开关
- 输出格式

### 批量测试
可以修改脚本支持批量处理多个RAW文件。

### 详细分析
可以添加更多分析功能：
- 直方图对比
- 频域分析
- 局部PSNR计算
- 颜色空间分析
