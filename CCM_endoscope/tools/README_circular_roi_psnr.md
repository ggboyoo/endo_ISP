# 圆形ROI PSNR计算功能

针对内窥镜圆视场图像，实现了基于圆形感兴趣区域（ROI）的PSNR计算功能。

## 功能特点

### 1. 自动圆形ROI检测
- **霍夫圆检测**: 使用OpenCV的霍夫圆变换检测圆形边界
- **亮度检测**: 备用方法，基于亮度变化检测圆形区域
- **自适应参数**: 根据图像尺寸自动调整检测参数

### 2. 精确PSNR计算
- **区域限制**: 只在检测到的圆形区域内计算PSNR
- **像素统计**: 显示ROI内的有效像素数量
- **高精度**: 使用float64精度进行计算

### 3. 可视化支持
- **ROI标记**: 在保存的图像上标记检测到的圆形ROI
- **颜色区分**: RAW图像使用白色圆圈，ISP图像使用绿色圆圈
- **中心点标记**: 标记圆心位置

## 使用方法

### 基本用法

```python
from example_invert_ISP import calculate_psnr_circular_roi

# 计算圆形ROI PSNR
psnr, roi_info = calculate_psnr_circular_roi(img1, img2, max_val=255.0)
print(f"PSNR: {psnr:.2f} dB")
print(f"ROI: center=({roi_info[0]}, {roi_info[1]}), radius={roi_info[2]}")
```

### 参数说明

- `img1`, `img2`: 输入图像（支持灰度或彩色）
- `max_val`: 图像的最大可能值（默认255.0）
- `threshold`: ROI检测阈值（默认0.1）

### 返回值

- `psnr`: PSNR值（dB）
- `roi_info`: ROI信息元组 `(center_x, center_y, radius)`

## 检测算法

### 1. 霍夫圆检测（主要方法）

```python
circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=min(gray.shape) // 4,
    param1=50,
    param2=30,
    minRadius=min(gray.shape) // 8,
    maxRadius=min(gray.shape) // 2
)
```

**参数说明**:
- `dp`: 累加器分辨率与图像分辨率的反比
- `minDist`: 检测到的圆心之间的最小距离
- `param1`: Canny边缘检测的高阈值
- `param2`: 累加器阈值
- `minRadius`, `maxRadius`: 圆的最小和最大半径

### 2. 亮度检测（备用方法）

```python
for radius in range(10, max_radius, 5):
    boundary_brightness = np.mean(image[boundary_mask])
    if boundary_brightness < threshold:
        return center_x, center_y, radius
```

**特点**:
- 从中心向外搜索
- 检测亮度显著下降的位置
- 适用于霍夫圆检测失败的情况

## 在example_invert_ISP.py中的应用

### RAW图像PSNR计算

```python
# 计算RAW图像的PSNR（使用圆形ROI）
raw_psnr, raw_roi_info = calculate_psnr_circular_roi(raw_data, reconstructed_raw, 4095.0)
print(f"RAW PSNR (circular ROI): {raw_psnr:.2f} dB")
print(f"ROI info: center=({raw_roi_info[0]}, {raw_roi_info[1]}), radius={raw_roi_info[2]}")
```

### ISP图像PSNR计算

```python
# 计算ISP图像的PSNR（使用圆形ROI）
isp_psnr, isp_roi_info = calculate_psnr_circular_roi(isp_result['color_img'], second_isp_result['color_img'], 255.0)
print(f"ISP PSNR (circular ROI): {isp_psnr:.2f} dB")
print(f"ROI info: center=({isp_roi_info[0]}, {isp_roi_info[1]}), radius={isp_roi_info[2]}")
```

## 可视化功能

### 保存带ROI标记的图像

```python
save_comparison_images(
    raw_data, reconstructed_raw,
    isp_result['color_img'], second_isp_result['color_img'],
    output_dir, raw_roi_info, isp_roi_info
)
```

**标记说明**:
- **RAW图像**: 白色圆圈标记ROI边界
- **ISP图像**: 绿色圆圈标记ROI边界
- **中心点**: 红色圆点标记圆心

## 测试和验证

### 运行测试脚本

```bash
python test_circular_roi.py
```

**测试内容**:
1. 1K分辨率圆形ROI检测
2. 4K分辨率圆形ROI检测
3. PSNR计算精度验证
4. 可视化功能测试

### 预期结果

- **1K分辨率**: 检测误差 < 50像素
- **4K分辨率**: 检测误差 < 100像素
- **PSNR计算**: 噪声图像PSNR > 20dB

## 优势

1. **准确性**: 只计算有效图像区域的PSNR
2. **自动化**: 无需手动指定ROI区域
3. **鲁棒性**: 多种检测方法确保成功率
4. **可视化**: 直观显示检测结果
5. **兼容性**: 支持不同分辨率和图像格式

## 注意事项

1. **图像质量**: 确保输入图像有清晰的圆形边界
2. **参数调整**: 可根据具体图像调整检测阈值
3. **内存使用**: 4K图像处理需要更多内存
4. **处理时间**: 霍夫圆检测可能需要较长时间

## 故障排除

### 检测失败
- 检查图像是否有清晰的圆形边界
- 尝试调整`threshold`参数
- 使用亮度检测作为备用方法

### PSNR异常
- 确认图像尺寸匹配
- 检查ROI内是否有有效像素
- 验证图像数据类型和范围

### 可视化问题
- 确认ROI信息正确传递
- 检查图像保存路径权限
- 验证OpenCV版本兼容性
