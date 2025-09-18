# ISP图像PSNR计算功能

修改后的 `example_invert_ISP.py` 现在只在ISP处理后的uint8图像上计算PSNR，并可视化圆形ROI区域。

## 主要修改

### 1. 删除RAW图像PSNR计算
- 移除了RAW图像的PSNR计算功能
- 简化了处理流程
- 减少了计算开销

### 2. 专注于ISP图像PSNR
- 只在ISP处理后的uint8图像上计算PSNR
- 使用圆形ROI检测有效区域
- 提供更准确的图像质量评估

### 3. 增强可视化功能
- 在保存的图像上标记圆形ROI边界
- 显示圆心位置
- 提供matplotlib对比图

## 功能特点

### 圆形ROI检测
```python
def detect_circular_roi(image: np.ndarray, threshold: float = 0.1) -> Tuple[int, int, int]:
    """
    检测内窥镜圆视场的有效区域
    - 使用霍夫圆检测（主要方法）
    - 使用亮度检测（备用方法）
    - 返回圆心坐标和半径
    """
```

### ISP图像PSNR计算
```python
def calculate_psnr_circular_roi(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0, 
                               threshold: float = 0.1) -> Tuple[float, Tuple[int, int, int]]:
    """
    计算两张ISP图像在圆形ROI区域内的PSNR值
    - 自动检测圆形ROI
    - 只在有效区域内计算PSNR
    - 返回PSNR值和ROI信息
    """
```

## 使用方法

### 基本流程
1. 加载原始RAW图像
2. 执行第一次ISP处理
3. 执行逆ISP处理
4. 执行第二次ISP处理
5. 计算ISP图像PSNR（圆形ROI）
6. 保存对比图像和可视化结果

### 配置参数
```python
config = {
    'RESOLUTION': '1K',  # 或 '4K'
    'LENS_SHADING_ENABLED': True,
    'WHITE_BALANCE_ENABLED': True,
    'CCM_ENABLED': True,
    # 其他参数...
}
```

## 输出文件

### 保存的图像
- `original_isp.jpg`: 原始ISP处理结果（带ROI标记）
- `reconstructed_isp.jpg`: 重建ISP处理结果（带ROI标记）
- `comparison_results.png`: matplotlib对比图（带ROI可视化）

### ROI标记说明
- **绿色圆圈**: 标记检测到的圆形ROI边界
- **红色圆点**: 标记圆心位置
- **线条宽度**: 2像素

## 可视化效果

### OpenCV保存的图像
- 在BGR图像上绘制绿色圆圈和红色中心点
- 使用 `cv2.circle()` 函数绘制
- 保存为JPG格式

### Matplotlib对比图
- 1x2布局显示两张ISP图像
- 使用 `plt.Circle()` 绘制ROI边界
- 使用 `plt.plot()` 标记中心点
- 保存为PNG格式

## 技术优势

### 1. 准确性
- 只计算有效图像区域的PSNR
- 避免无效区域对结果的影响
- 提供更准确的图像质量评估

### 2. 自动化
- 自动检测圆形ROI区域
- 无需手动指定感兴趣区域
- 减少人工干预

### 3. 可视化
- 直观显示检测结果
- 便于验证ROI检测准确性
- 提供多种可视化方式

### 4. 性能
- 删除不必要的RAW图像处理
- 专注于ISP图像质量评估
- 减少计算开销

## 测试和验证

### 运行测试脚本
```bash
python test_isp_psnr_only.py
```

### 测试内容
1. ISP图像PSNR计算功能
2. 圆形ROI检测准确性
3. 可视化功能验证
4. 4K分辨率支持测试

### 预期结果
- PSNR计算正常（>15dB）
- ROI检测准确（误差<50像素）
- 可视化文件正确生成
- 4K分辨率支持正常

## 配置选项

### 分辨率选择
```python
'RESOLUTION': '1K',  # 1920x1080
'RESOLUTION': '4K',  # 3840x2160
```

### ROI检测参数
```python
threshold: float = 0.1  # ROI检测阈值
```

### 可视化参数
```python
# OpenCV绘制参数
cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)  # 绿色圆圈
cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)     # 红色中心点

# Matplotlib绘制参数
plt.Circle((center_x, center_y), radius, fill=False, color='green', linewidth=2)
plt.plot(center_x, center_y, 'ro', markersize=5)
```

## 故障排除

### 常见问题
1. **ROI检测失败**: 检查图像是否有清晰的圆形边界
2. **PSNR异常**: 确认图像尺寸匹配和数据类型正确
3. **可视化错误**: 检查OpenCV版本和图像格式
4. **文件保存失败**: 确认输出目录权限

### 调试建议
1. 使用测试脚本验证功能
2. 检查输入图像质量
3. 调整ROI检测阈值
4. 验证输出目录权限

## 性能优化

### 内存使用
- 4K图像需要更多内存
- 建议在充足内存环境下运行
- 可考虑降低分辨率进行测试

### 处理时间
- ROI检测需要一定时间
- 霍夫圆检测可能较慢
- 建议在性能较好的机器上运行

## 扩展功能

### 自定义ROI
- 可以修改检测算法
- 支持手动指定ROI参数
- 可添加其他形状的ROI检测

### 批量处理
- 支持批量图像处理
- 可添加进度显示
- 支持结果统计和分析
