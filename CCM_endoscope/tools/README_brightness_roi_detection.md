# 基于亮度检测的ROI区域识别

修改后的ROI检测功能完全基于亮度检测来确定内窥镜圆视场的有效区域，移除了霍夫圆检测，提供更稳定和可靠的检测结果。

## 主要修改

### 1. 移除霍夫圆检测

#### 修改前：
```python
def detect_circular_roi(image: np.ndarray, threshold: float = 0.1) -> Tuple[int, int, int]:
    # 使用Canny边缘检测
    edges = cv2.Canny((gray_norm * 255).astype(np.uint8), 50, 150)
    
    # 使用霍夫圆检测
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, ...)
    
    if circles is not None:
        # 处理霍夫圆检测结果
        return center_x, center_y, radius
    else:
        # 如果霍夫圆检测失败，使用基于亮度的方法
        return detect_circular_roi_brightness(gray_norm)
```

#### 修改后：
```python
def detect_circular_roi(image: np.ndarray, threshold: float = 0.1) -> Tuple[int, int, int]:
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 归一化到0-1
    gray_norm = gray.astype(np.float32) / 255.0
    
    # 直接使用基于亮度的方法
    print("Using brightness-based method for circular ROI detection")
    return detect_circular_roi_brightness(gray_norm, threshold)
```

### 2. 改进亮度检测算法

#### 核心算法：
```python
def detect_circular_roi_brightness(image: np.ndarray, threshold: float = 0.1) -> Tuple[int, int, int]:
    h, w = image.shape
    center_x, center_y = w // 2, h // 2
    
    # 计算中心区域的亮度作为参考
    center_region_size = min(h, w) // 20  # 中心区域大小
    center_region = image[
        center_y - center_region_size//2:center_y + center_region_size//2,
        center_x - center_region_size//2:center_x + center_region_size//2
    ]
    center_brightness = np.mean(center_region)
    
    # 从中心向外搜索，找到亮度显著下降的位置
    best_radius = 10
    best_score = 0
    
    for radius in range(10, max_radius, 2):  # 更细粒度的搜索
        # 检查圆形边界上的像素
        boundary_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > (radius - 3) ** 2
        boundary_mask = boundary_mask & ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        
        if np.sum(boundary_mask) > 0:
            boundary_brightness = np.mean(image[boundary_mask])
            # 计算亮度下降的幅度
            brightness_drop = center_brightness - boundary_brightness
            # 计算得分：亮度下降越大，得分越高
            score = brightness_drop * radius  # 考虑半径，避免选择过小的圆
            
            if score > best_score:
                best_score = score
                best_radius = radius
```

## 算法特点

### 1. 中心亮度参考
- 计算图像中心区域的亮度作为参考值
- 中心区域大小为图像最小尺寸的1/20
- 提供稳定的亮度基准

### 2. 边界亮度检测
- 从中心向外搜索，检查不同半径的圆形边界
- 计算边界像素的平均亮度
- 检测亮度显著下降的位置

### 3. 智能评分机制
```python
# 计算得分：亮度下降越大，得分越高
score = brightness_drop * radius  # 考虑半径，避免选择过小的圆
```

### 4. 细粒度搜索
- 搜索步长为2像素，提供更精确的检测
- 边界检测宽度为3像素，提高稳定性
- 避免噪声影响检测结果

## 检测流程

### 1. 图像预处理
```python
# 转换为灰度图像
if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image.copy()

# 归一化到0-1
gray_norm = gray.astype(np.float32) / 255.0
```

### 2. 中心区域分析
```python
# 计算中心区域的亮度作为参考
center_region_size = min(h, w) // 20  # 中心区域大小
center_region = image[
    center_y - center_region_size//2:center_y + center_region_size//2,
    center_x - center_region_size//2:center_x + center_region_size//2
]
center_brightness = np.mean(center_region)
```

### 3. 边界搜索
```python
for radius in range(10, max_radius, 2):  # 更细粒度的搜索
    # 检查圆形边界上的像素
    boundary_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > (radius - 3) ** 2
    boundary_mask = boundary_mask & ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    
    if np.sum(boundary_mask) > 0:
        boundary_brightness = np.mean(image[boundary_mask])
        brightness_drop = center_brightness - boundary_brightness
        score = brightness_drop * radius
        
        if score > best_score:
            best_score = score
            best_radius = radius
```

### 4. 结果验证
```python
# 如果找到了合适的半径，使用它
if best_score > threshold * center_brightness:
    return center_x, center_y, best_radius

# 如果没找到合适的半径，使用基于图像尺寸的默认值
default_radius = min(center_x, center_y, w - center_x, h - center_y)
radius = min(default_radius, min(h, w) // 3)
radius = max(10, radius)  # 确保最小半径为10
return center_x, center_y, radius
```

## 优势特点

### 1. 稳定性
- 不依赖边缘检测，避免噪声影响
- 基于亮度变化，更符合内窥镜图像特点
- 提供一致的检测结果

### 2. 准确性
- 细粒度搜索，提高检测精度
- 智能评分机制，选择最佳半径
- 考虑半径因素，避免过小的圆

### 3. 鲁棒性
- 适应不同亮度的图像
- 处理各种尺寸的图像
- 提供默认值作为备选

### 4. 效率
- 直接基于亮度检测，无需复杂计算
- 细粒度搜索，平衡精度和速度
- 减少计算开销

## 参数说明

### 1. 阈值参数
```python
threshold: float = 0.1  # 亮度阈值，用于确定圆形边界
```

### 2. 中心区域大小
```python
center_region_size = min(h, w) // 20  # 中心区域大小为图像最小尺寸的1/20
```

### 3. 搜索参数
```python
for radius in range(10, max_radius, 2):  # 搜索步长为2像素
boundary_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > (radius - 3) ** 2  # 边界检测宽度为3像素
```

### 4. 边界约束
```python
max_radius = min(center_x, center_y, w - center_x, h - center_y) - 5  # 最大半径
radius = max(10, radius)  # 最小半径
```

## 测试验证

### 测试用例
1. **不同分辨率图像测试**
   - 1K图像 (1920x1080)
   - 4K图像 (3840x2160)
   - 小图像 (640x480)

2. **边界情况测试**
   - 极小图像 (10x10)
   - 极长图像 (100x2000)
   - 极宽图像 (2000x100)

3. **功能测试**
   - 亮度检测边界检查
   - PSNR计算边界检查
   - 可视化功能测试

### 运行测试
```bash
python test_roi_boundary_check.py
```

### 预期结果
- 所有检测到的圆心坐标都在图像范围内
- 所有检测到的半径都不会超出图像边界
- PSNR计算不会因为ROI超出范围而失败
- 可视化功能正常显示ROI区域

## 使用建议

### 1. 图像要求
- 建议图像有明确的中心区域
- 中心区域应该比边界区域更亮
- 避免过度曝光或欠曝光的图像

### 2. 参数调整
- 可以根据图像特点调整阈值
- 可以修改中心区域大小
- 可以调整搜索步长和边界检测宽度

### 3. 错误处理
- 监控检测结果的日志输出
- 检查检测结果的合理性
- 必要时可以手动指定ROI参数

## 技术细节

### 1. 亮度计算
```python
# 中心区域亮度
center_brightness = np.mean(center_region)

# 边界区域亮度
boundary_brightness = np.mean(image[boundary_mask])

# 亮度下降
brightness_drop = center_brightness - boundary_brightness
```

### 2. 评分机制
```python
# 计算得分：亮度下降越大，得分越高
score = brightness_drop * radius  # 考虑半径，避免选择过小的圆
```

### 3. 边界约束
```python
# 确保圆心在图像范围内
center_x = max(0, min(center_x, w - 1))
center_y = max(0, min(center_y, h - 1))

# 确保半径不会超出图像边界
max_radius = min(center_x, center_y, w - center_x, h - center_y)
radius = min(radius, max_radius)
```

## 总结

基于亮度检测的ROI区域识别提供了：

1. **稳定性**: 不依赖边缘检测，避免噪声影响
2. **准确性**: 细粒度搜索，智能评分机制
3. **鲁棒性**: 适应不同图像，提供默认值
4. **效率**: 直接基于亮度，减少计算开销

这种方法特别适合内窥镜图像的特点，能够准确识别圆视场的有效区域，为PSNR计算提供可靠的ROI信息。
