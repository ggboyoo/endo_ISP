# process_single_image 参数类型支持

`process_single_image` 函数现在支持灵活的参数类型，可以接受路径字符串或预加载的数据值。

## 支持的参数类型

### raw_file 参数
- **字符串路径**: `"path/to/image.raw"` - 从文件加载 RAW 数据
- **numpy 数组**: `np.ndarray` - 直接使用提供的 RAW 数据

### 其他参数
- **dark_data**: 字符串路径或 numpy 数组
- **lens_shading_params**: 字符串路径或参数字典
- **wb_params**: 字符串路径或参数字典
- **ccm_matrix**: numpy 数组（直接提供矩阵值）

## 使用示例

### 1. 使用文件路径

```python
from ISP import process_single_image

result = process_single_image(
    raw_file="image.raw",  # 文件路径
    dark_data="dark.raw",  # 文件路径
    lens_shading_params="lens_dir",  # 目录路径
    wb_params="wb.json",  # 文件路径
    ccm_matrix_path="ccm.json",  # 文件路径
    width=3840,
    height=2160,
    data_type='uint16'
)
```

### 2. 使用数据数组

```python
import numpy as np
from ISP import process_single_image

# 创建或加载数据
raw_data = np.random.randint(0, 4096, size=(2160, 3840), dtype=np.uint16)
dark_data = np.random.randint(0, 100, size=(2160, 3840), dtype=np.uint16)
ccm_matrix = np.array([[1.5, -0.3, 0.1], [-0.1, 1.2, -0.2], [0.05, -0.1, 1.1]])

result = process_single_image(
    raw_file=raw_data,  # 数据数组
    dark_data=dark_data,  # 数据数组
    ccm_matrix=ccm_matrix,  # 数据数组
    width=3840,
    height=2160,
    data_type='uint16'
)
```

### 3. 混合使用

```python
from ISP import process_single_image
import numpy as np

# 部分参数使用数据，部分使用路径
result = process_single_image(
    raw_file=raw_data_array,  # 数据数组
    dark_data="dark.raw",  # 文件路径
    lens_shading_params=lens_params_dict,  # 数据字典
    wb_params="wb.json",  # 文件路径
    ccm_matrix=ccm_array,  # 数据数组
    width=3840,
    height=2160,
    data_type='uint16'
)
```

## 优势

1. **灵活性**: 支持多种数据输入方式
2. **性能**: 避免重复加载已存在的数据
3. **便利性**: 自动处理参数加载
4. **兼容性**: 保持向后兼容

## 错误处理

如果 `raw_file` 不是字符串或 numpy 数组，函数会返回错误：

```python
result = process_single_image(raw_file=123)  # 无效类型
# 返回: {'processing_success': False, 'error': 'raw_file must be a string path or numpy array'}
```

## 注意事项

1. 当使用数据数组时，确保数据格式正确
2. 当使用文件路径时，确保文件存在且可读
3. 混合使用时，路径参数会被自动加载，数据参数会被直接使用
4. 如果同时提供 `ccm_matrix` 和 `ccm_matrix_path`，优先使用 `ccm_matrix`
