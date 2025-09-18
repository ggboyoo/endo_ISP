# 分辨率选项配置

`example_invert_ISP.py` 现在支持灵活的分辨率选择，可以在 1K 和 4K 之间切换。

## 支持的分辨率

| 选项 | 分辨率 | 像素数 | 说明 |
|------|--------|--------|------|
| `'1K'` | 1920x1080 | 2,073,600 | 标准 1K 分辨率 |
| `'4K'` | 3840x2160 | 8,294,400 | 标准 4K 分辨率 |

## 配置方法

在 `example_invert_ISP.py` 中修改配置：

```python
config = {
    # 图像参数 - 可选择 1K 或 4K 分辨率
    'RESOLUTION': '1K',  # 可选: '1K' (1920x1080) 或 '4K' (3840x2160)
    'DATA_TYPE': 'uint16',
    
    # 其他配置...
}
```

## 使用示例

### 1K 分辨率配置
```python
config = {
    'RESOLUTION': '1K',  # 使用 1K 分辨率
    'DATA_TYPE': 'uint16',
    # 其他参数...
}
```

### 4K 分辨率配置
```python
config = {
    'RESOLUTION': '4K',  # 使用 4K 分辨率
    'DATA_TYPE': 'uint16',
    # 其他参数...
}
```

## 自动解析

程序会自动根据 `RESOLUTION` 设置解析出对应的宽度和高度：

```python
# 解析分辨率参数
resolution = config['RESOLUTION'].upper()
if resolution == '1K':
    config['IMAGE_WIDTH'] = 1920
    config['IMAGE_HEIGHT'] = 1080
    print("Using 1K resolution: 1920x1080")
elif resolution == '4K':
    config['IMAGE_WIDTH'] = 3840
    config['IMAGE_HEIGHT'] = 2160
    print("Using 4K resolution: 3840x2160")
else:
    raise ValueError(f"Unsupported resolution: {resolution}. Please use '1K' or '4K'")
```

## 注意事项

1. **文件匹配**: 确保你的 RAW 文件尺寸与选择的分辨率匹配
2. **参数匹配**: 确保暗电流参考图、镜头阴影参数等也匹配相应分辨率
3. **大小写不敏感**: 分辨率选项不区分大小写，`'1k'` 和 `'1K'` 都可以
4. **错误处理**: 如果输入不支持的分辨率，程序会报错并提示

## 错误处理

如果输入不支持的分辨率，程序会抛出错误：

```python
# 错误示例
config = {'RESOLUTION': '2K'}  # 不支持的分辨率

# 错误信息
ValueError: Unsupported resolution: 2K. Please use '1K' or '4K'
```

## 扩展性

如果需要添加更多分辨率，可以在解析逻辑中添加新的条件：

```python
elif resolution == '2K':
    config['IMAGE_WIDTH'] = 2560
    config['IMAGE_HEIGHT'] = 1440
    print("Using 2K resolution: 2560x1440")
```

## 性能考虑

- **1K 分辨率**: 处理速度较快，内存占用较少
- **4K 分辨率**: 处理速度较慢，内存占用较多，但图像质量更高

根据你的硬件配置和需求选择合适的分辨率。
