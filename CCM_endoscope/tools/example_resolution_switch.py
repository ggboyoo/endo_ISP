#!/usr/bin/env python3
"""
示例：如何在 example_invert_ISP.py 中切换分辨率
"""

def show_resolution_options():
    """展示分辨率选项的使用方法"""
    print("=== 分辨率切换示例 ===")
    print()
    
    print("在 example_invert_ISP.py 中，你可以通过修改配置来切换分辨率：")
    print()
    
    print("1. 使用 1K 分辨率 (1920x1080):")
    print("   'RESOLUTION': '1K'")
    print()
    
    print("2. 使用 4K 分辨率 (3840x2160):")
    print("   'RESOLUTION': '4K'")
    print()
    
    print("配置示例：")
    print("""
    config = {
        # 图像参数 - 可选择 1K 或 4K 分辨率
        'RESOLUTION': '1K',  # 可选: '1K' (1920x1080) 或 '4K' (3840x2160)
        'DATA_TYPE': 'uint16',
        
        # 其他配置...
    }
    """)
    
    print("支持的选项：")
    print("- '1K': 1920x1080 像素")
    print("- '4K': 3840x2160 像素")
    print()
    
    print("注意事项：")
    print("1. 确保你的 RAW 文件尺寸与选择的分辨率匹配")
    print("2. 确保暗电流参考图、镜头阴影参数等也匹配相应分辨率")
    print("3. 分辨率不区分大小写，'1k' 和 '1K' 都可以")
    print("4. 如果输入不支持的分辨率，程序会报错并提示")

if __name__ == "__main__":
    show_resolution_options()
