#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试invert_ISP.py对中文路径的支持
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_chinese_path_support():
    """测试中文路径支持"""
    print("测试 invert_ISP.py 对中文路径的支持...")
    
    # 创建临时目录，包含中文字符
    temp_dir = tempfile.mkdtemp(prefix="测试_")
    print(f"创建临时目录: {temp_dir}")
    
    try:
        # 复制测试图像到中文路径
        test_image = "Colorcheck_1_raw_1004W_ccm.jpg"
        if os.path.exists(test_image):
            chinese_path = os.path.join(temp_dir, "测试图像.jpg")
            shutil.copy2(test_image, chinese_path)
            print(f"复制测试图像到: {chinese_path}")
            
            # 测试图像读取
            from invert_ISP import load_image_as_12bit, imread_unicode
            
            print("\n测试图像读取...")
            try:
                img_12bit, width, height = load_image_as_12bit(chinese_path)
                print(f"✅ 成功读取图像: {width}x{height}, 数据类型: {img_12bit.dtype}")
            except Exception as e:
                print(f"❌ 读取图像失败: {e}")
                return False
            
            # 测试图像保存
            print("\n测试图像保存...")
            try:
                from invert_ISP import save_intermediate_image
                output_path = os.path.join(temp_dir, "输出图像.png")
                save_intermediate_image(img_12bit, output_path, is_12bit=True)
                print(f"✅ 成功保存图像: {output_path}")
            except Exception as e:
                print(f"❌ 保存图像失败: {e}")
                return False
            
            print("\n✅ 中文路径支持测试通过！")
            return True
        else:
            print(f"❌ 测试图像不存在: {test_image}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
            print(f"清理临时目录: {temp_dir}")
        except:
            pass

if __name__ == "__main__":
    success = test_chinese_path_support()
    sys.exit(0 if success else 1)
