#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试视频帧保存功能
"""

import cv2
import numpy as np
import os

def test_save_functionality():
    """测试保存功能"""
    print("测试视频帧保存功能...")
    
    # 创建一个测试视频
    test_video_path = "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
    
    # 生成几帧测试图像
    for i in range(10):
        # 创建彩色测试图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 25) % 255  # 蓝色通道
        frame[:, :, 1] = ((i + 10) * 25) % 255  # 绿色通道
        frame[:, :, 2] = ((i + 20) * 25) % 255  # 红色通道
        
        # 添加帧号文字
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"测试视频已创建: {test_video_path}")
    
    # 测试读取和保存
    cap = cv2.VideoCapture(test_video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"成功读取帧，形状: {frame.shape}")
            
            # 测试保存
            save_path = "test_frame.png"
            success = cv2.imwrite(save_path, frame)
            print(f"保存结果: {success}")
            
            if success and os.path.exists(save_path):
                print(f"帧已成功保存到: {save_path}")
                # 清理测试文件
                os.remove(save_path)
            else:
                print("保存失败")
        else:
            print("无法读取帧")
    else:
        print("无法打开视频")
    
    cap.release()
    
    # 清理测试视频
    if os.path.exists(test_video_path):
        os.remove(test_video_path)
        print("测试视频已清理")

if __name__ == "__main__":
    test_save_functionality()
