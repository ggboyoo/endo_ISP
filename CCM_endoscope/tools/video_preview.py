#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频预览程序
功能：
1. 读取MP4文件并预览
2. 实时显示帧号
3. 提供交互界面，可以保存当前帧
4. 支持播放控制（播放/暂停、进度条、帧跳转）
"""

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
from PIL import Image, ImageTk
import threading
import time


class VideoPreview:
    def __init__(self, root):
        self.root = root
        self.root.title("视频预览器")
        self.root.geometry("1000x700")
        
        # 视频相关变量
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame = 0
        self.is_playing = False
        self.current_image = None
        
        # 保存相关变量
        self.last_save_dir = os.getcwd()  # 默认保存目录
        self.save_counter = 0  # 保存计数器
        self.config_file = "video_preview_config.json"  # 配置文件
        
        # 加载配置
        self.load_config()
        
        # 创建界面
        self.create_widgets()
        
        # 播放线程
        self.play_thread = None
        self.stop_playing = False
    
    def load_config(self):
        """加载配置文件"""
        try:
            import json
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.last_save_dir = config.get('last_save_dir', os.getcwd())
                    self.save_counter = config.get('save_counter', 0)
                print(f"加载配置: 保存目录={self.last_save_dir}, 计数器={self.save_counter}")
        except Exception as e:
            print(f"加载配置失败: {e}")
    
    def save_config(self):
        """保存配置文件"""
        try:
            import json
            config = {
                'last_save_dir': self.last_save_dir,
                'save_counter': self.save_counter
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"保存配置: 保存目录={self.last_save_dir}, 计数器={self.save_counter}")
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def get_next_filename(self):
        """获取下一个文件名"""
        # 查找下一个可用的文件名
        while True:
            filename = f"{self.save_counter}.png"
            file_path = os.path.join(self.last_save_dir, filename)
            if not os.path.exists(file_path):
                break
            self.save_counter += 1
        
        return filename, file_path
        
    def create_widgets(self):
        """创建GUI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件选择
        ttk.Button(control_frame, text="选择视频文件", command=self.select_video).pack(side=tk.LEFT, padx=(0, 10))
        
        # 播放控制
        self.play_button = ttk.Button(control_frame, text="播放", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(control_frame, text="停止", command=self.stop_video).pack(side=tk.LEFT, padx=(0, 5))
        
        # 帧控制
        ttk.Label(control_frame, text="帧:").pack(side=tk.LEFT, padx=(10, 5))
        self.frame_var = tk.StringVar(value="0/0")
        frame_label = ttk.Label(control_frame, textvariable=self.frame_var)
        frame_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # 保存按钮
        ttk.Button(control_frame, text="保存当前帧", command=self.save_current_frame).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(control_frame, text="快速保存", command=self.quick_save_frame).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(control_frame, text="重置计数", command=self.reset_counter).pack(side=tk.RIGHT)
        
        # 进度条
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(progress_frame, text="进度:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        # 进度条绑定事件
        self.progress_bar.bind("<Button-1>", self.on_progress_click)
        
        # 帧跳转控制
        frame_control = ttk.Frame(main_frame)
        frame_control.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(frame_control, text="上一帧", command=self.prev_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(frame_control, text="下一帧", command=self.next_frame).pack(side=tk.LEFT, padx=(0, 5))
        
        # 帧号输入
        ttk.Label(frame_control, text="跳转到帧:").pack(side=tk.LEFT, padx=(20, 5))
        self.frame_entry = ttk.Entry(frame_control, width=10)
        self.frame_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(frame_control, text="跳转", command=self.jump_to_frame).pack(side=tk.LEFT)
        
        # 视频显示区域
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        # 视频显示标签
        self.video_label = ttk.Label(video_frame, text="请选择视频文件", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # 视频信息显示
        self.info_var = tk.StringVar()
        info_label = ttk.Label(status_frame, textvariable=self.info_var)
        info_label.pack(side=tk.RIGHT)
        
        # 保存信息显示
        self.save_info_var = tk.StringVar(value=f"保存目录: {os.path.basename(self.last_save_dir)} | 下次保存: {self.save_counter}.png")
        save_info_label = ttk.Label(status_frame, textvariable=self.save_info_var)
        save_info_label.pack(side=tk.RIGHT, padx=(0, 20))
        
    def select_video(self):
        """选择视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, video_path):
        """加载视频文件"""
        try:
            # 关闭之前的视频
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开视频文件")
                return
            
            self.video_path = video_path
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.current_frame = 0
            
            # 更新界面
            self.frame_var.set(f"0/{self.total_frames}")
            self.progress_var.set(0)
            self.info_var.set(f"文件: {os.path.basename(video_path)} | 总帧数: {self.total_frames} | FPS: {self.fps:.2f}")
            self.status_var.set("视频已加载")
            
            # 显示第一帧
            self.show_frame(0)
            
        except Exception as e:
            messagebox.showerror("错误", f"加载视频时出错: {str(e)}")
    
    def show_frame(self, frame_number):
        """显示指定帧"""
        if not self.cap:
            return
        
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                self.current_frame = frame_number
                # 确保保存原始帧数据用于保存
                self.current_image = frame.copy()
                
                # 转换颜色空间 (BGR -> RGB) 用于显示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 调整图像大小以适应显示区域
                display_frame = self.resize_frame(frame_rgb)
                
                # 转换为PIL图像
                pil_image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                # 更新显示
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo  # 保持引用
                
                # 更新进度和帧号
                progress = (frame_number / self.total_frames) * 100 if self.total_frames > 0 else 0
                self.progress_var.set(progress)
                self.frame_var.set(f"{frame_number}/{self.total_frames}")
                
                # 调试信息
                print(f"显示帧 {frame_number}, 图像尺寸: {self.current_image.shape}")
                
            else:
                print(f"无法读取帧 {frame_number}")
                
        except Exception as e:
            print(f"显示帧时出错: {str(e)}")
    
    def resize_frame(self, frame, max_width=800, max_height=600):
        """调整帧大小以适应显示区域"""
        height, width = frame.shape[:2]
        
        # 计算缩放比例
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # 不放大图像
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(frame, (new_width, new_height))
    
    def toggle_play(self):
        """切换播放/暂停状态"""
        if not self.cap:
            return
        
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()
    
    def play_video(self):
        """开始播放视频"""
        if not self.cap or self.current_frame >= self.total_frames - 1:
            return
        
        self.is_playing = True
        self.stop_playing = False
        self.play_button.configure(text="暂停")
        self.status_var.set("正在播放...")
        
        # 启动播放线程
        self.play_thread = threading.Thread(target=self._play_loop)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def pause_video(self):
        """暂停视频"""
        self.is_playing = False
        self.stop_playing = True
        self.play_button.configure(text="播放")
        self.status_var.set("已暂停")
    
    def stop_video(self):
        """停止视频"""
        self.pause_video()
        self.current_frame = 0
        self.show_frame(0)
        self.status_var.set("已停止")
    
    def _play_loop(self):
        """播放循环（在单独线程中运行）"""
        while self.is_playing and not self.stop_playing and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.show_frame(self.current_frame)
            
            # 控制播放速度
            if self.fps > 0:
                time.sleep(1.0 / self.fps)
            else:
                time.sleep(0.033)  # 默认30fps
        
        # 播放结束
        if self.current_frame >= self.total_frames - 1:
            self.is_playing = False
            self.play_button.configure(text="播放")
            self.status_var.set("播放完成")
    
    def prev_frame(self):
        """上一帧"""
        if self.current_frame > 0:
            self.pause_video()
            self.show_frame(self.current_frame - 1)
    
    def next_frame(self):
        """下一帧"""
        if self.current_frame < self.total_frames - 1:
            self.pause_video()
            self.show_frame(self.current_frame + 1)
    
    def jump_to_frame(self):
        """跳转到指定帧"""
        try:
            frame_num = int(self.frame_entry.get())
            if 0 <= frame_num < self.total_frames:
                self.pause_video()
                self.show_frame(frame_num)
            else:
                messagebox.showwarning("警告", f"帧号必须在 0 到 {self.total_frames - 1} 之间")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的帧号")
    
    def on_progress_click(self, event):
        """进度条点击事件"""
        if not self.cap or self.total_frames == 0:
            return
        
        # 计算点击位置对应的帧号
        progress_bar_width = self.progress_bar.winfo_width()
        click_x = event.x
        progress_ratio = click_x / progress_bar_width
        target_frame = int(progress_ratio * self.total_frames)
        
        self.pause_video()
        self.show_frame(target_frame)
    
    def save_current_frame(self):
        """保存当前帧"""
        if not self.cap:
            messagebox.showwarning("警告", "请先加载视频文件")
            return
        
        if self.current_image is None:
            messagebox.showwarning("警告", "没有可保存的帧")
            return
        
        # 获取下一个文件名
        filename, default_path = self.get_next_filename()
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="保存当前帧",
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("所有文件", "*.*")],
            initialdir=self.last_save_dir,  # 使用上次保存的目录
            initialfile=filename  # 使用自动生成的文件名
        )
        
        if file_path:
            try:
                # 调试信息
                print(f"尝试保存帧 {self.current_frame}")
                print(f"图像类型: {type(self.current_image)}")
                print(f"图像形状: {self.current_image.shape if self.current_image is not None else 'None'}")
                print(f"保存路径: {file_path}")
                
                # 确保图像数据有效
                if self.current_image is None or self.current_image.size == 0:
                    messagebox.showerror("错误", "当前帧数据无效")
                    return
                
                # 确保目录存在
                save_dir = os.path.dirname(file_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"创建目录: {save_dir}")
                
                # 尝试不同的保存方法
                success = False
                
                # 方法1: 直接使用cv2.imwrite
                success = cv2.imwrite(file_path, self.current_image)
                print(f"cv2.imwrite结果: {success}")
                
                if not success:
                    # 方法2: 使用PIL保存
                    try:
                        from PIL import Image
                        # 转换BGR到RGB
                        image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(image_rgb)
                        pil_image.save(file_path)
                        success = True
                        print("使用PIL保存成功")
                    except Exception as pil_error:
                        print(f"PIL保存失败: {pil_error}")
                
                if not success:
                    # 方法3: 尝试保存到临时位置再移动
                    try:
                        import tempfile
                        temp_dir = tempfile.gettempdir()
                        temp_path = os.path.join(temp_dir, f"temp_frame_{self.current_frame}.png")
                        success = cv2.imwrite(temp_path, self.current_image)
                        if success:
                            import shutil
                            shutil.move(temp_path, file_path)
                            print("通过临时文件保存成功")
                    except Exception as temp_error:
                        print(f"临时文件保存失败: {temp_error}")
                
                if success:
                    # 更新保存目录和计数器
                    self.last_save_dir = os.path.dirname(file_path)
                    self.save_counter += 1
                    self.save_config()  # 保存配置
                    self.update_save_info()  # 更新保存信息显示
                    
                    messagebox.showinfo("成功", f"帧已保存到: {file_path}")
                    self.status_var.set(f"已保存帧 {self.current_frame} 到 {os.path.basename(file_path)}")
                else:
                    messagebox.showerror("错误", f"保存失败，请检查文件路径和权限。\n路径: {file_path}")
                    
            except Exception as e:
                messagebox.showerror("错误", f"保存帧时出错: {str(e)}")
                print(f"保存错误详情: {str(e)}")  # 调试信息
    
    def quick_save_frame(self):
        """快速保存当前帧到程序目录"""
        if not self.cap:
            messagebox.showwarning("警告", "请先加载视频文件")
            return
        
        if self.current_image is None:
            messagebox.showwarning("警告", "没有可保存的帧")
            return
        
        try:
            # 使用自动命名
            filename, file_path = self.get_next_filename()
            
            print(f"快速保存到: {file_path}")
            
            # 确保图像数据有效
            if self.current_image is None or self.current_image.size == 0:
                messagebox.showerror("错误", "当前帧数据无效")
                return
            
            # 尝试保存
            success = cv2.imwrite(file_path, self.current_image)
            print(f"快速保存结果: {success}")
            
            if success:
                # 更新保存目录和计数器
                self.last_save_dir = os.path.dirname(file_path)
                self.save_counter += 1
                self.save_config()  # 保存配置
                self.update_save_info()  # 更新保存信息显示
                
                messagebox.showinfo("成功", f"帧已保存到: {filename}")
                self.status_var.set(f"已快速保存帧 {self.current_frame} 到 {filename}")
            else:
                # 尝试PIL保存
                try:
                    from PIL import Image
                    image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                    pil_image.save(file_path)
                    
                    # 更新保存目录和计数器
                    self.last_save_dir = os.path.dirname(file_path)
                    self.save_counter += 1
                    self.save_config()  # 保存配置
                    self.update_save_info()  # 更新保存信息显示
                    
                    messagebox.showinfo("成功", f"帧已保存到: {filename} (使用PIL)")
                    self.status_var.set(f"已快速保存帧 {self.current_frame} 到 {filename}")
                except Exception as pil_error:
                    messagebox.showerror("错误", f"快速保存失败: {str(pil_error)}")
                    
        except Exception as e:
            messagebox.showerror("错误", f"快速保存时出错: {str(e)}")
            print(f"快速保存错误详情: {str(e)}")
    
    def reset_counter(self):
        """重置保存计数器"""
        self.save_counter = 0
        self.save_config()
        self.update_save_info()  # 更新保存信息显示
        messagebox.showinfo("成功", "保存计数器已重置为0")
        self.status_var.set("保存计数器已重置")
    
    def update_save_info(self):
        """更新保存信息显示"""
        try:
            next_filename, _ = self.get_next_filename()
            self.save_info_var.set(f"保存目录: {os.path.basename(self.last_save_dir)} | 下次保存: {next_filename}")
        except:
            self.save_info_var.set(f"保存目录: {os.path.basename(self.last_save_dir)} | 计数器: {self.save_counter}")
    
    def __del__(self):
        """析构函数，释放资源"""
        if self.cap:
            self.cap.release()


def main():
    """主函数"""
    root = tk.Tk()
    app = VideoPreview(root)
    
    try:
        root.mainloop()
    finally:
        # 确保释放资源
        if app.cap:
            app.cap.release()


if __name__ == "__main__":
    main()
