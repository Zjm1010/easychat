import tkinter as tk
from datetime import datetime
import os

import pyautogui

class DynamicScreenshot:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-alpha', 0.3)  # 设置窗口透明度
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)

        # 鼠标事件绑定标志位
        self.is_drawing = True
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Create main canvas with dark background
        self.canvas = tk.Canvas(self.root, bg='black', width=screen_width, height=screen_height)
        self.canvas.pack()
        
        # Create a transparent rectangle for the selection area
        self.selection_rect = self.canvas.create_rectangle(0, 0, 0, 0, fill='white', stipple='gray50')
        
        # 鼠标事件绑定
        self.canvas.bind('<Button-1>', self.on_start)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_end)
        self.start_pos = None
        self.end_pos = None
        
        # Fixed filename for screenshot
        self.screenshot_filename = "screenshot.png"

    def push_screenshot(self):
        self.root.mainloop()
        print("状态改变")

    def on_start(self, event):
        if not self.is_drawing:
            print("start return")
            return
        print("start drawing")
        self.start_pos = (event.x, event.y)
        self.canvas.coords(self.selection_rect, self.start_pos[0], self.start_pos[1],
                          self.start_pos[0], self.start_pos[1])

    def on_drag(self, event):
        if not self.is_drawing:
            return
        self.end_pos = (event.x, event.y)
        self.canvas.coords(self.selection_rect, min(self.start_pos[0], self.end_pos[0]),
                          min(self.start_pos[1], self.end_pos[1]),
                          max(self.start_pos[0], self.end_pos[0]),
                          max(self.start_pos[1], self.end_pos[1]))

    def on_end(self, event):
        if not self.is_drawing:
            return
        self.root.destroy()
        try:
            x1, y1 = min(self.start_pos[0], self.end_pos[0]), min(self.start_pos[1], self.end_pos[1])
            x2, y2 = max(self.start_pos[0], self.end_pos[0]), max(self.start_pos[1], self.end_pos[1])
            region = (x1, y1, x2 - x1, y2 - y1)
            
            # Delete old screenshot if it exists
            if os.path.exists(self.screenshot_filename):
                os.remove(self.screenshot_filename)
                print(f"已删除旧截图：{self.screenshot_filename}")
            
            # Save new screenshot
            screenshot = pyautogui.screenshot(region=region)
            screenshot.save(self.screenshot_filename)
            print(f"截图已保存为：{self.screenshot_filename}")
        except Exception as e:
            print(f"截图失败：{str(e)}")
        finally:
            self.is_drawing = False
            self.root.after(100, lambda: setattr(self, 'is_drawing', False))  # 延迟重置状态

    def screenshot_mask(self):
        self.root.attributes('-alpha', 0.3)  # 设置窗口透明度
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'grey')
        self.root.attributes('-transparent', True)

