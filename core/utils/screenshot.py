import tkinter as tk
from datetime import datetime

import pyautogui

from ui.main_window import MainWindow


class DynamicScreenshot:
    def __init__(self, callback):
        self.root = tk.Tk()
        self.root.attributes('-alpha', 0.3)  # 设置窗口透明度[6,7,8](@ref)
        self.root.attributes('-fullscreen', True)
        self.callback = callback

        # 鼠标事件绑定标志位
        self.is_drawing = True
        screen_width = self.root.winfo_screenwidth()
        #print(screenWidth)
        screen_height = self.root.winfo_screenheight()
        self.canvas = tk.Canvas(self.root, bg='grey', width=screen_width, height=screen_height)
        self.canvas.pack()
        # 鼠标事件绑定
        self.canvas.bind('<Button-1>', self.on_start)  # 完整事件名称[2](@ref)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_end)

        self.start_pos = None
        self.end_pos = None
        self.root.mainloop()

    def on_start(self, event):
        if not self.is_drawing:
            print("start return")
            return
        print("start drawing")
        self.start_pos = (event.x, event.y)
        self.canvas.coords('rect', self.start_pos[0], self.start_pos[1],
                          self.start_pos[0], self.start_pos[1])

    def on_drag(self, event):
        if not self.is_drawing:
            return
        self.end_pos = (event.x, event.y)
        self.canvas.coords('rect', min(self.start_pos[0], self.end_pos[0]),
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"  # 动态生成文件名[11](@ref)
            screenshot = pyautogui.screenshot(region=region)
            screenshot.save(filename)
            print(f"截图已保存为：{filename}")
            MainWindow.screenshot_preview = filename
        except Exception as e:
            print(f"截图失败：{str(e)}")
        finally:
            self.is_drawing = False
            self.root.after(100, lambda: setattr(self, 'is_drawing', False))  # 延迟重置状态[5](@ref)