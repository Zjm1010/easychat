import sys
import keyboard
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.widgets.config_window import ConfigWindow

def load_stylesheet():
    with open('ui/styles/main_style.qss', 'r', encoding='utf-8') as f:
        return f.read()

def show_window():
    window.show()
    window.activateWindow()  # 激活窗口


def open_config_window(self):
    """打开配置窗口"""
    self.config_window = ConfigWindow()
    self.config_window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet())  # 加载样式表
    window = MainWindow()
    keyboard.add_hotkey('ctrl+shift+s', show_window)

    window.show()
    sys.exit(app.exec_())