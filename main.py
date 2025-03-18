import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def load_stylesheet():
    with open('ui/styles/main_style.qss', 'r') as f:
        return f.read()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet())  # 加载样式表
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())