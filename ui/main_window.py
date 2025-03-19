import os
from idlelib.help import HelpWindow


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QAction, QTextEdit, QHBoxLayout, \
    QMessageBox, QGroupBox, QSizePolicy

from core.logic.chat_logic import DeepSeekLogic
from core.utils.screenshot import DynamicScreenshot
from ui.widgets.config_window import ConfigWindow


class MainWindow(QMainWindow):
    screenshot_preview = None

    def __init__(self, qss_path='ui/styles/main_style.qss'):
        super().__init__()
        self.qss_path = qss_path
        self.take_screenshot = None
        self.initUI()
        self.logic = DeepSeekLogic()
        self.screenshot_button.clicked.connect(self.start_shortcut)
        self.screenshot_preview.mousePressEvent = self.on_preview_click

    def initUI(self):
        self.setWindowTitle("EasyChat")
        self.setGeometry(100, 100, 800, 600)

        # 创建主窗口布局
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        #创建菜单
        menubar = self.menuBar()
        settings_action = QAction("设置", self)
        help_action = QAction("帮助", self)
        menubar.addAction(settings_action)
        menubar.addAction(help_action)
        settings_action.triggered.connect(self.open_config_window)

        # 顶部区域：对话返回框 + 截图回显框
        top_layout = QHBoxLayout()
        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)
        top_layout.addWidget(self.chat_history)
        main_layout.addLayout(top_layout, 3)  # 占1份空间

        # 独立截图回显区域
        screenshot_widget = QGroupBox("预览", self)
        screenshot_layout = QVBoxLayout()
        self.screenshot_preview = QLabel(self)
        self.screenshot_preview.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.screenshot_preview.setFixedSize(40, 40)  # 统一尺寸
        screenshot_layout.addWidget(self.screenshot_preview)
        screenshot_widget.setLayout(screenshot_layout)
        screenshot_widget.setFixedSize(200,100)
        main_layout.addWidget(screenshot_widget, stretch=0)

        # 中间区域：对话输入框 + 发送/截图按钮组
        input_layout = QHBoxLayout()
        self.input_text_edit = QTextEdit(self)
        input_layout.addWidget(self.input_text_edit)

        button_group = QHBoxLayout()  # 新增按钮组布局
        self.send_button = QPushButton("发送", self)
        self.send_button.clicked.connect(self.on_button_click)
        button_group.addWidget(self.send_button)

        self.screenshot_button = QPushButton("划屏截图", self)
        self.screenshot_button.clicked.connect(self.start_shortcut)
        button_group.addWidget(self.screenshot_button)
        self.input_text_edit.setFixedHeight(100)  # 固定输入框高度[3](@ref)

        main_layout.addLayout(input_layout, 3)  # 占3份空间，确保输入框区域更大
        input_layout.addLayout(button_group)  # 将按钮组添加到输入框下方

        # 响应式布局优化（引用[1](@ref)）
        main_layout.setStretch(0, 1)    # 顶部区域可伸缩
        main_layout.setStretch(1, 4)    # 输入区域占主导

        # 初始化标签（修复未定义错误）
        self.result_label = QLabel("", self)
        main_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
        self.apply_stylesheet()

    ##发送截图(若有)/问题
    def on_button_click(self):
        response = self.logic.capture_and_ask()
        self.result_label.setText(response.get('answer', 'No answer found'))  # 修复未定义标签[6](@ref)

    ##打开配置窗口
    def open_config_window(self):
        self.config_window = ConfigWindow()
        self.config_window.show()

    def open_help_window(self):
        self.help_window = HelpWindow()
        self.config_window.show()

    ##开始划屏截图
    def start_shortcut(self):
        self.screenshot_window = DynamicScreenshot(self.on_screenshot_done)

    def apply_stylesheet(self):
        """读取并应用 QSS 样式表"""
        if os.path.exists(self.qss_path):
            try:
                with open(self.qss_path, 'r', encoding='utf-8') as f:
                    stylesheet = f.read()
                self.setStyleSheet(stylesheet)
            except Exception as e:
                QMessageBox.warning(self, '样式加载错误', f'无法加载 QSS 文件: {str(e)}')
        else:
            QMessageBox.warning(self, '样式文件缺失', f'未找到 QSS 文件: {self.qss_path}')

    def on_screenshot_done(self, filename):
        pixmap = QPixmap(filename)
        self.screenshot_preview.setPixmap(pixmap.scaled(40, 40))  # 更新预览图[5](@ref)
        self.result_label.setText(f"截图已保存：{filename}")  # 显示保存信息

    def on_preview_click(self, event):
        if self.screenshot_window:
            self.screenshot_window.root.deiconify()  # 点击预览图重新显示截图窗口[5](@ref)