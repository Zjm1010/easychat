from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from core.logic.chat_logic import DeepSeekLogic
from ui.widgets.config_window import ConfigWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.logic = DeepSeekLogic()

    def initUI(self):
        self.setWindowTitle('DeepSeek Client')
        self.setGeometry(100, 100, 400, 300)

        self.label = QLabel('Press shortcut to capture and ask', self)
        self.result_label = QLabel('Result will be shown here', self)
        self.button = QPushButton('Capture and Ask', self)
        self.button.clicked.connect(self.on_button_click)
        # 打开配置窗口的按钮
        self.config_button = QPushButton('Open Config', self)
        self.config_button.clicked.connect(self.open_config_window)
        self.setCentralWidget(self.config_button)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.button)
        layout.addWidget(self.config_button)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

##
    def on_button_click(self):
        response = self.logic.capture_and_ask()
        self.result_label.setText(response.get('answer', 'No answer found'))

    def open_config_window(self):
        """打开配置窗口"""
        self.config_window = ConfigWindow()
        self.config_window.show()