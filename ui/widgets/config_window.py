import json
import os

from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QMessageBox, QShortcut, QKeySequenceEdit
)

class ConfigWindow(QWidget):
    def __init__(self, config_path='config/config.json'):
        super().__init__()
        self.config_path = config_path
        self.config = self.load_config()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Configuration')
        self.setGeometry(300, 300, 400, 300)

        # 布局
        layout = QVBoxLayout()

        # API Key 输入
        self.api_key_label = QLabel('API Key:', self)
        self.api_key_input = QLineEdit(self)
        self.api_key_input.setText(self.config.get('api_key', ''))
        layout.addWidget(self.api_key_label)
        layout.addWidget(self.api_key_input)

        # 快捷键设置
        self.shortcut_label = QLabel('Shortcut:', self)
        self.shortcut_input = QKeySequenceEdit(QKeySequence(self.config.get('shortcut', 'Ctrl+Shift+S')), self)
        layout.addWidget(self.shortcut_label)
        layout.addWidget(self.shortcut_input)

        # 深度思考选项
        self.deep_think_checkbox = QCheckBox('Enable Deep Thinking', self)
        self.deep_think_checkbox.setChecked(self.config.get('deep_think', False))
        layout.addWidget(self.deep_think_checkbox)

        # 联网搜索选项
        self.network_search_checkbox = QCheckBox('Enable Network Search', self)
        self.network_search_checkbox.setChecked(self.config.get('network_search', False))
        layout.addWidget(self.network_search_checkbox)

        # 保存按钮
        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_config)
        layout.addWidget(self.save_button)

        # 设置布局
        self.setLayout(layout)

    def load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            return {}
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to load config: {str(e)}')
            return {}

    def save_config(self):
        """保存配置文件"""
        config = {
            'api_key': self.api_key_input.text(),
            'shortcut': self.shortcut_input.keySequence().toString(),
            'deep_think': self.deep_think_checkbox.isChecked(),
            'network_search': self.network_search_checkbox.isChecked()
        }
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            QMessageBox.information(self, 'Success', 'Configuration saved successfully!')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to save config: {str(e)}')

    def get_config(self):
        """获取当前配置"""
        return {
            'api_key': self.api_key_input.text(),
            'shortcut': self.shortcut_input.keySequence().toString(),
            'deep_think': self.deep_think_checkbox.isChecked(),
            'network_search': self.network_search_checkbox.isChecked()
        }