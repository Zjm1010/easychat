from core.api.deepseek_api import DeepSeekAPI
from core.utils.config_manager import ConfigManager
import os


class DeepSeekLogic:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.api = DeepSeekAPI(self.config['api_key'], self.config['base_url'])
        self.capture = None

    def capture_and_ask(self, message):
        if self.capture is not None:
            response = self.api.ask(
                self.capture,
                message
            )
        else:
            response = self.api.ask(
                None,
                message
            )
        return response

    def set_capture(self, capture):
        self.capture = capture

    def remove_capture(self):
        if self.capture and os.path.exists("screenshot.png"):
            try:
                os.remove(self.capture)
                print("已删除旧图")
            except Exception as e:
                print(f"Error removing screenshot file: {str(e)}")
        self.capture = None
        
