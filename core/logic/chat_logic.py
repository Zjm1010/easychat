from core.api.deepseek_api import DeepSeekAPI
from core.utils.screenshot import capture_screenshot
from core.utils.config_manager import ConfigManager

class DeepSeekLogic:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.api = DeepSeekAPI(self.config['api_key'], self.config['selected_api'])

    def capture_and_ask(self):
        image_path = capture_screenshot()
        response = self.api.ask(
            image_path,
            deep_think=self.config['enable_deep_think'],
            network_search=self.config['enable_network_search']
        )
        return response