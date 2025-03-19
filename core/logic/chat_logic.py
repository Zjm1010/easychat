from core.api.deepseek_api import DeepSeekAPI
from core.utils.screenshot import screenshot
from core.utils.config_manager import ConfigManager

class DeepSeekLogic:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.api = DeepSeekAPI(self.config['api_key'], self.config['base_url'])
        self.capture = None

    def capture_and_ask(self):
        response = self.api.ask(
            self.capture,
            deep_think=self.config['enable_deep_think'],
            network_search=self.config['enable_network_search']
        )
        return response

    def set_capture(self, capture):
        self.capture = capture