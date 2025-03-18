import json

class ConfigManager:
    def __init__(self, config_path='config/config.json'):
        self.config_path = config_path

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_config(self, config):
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)