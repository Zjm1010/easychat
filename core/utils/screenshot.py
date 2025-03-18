from pillow import ImageGrab

from core.utils.config_manager import ConfigManager


class screenshot:
    def __init__(self):
        self.config_manager = ConfigManager()

    def capture_screenshot(save_path='screenshot.png'):
        screenshot = ImageGrab.grab()
        screenshot.save(save_path)
        return save_path