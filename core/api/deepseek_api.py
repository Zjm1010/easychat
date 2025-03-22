import requests
from openai import OpenAI


class DeepSeekAPI:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def ask(self, image_path, message, deep_think=False, network_search=False):
        try:
            if image_path is None:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": message}
                    ],
                    stream=False
                )
            else:
                files = {'image': open(image_path, 'rb')}
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": message, "file": files}
                    ],
                    stream=False
                )
        except Exception as e:
            return {"answer": "服务异常"}
        return response.json()
