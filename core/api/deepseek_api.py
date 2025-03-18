import requests

def __init__(self, api_key, base_url):
    self.api_key = api_key
    self.base_url = base_url


def ask(self, image_path, deep_think=False, network_search=False):
    headers = {'Authorization': f'Bearer {self.api_key}'}
    data = {
        'deep_think': deep_think,
        'network_search': network_search
    }
    try:
        files = {'image': open(image_path, 'rb')}
        response = requests.post(self.base_url, headers=headers, files=files, data=data)
    except FileNotFoundError:
        response = requests.post(self.base_url, headers=headers, files=files, data=data)
    return response.json()