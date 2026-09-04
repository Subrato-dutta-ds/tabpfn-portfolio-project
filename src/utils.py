import os, requests

API_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

def check_api_health():
    try:
        r = requests.get(f'{API_URL}/health', timeout=2)
        return r.status_code == 200
    except:
        return False
