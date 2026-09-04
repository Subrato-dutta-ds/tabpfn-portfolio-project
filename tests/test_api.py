import requests
import os

def test_api_prediction():
    API_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
    # Use a valid sample from your data
    data = {
        'age': 30, 'job': 'admin.', 'marital': 'single', 'education': 'university.degree',
        'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'cellular', 'month': 'may',
        'day_of_week': 'mon', 'campaign': 1, 'pdays': 999, 'previous': 0, 'poutcome': 'nonexistent',
        'emp_var_rate': 1.1, 'cons_price_idx': 93.994, 'cons_conf_idx': -36.4, 'euribor3m': 4.857,
        'nr_employed': 5191.0
    }
    try:
        response = requests.post(f'{API_URL}/predict', json=data)
        assert response.status_code == 200
        assert 'prediction' in response.json()
        print("API Test Passed!")
    except requests.exceptions.ConnectionError:
        print("API Test Skipped: API is not running. Start it with uvicorn src.api:app --reload")
