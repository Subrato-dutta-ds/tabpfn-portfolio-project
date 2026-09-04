from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_prediction():
    data = {
        'age': 30, 'job': 'admin.', 'marital': 'single', 'education': 'university.degree',
        'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'cellular', 'month': 'may',
        'day_of_week': 'mon', 'campaign': 1, 'pdays': 999, 'previous': 0, 'poutcome': 'nonexistent',
        'emp_var_rate': 1.1, 'cons_price_idx': 93.994, 'cons_conf_idx': -36.4, 'euribor3m': 4.857,
        'nr_employed': 5191.0
    }
    # This will now actually FAIL if the API is broken
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    print("Test Passed!")
