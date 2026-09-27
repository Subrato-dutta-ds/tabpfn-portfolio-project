from fastapi.testclient import TestClient
from src.api import app
import numpy as np
import pandas as pd
import math

client = TestClient(app)

valid_data = {
    'age': 30, 'job': 'admin.', 'marital': 'single', 'education': 'university.degree',
    'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'cellular', 'month': 'may',
    'day_of_week': 'mon', 'campaign': 1, 'pdays': 999, 'previous': 0, 'poutcome': 'nonexistent',
    'emp_var_rate': 1.1, 'cons_price_idx': 93.994, 'cons_conf_idx': -36.4, 'euribor3m': 4.857,
    'nr_employed': 5191.0
}

# -------- API --------

def test_health_endpoint():
    assert client.get("/api/v1/health").status_code == 200

def test_valid_prediction():
    r = client.post("/api/v1/predict", json=valid_data)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probability"] <= 1.0

def test_missing_feature():
    inv = {k: v for k, v in valid_data.items() if k != 'age'}
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_invalid_category():
    inv = valid_data.copy(); inv['job'] = 'not-a-real-job'
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_invalid_marital():
    inv = valid_data.copy(); inv['marital'] = 'banana'
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_invalid_age():
    inv = valid_data.copy(); inv['age'] = 150
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_campaign_zero():
    inv = valid_data.copy(); inv['campaign'] = 0
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_nan_age_rejected():
    inv = valid_data.copy(); inv['age'] = None
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_batch_prediction():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data, valid_data]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2

def test_batch_size_limit():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data] * 5001})
    assert r.status_code == 413

def test_empty_batch():
    r = client.post("/api/v1/predict-batch", json={"data": []})
    assert r.status_code == 200
    assert r.json()["total"] == 0

# -------- Business logic --------

def test_rank_customers_descending():
    df = pd.DataFrame({'customer_id': ['A', 'B', 'C'], 'probability': [0.3, 0.9, 0.5]})
    from src.business import rank_customers
    assert rank_customers(df)['customer_id'].tolist() == ['B', 'C', 'A']

def test_expected_profit_formula():
    from src.business import expected_profit
    assert expected_profit(0.5, 2000, 50) == 950.0
    assert expected_profit(0.0, 2000, 50) == -50.0

def test_optimal_k_positive():
    from src.business import optimal_k
    k, profit = optimal_k(np.array([0.9, 0.8, 0.01]), 2000, 50)
    assert k == 2 and profit > 0

def test_optimal_k_zero_when_unprofitable():
    from src.business import optimal_k
    k, profit = optimal_k(np.array([0.001, 0.001, 0.001]), 2000, 50)
    assert k == 0 and profit == 0.0

def test_optimal_k_uplift():
    from src.business import optimal_k_uplift
    k, profit = optimal_k_uplift(np.array([0.3, 0.2, 0.01]), 2000, 50)
    assert k == 2 and profit > 0

def test_campaign_summary_zero_contacts():
    from src.business import campaign_summary
    s = campaign_summary(np.array([0.9, 0.7, 0.5]), 0, 2000, 50)
    assert s['contacts'] == 0 and s['expected_profit'] == 0.0

