from fastapi.testclient import TestClient
from src.api import app
import numpy as np
from src.business import rank_customers, expected_profit, optimal_k, campaign_summary
import pandas as pd

client = TestClient(app)

valid_data = {
    'age': 30, 'job': 'admin.', 'marital': 'single', 'education': 'university.degree',
    'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'cellular', 'month': 'may',
    'day_of_week': 'mon', 'campaign': 1, 'pdays': 999, 'previous': 0, 'poutcome': 'nonexistent',
    'emp_var_rate': 1.1, 'cons_price_idx': 93.994, 'cons_conf_idx': -36.4, 'euribor3m': 4.857,
    'nr_employed': 5191.0
}

# -------- API tests --------

def test_health_endpoint():
    r = client.get("/api/v1/health")
    assert r.status_code == 200

def test_valid_prediction():
    r = client.post("/api/v1/predict", json=valid_data)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0

def test_missing_feature():
    invalid = {k: v for k, v in valid_data.items() if k != 'age'}
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_invalid_job_rejected():
    invalid = valid_data.copy()
    invalid['job'] = 'not-a-real-job'
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_invalid_marital_rejected():
    invalid = valid_data.copy()
    invalid['marital'] = 'banana'
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_age_out_of_range():
    invalid = valid_data.copy()
    invalid['age'] = 150
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_campaign_zero_rejected():
    invalid = valid_data.copy()
    invalid['campaign'] = 0
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_batch_prediction():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data, valid_data]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2

def test_batch_size_limit():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data] * 5001})
    assert r.status_code == 413

# -------- Business logic tests --------

def test_rank_customers_sorts_descending():
    df = pd.DataFrame({
        'customer_id': ['A', 'B', 'C'],
        'probability': [0.3, 0.9, 0.5]
    })
    ranked = rank_customers(df)
    assert ranked['customer_id'].tolist() == ['B', 'C', 'A']
    assert ranked['probability'].tolist() == [0.9, 0.5, 0.3]

def test_expected_profit_formula():
    assert expected_profit(0.5, 2000, 50) == 950.0
    assert expected_profit(0.0, 2000, 50) == -50.0
    assert expected_profit(0.025, 2000, 50) == 0.0

def test_optimal_k_picks_max_profit():
    # With high prob customers, optimal should be 2 (not 3) when 3rd has low prob
    probs = np.array([0.9, 0.8, 0.01])
    k, profit = optimal_k(probs, revenue=2000, cost=50)
    assert k == 2
    assert profit > 0

def test_optimal_k_with_zero_probabilities():
    probs = np.array([0.0, 0.0, 0.0])
    k, profit = optimal_k(probs, revenue=2000, cost=50)
    # No customers should be contacted (any K is negative)
    assert k == 1  # argmax of [-50, -100, -150] is index 0

def test_campaign_summary_metrics():
    probs = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    s = campaign_summary(probs, k=2, revenue=2000, cost=50)
    assert s['contacts'] == 2
    assert abs(s['expected_conversions'] - 1.6) < 1e-6
    assert abs(s['expected_profit'] - (1.6 * 2000 - 100)) < 1e-6
    assert s['lift'] > 1.0
