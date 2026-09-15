import numpy as np
import pandas as pd
from src.business import rank_customers, expected_profit, optimal_k, campaign_summary


def test_rank_customers_sorts_descending():
    df = pd.DataFrame({'customer_id': ['A', 'B', 'C'], 'probability': [0.3, 0.9, 0.5]})
    ranked = rank_customers(df)
    assert ranked['customer_id'].tolist() == ['B', 'C', 'A']
    assert ranked['probability'].tolist() == [0.9, 0.5, 0.3]


def test_expected_profit_formula():
    assert expected_profit(0.5, 2000, 50) == 950.0
    assert expected_profit(0.0, 2000, 50) == -50.0
    assert expected_profit(0.025, 2000, 50) == 0.0


def test_optimal_k_positive_case():
    probs = np.array([0.9, 0.8, 0.01])
    k, profit = optimal_k(probs, revenue=2000, cost=50)
    assert k == 2
    assert profit > 0


def test_optimal_k_returns_zero_when_unprofitable():
    probs = np.array([0.001, 0.001, 0.001])
    k, profit = optimal_k(probs, revenue=2000, cost=50)
    assert k == 0
    assert profit == 0.0


def test_optimal_k_at_break_even():
    # p = cost/revenue => profit = 0; K=0 also gives 0 (tie -> argmax picks first)
    probs = np.array([0.025, 0.025])
    k, profit = optimal_k(probs, revenue=2000, cost=50)
    assert k == 0
    assert profit == 0.0


def test_campaign_summary_k_zero():
    s = campaign_summary(np.array([0.9, 0.7, 0.5]), k=0, revenue=2000, cost=50)
    assert s['contacts'] == 0
    assert s['expected_profit'] == 0.0
    assert s['roi_pct'] == 0.0
    assert s['lift'] == 0.0


def test_campaign_summary_metrics():
    probs = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    s = campaign_summary(probs, k=2, revenue=2000, cost=50)
    assert s['contacts'] == 2
    assert abs(s['expected_conversions'] - 1.6) < 1e-6
    assert abs(s['expected_profit'] - (1.6 * 2000 - 100)) < 1e-6
    assert s['lift'] > 1.0
