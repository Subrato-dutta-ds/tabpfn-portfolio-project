import numpy as np
from src.business import (rank_customers, expected_profit, incremental_profit,
                          optimal_k, optimal_k_uplift, campaign_summary)


def test_rank_customers_descending():
    import pandas as pd
    df = pd.DataFrame({'customer_id': ['A', 'B', 'C'], 'probability': [0.3, 0.9, 0.5]})
    assert rank_customers(df)['customer_id'].tolist() == ['B', 'C', 'A']


def test_expected_profit_formula():
    assert expected_profit(0.5, 2000, 50) == 950.0
    assert expected_profit(0.0, 2000, 50) == -50.0
    assert abs(expected_profit(0.025, 2000, 50)) < 1e-9


def test_incremental_profit_formula():
    assert incremental_profit(0.3, 2000, 50) == 550.0
    assert incremental_profit(0.0, 2000, 50) == -50.0


def test_optimal_k_positive_case():
    k, profit = optimal_k(np.array([0.9, 0.8, 0.01]), 2000, 50)
    assert k == 2 and profit > 0


def test_optimal_k_returns_zero_when_unprofitable():
    k, profit = optimal_k(np.array([0.001, 0.001, 0.001]), 2000, 50)
    assert k == 0 and profit == 0.0


def test_optimal_k_uplift_positive():
    k, profit = optimal_k_uplift(np.array([0.4, 0.3, 0.01]), 2000, 50)
    assert k == 2 and profit > 0


def test_optimal_k_uplift_zero():
    k, profit = optimal_k_uplift(np.array([0.001, 0.001]), 2000, 50)
    assert k == 0 and profit == 0.0


def test_campaign_summary_k_zero():
    s = campaign_summary(np.array([0.9, 0.7, 0.5]), 0, 2000, 50)
    assert s['contacts'] == 0 and s['expected_profit'] == 0.0


def test_campaign_summary_metrics():
    s = campaign_summary(np.array([0.9, 0.7, 0.5, 0.3, 0.1]), 2, 2000, 50)
    assert s['contacts'] == 2
    assert abs(s['expected_conversions'] - 1.6) < 1e-6
    assert abs(s['expected_profit'] - (1.6 * 2000 - 100)) < 1e-6




# --- Promotion gate tests (using PROMO metrics) ---

def test_promotion_accepts_improvement():
    from src.promote_model import gate
    cand = {'promo_pr_auc': 0.50, 'promo_brier': 0.07, 'promo_f1': 0.55,
            'auuc': 250.0, 'uplift_at_20': 0.15}
    prod = {'promo_pr_auc': 0.48, 'promo_brier': 0.075, 'promo_f1': 0.52,
            'auuc': 200.0, 'uplift_at_20': 0.13}
    promote, _ = gate(cand, prod)
    assert promote is True


def test_promotion_rejects_bad_auuc():
    from src.promote_model import gate
    cand = {'promo_pr_auc': 0.50, 'promo_brier': 0.07, 'promo_f1': 0.55,
            'auuc': 50.0, 'uplift_at_20': 0.10}
    prod = {'promo_pr_auc': 0.48, 'promo_brier': 0.075, 'promo_f1': 0.52,
            'auuc': 200.0, 'uplift_at_20': 0.13}
    promote, _ = gate(cand, prod)
    assert promote is False


def test_promotion_rejects_bad_uplift():
    from src.promote_model import gate
    cand = {'promo_pr_auc': 0.50, 'promo_brier': 0.07, 'promo_f1': 0.55,
            'auuc': 250.0, 'uplift_at_20': 0.05}
    prod = {'promo_pr_auc': 0.48, 'promo_brier': 0.075, 'promo_f1': 0.52,
            'auuc': 200.0, 'uplift_at_20': 0.13}
    promote, _ = gate(cand, prod)
    assert promote is False


def test_promotion_rejects_bad_pr_auc():
    from src.promote_model import gate
    cand = {'promo_pr_auc': 0.40, 'promo_brier': 0.07, 'promo_f1': 0.55,
            'auuc': 250.0, 'uplift_at_20': 0.15}
    prod = {'promo_pr_auc': 0.48, 'promo_brier': 0.075, 'promo_f1': 0.52,
            'auuc': 200.0, 'uplift_at_20': 0.13}
    promote, _ = gate(cand, prod)
    assert promote is False