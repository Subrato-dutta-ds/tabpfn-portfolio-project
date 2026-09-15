import numpy as np
import pandas as pd


def rank_customers(df, probability_col='probability'):
    return df.sort_values(probability_col, ascending=False).reset_index(drop=True)


def expected_profit(probability, revenue, cost):
    return float(probability) * float(revenue) - float(cost)


def optimal_k(probabilities, revenue, cost):
    probs = np.asarray(probabilities, dtype=float)
    cumulative = probs.cumsum()
    k_values = np.arange(1, len(probs) + 1)
    profits = cumulative * revenue - k_values * cost
    profits_with_zero = np.concatenate([[0.0], profits])
    best_idx = int(np.argmax(profits_with_zero))
    return best_idx, float(profits_with_zero[best_idx])


def campaign_summary(probabilities, k, revenue, cost):
    probs = np.asarray(probabilities, dtype=float)
    if k <= 0:
        return {'contacts': 0, 'expected_conversions': 0.0, 'expected_revenue': 0.0,
                'expected_cost': 0.0, 'expected_profit': 0.0, 'roi_pct': 0.0, 'lift': 0.0}
    top = probs[:k]
    expected_conversions = float(top.sum())
    expected_revenue = float(top.sum() * revenue)
    expected_cost = int(k) * float(cost)
    profit = expected_revenue - expected_cost
    roi = (profit / expected_cost * 100) if expected_cost > 0 else 0.0
    baseline = float(probs.mean())
    lift = (float(top.mean()) / baseline) if baseline > 0 else 0.0
    return {'contacts': int(k), 'expected_conversions': expected_conversions,
            'expected_revenue': expected_revenue, 'expected_cost': expected_cost,
            'expected_profit': profit, 'roi_pct': roi, 'lift': lift}
