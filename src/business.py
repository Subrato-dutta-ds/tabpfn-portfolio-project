import numpy as np
import pandas as pd


def rank_customers(df, score_col='probability'):
    return df.sort_values(score_col, ascending=False).reset_index(drop=True)


def expected_profit(probability, revenue, cost):
    return float(probability) * float(revenue) - float(cost)


def incremental_profit(uplift, revenue, cost):
    """Expected incremental profit from contacting a customer: uplift * R - C."""
    return float(uplift) * float(revenue) - float(cost)


def optimal_k(probabilities, revenue, cost):
    """K* maximizing cumulative expected profit (K=0 allowed)."""
    probs = np.asarray(probabilities, dtype=float)
    cumulative = probs.cumsum()
    k_values = np.arange(1, len(probs) + 1)
    profits = cumulative * revenue - k_values * cost
    profits_with_zero = np.concatenate([[0.0], profits])
    best_idx = int(np.argmax(profits_with_zero))
    return best_idx, float(profits_with_zero[best_idx])


def optimal_k_uplift(uplifts, revenue, cost):
    """K* maximizing cumulative INCREMENTAL profit using uplift scores."""
    u = np.asarray(uplifts, dtype=float)
    cumulative = u.cumsum()
    k_values = np.arange(1, len(u) + 1)
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
    ec = float(top.sum())
    er = float(top.sum() * revenue)
    cost_total = int(k) * float(cost)
    profit = er - cost_total
    roi = (profit / cost_total * 100) if cost_total > 0 else 0.0
    baseline = float(probs.mean())
    lift = (float(top.mean()) / baseline) if baseline > 0 else 0.0
    return {'contacts': int(k), 'expected_conversions': ec, 'expected_revenue': er,
            'expected_cost': cost_total, 'expected_profit': profit, 'roi_pct': roi, 'lift': lift}
