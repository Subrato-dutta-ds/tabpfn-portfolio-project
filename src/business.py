import numpy as np
import pandas as pd


def rank_customers(df, probability_col='probability'):
    """Sort customers by probability descending."""
    return df.sort_values(probability_col, ascending=False).reset_index(drop=True)


def expected_profit(probability, revenue, cost):
    """Expected profit for a single customer: p*R - C."""
    return float(probability) * float(revenue) - float(cost)


def optimal_k(probabilities, revenue, cost):
    """Find K* that maximizes cumulative expected profit."""
    probs = np.asarray(probabilities, dtype=float)
    cumulative = probs.cumsum()
    k_values = np.arange(1, len(probs) + 1)
    profits = cumulative * revenue - k_values * cost
    best_idx = int(np.argmax(profits))
    return best_idx + 1, float(profits[best_idx])


def campaign_summary(probabilities, k, revenue, cost):
    """Return dictionary of business metrics for top-K campaign."""
    probs = np.asarray(probabilities, dtype=float)
    top = probs[:k]
    expected_conversions = float(top.sum())
    expected_revenue = float(top.sum() * revenue)
    expected_cost = int(k) * float(cost)
    profit = expected_revenue - expected_cost
    roi = (profit / expected_cost * 100) if expected_cost > 0 else 0.0
    baseline = float(probs.mean())
    lift = (float(top.mean()) / baseline) if baseline > 0 else 0.0
    return {
        'contacts': int(k),
        'expected_conversions': expected_conversions,
        'expected_revenue': expected_revenue,
        'expected_cost': expected_cost,
        'expected_profit': profit,
        'roi_pct': roi,
        'lift': lift,
    }
