import os, json, numpy as np, pandas as pd
from src.config import BASE_DIR
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

REPORTS = os.path.join(BASE_DIR, 'reports')
LOGS = os.path.join(BASE_DIR, 'logs', 'predictions.jsonl')
OUTCOMES = os.path.join(BASE_DIR, 'logs', 'outcomes.csv')


def load_predictions():
    if not os.path.exists(LOGS):
        return pd.DataFrame()
    rows = []
    with open(LOGS) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def load_outcomes():
    if not os.path.exists(OUTCOMES):
        return pd.DataFrame()
    return pd.read_csv(OUTCOMES)


def compute_delayed_metrics():
    preds = load_predictions()
    outcomes = load_outcomes()

    if preds.empty:
        print("No predictions logged yet.")
        return None

    if outcomes.empty:
        print("No outcomes file. Create logs/outcomes.csv with 'row_index,actual' columns.")
        print("Tip: simulate by adding {'row_index': 0, 'actual': 1} rows.")
        return None

    preds = preds.reset_index().rename(columns={'index': 'row_index'})
    merged = preds.merge(outcomes, on='row_index', how='inner')

    if len(merged) < 10:
        print(f"Only {len(merged)} labeled rows. Need at least 10.")
        return None

    y = merged['actual'].values
    p = merged['probability'].values

    metrics = {
        'n_labeled': int(len(merged)),
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'brier': float(brier_score_loss(y, p)),
        'f1_at_0.5': float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
    }

    # Top-K precision at 20%
    k = int(len(merged) * 0.20)
    top = merged.nlargest(k, 'probability')
    metrics['precision_at_20'] = float(top['actual'].mean())

    return metrics


if __name__ == '__main__':
    m = compute_delayed_metrics()
    if m:
        print("=" * 50)
        print("DELAYED-LABEL PERFORMANCE")
        print("=" * 50)
        for k, v in m.items():
            print(f"  {k}: {v}")
        with open(os.path.join(REPORTS, 'delayed_performance.json'), 'w') as f:
            json.dump(m, f, indent=4)
        print(f"\nSaved: reports/delayed_performance.json")
