import os, joblib, json, numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss
from src.config import BASE_DIR
from src.schema import REVENUE_PER_SUBSCRIPTION, COST_PER_CONTACT

reports_dir = os.path.join(BASE_DIR, 'reports')

# Load frozen artifacts (do NOT overwrite anything!)
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'production_model.pkl'))
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json')) as f:
    meta = json.load(f)

X_test = pd.read_csv(os.path.join(reports_dir, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(reports_dir, 'y_test.csv'))['y']

threshold_f1 = meta['threshold_f1']
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Classification metrics at F1 threshold
y_pred_f1 = (y_proba >= threshold_f1).astype(int)
f1 = f1_score(y_test, y_pred_f1)
pr_auc = average_precision_score(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
brier = brier_score_loss(y_test, y_proba)

# Campaign metrics (top 20% budget)
k = int(len(y_test) * 0.20)
top_idx = np.argsort(y_proba)[::-1][:k]
p_at_20 = y_test.iloc[top_idx].mean()
lift_at_20 = p_at_20 / y_test.mean()
exp_profit_top20 = float(np.sum(y_proba[top_idx] * REVENUE_PER_SUBSCRIPTION - COST_PER_CONTACT))

# Final report (single row — one model, one test set)
final = pd.DataFrame([{
    'Model': meta['model'],
    'Threshold_F1': threshold_f1,
    'Test_F1': round(f1, 4),
    'Test_PR_AUC': round(pr_auc, 4),
    'Test_ROC_AUC': round(roc_auc, 4),
    'Test_Brier': round(brier, 4),
    'Precision@20%': round(p_at_20, 4),
    'Lift@20%': round(lift_at_20, 2),
    'Expected_Profit@20%': round(exp_profit_top20, 2),
}])
final.to_csv(os.path.join(reports_dir, 'final_test_results.csv'), index=False)

# Update metadata with test results (append, don't overwrite selection info)
meta.update({
    'test_f1': float(f1),
    'test_pr_auc': float(pr_auc),
    'test_roc_auc': float(roc_auc),
    'test_brier': float(brier),
    'test_lift_at_20': float(lift_at_20),
})
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=4)

print(f'=== FINAL TEST RESULTS (untouched holdout set) ===')
print(f'Model: {meta["model"]}')
print(f'F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}')
print(f'Brier: {brier:.4f} (calibrated)')
print(f'Lift@20%: {lift_at_20:.2f}x | P@20%: {p_at_20:.4f}')
print(f'Expected Profit@20%: Rs {exp_profit_top20:,.0f}')
print('Saved: reports/final_test_results.csv')
