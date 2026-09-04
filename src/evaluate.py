import os, joblib, pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, brier_score_loss, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_loader import load_data
from src.config import RANDOM_STATE, BASE_DIR

df = load_data()
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_test.to_csv(os.path.join(BASE_DIR, 'reports', 'X_test.csv'), index=False)

# Load deployed model and metadata
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json')) as f:
    metadata = json.load(f)
deployed_threshold = metadata['threshold']

# Calculate probabilities
y_proba = pipeline.predict_proba(X_test)[:, 1]

# FIX #1: Use deployed threshold for final report
final_predictions = (y_proba >= deployed_threshold).astype(int)
f1 = f1_score(y_test, final_predictions)
pr_auc = average_precision_score(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
brier = brier_score_loss(y_test, y_proba)

# Lift / Gain Analysis
k = int(len(y_test) * 0.20)
top_indices = np.argsort(y_proba)[::-1][:k]
precision_at_20 = y_test.iloc[top_indices].mean()
baseline_rate = y_test.mean()
lift_at_20 = precision_at_20 / baseline_rate

# Save updated metadata
metadata.update({
    'f1': f1,
    'pr_auc': pr_auc,
    'roc_auc': auc,
    'precision_at_20': precision_at_20,
    'lift_at_20': lift_at_20,
    'brier_score': brier
})
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

# Save comparison table
results = [{'Model': metadata['model'], 'F1': f1, 'ROC-AUC': auc, 'PR-AUC': pr_auc, 'Precision@20%': precision_at_20, 'Lift@20%': lift_at_20, 'Brier': brier}]
pd.DataFrame(results).to_csv(os.path.join(BASE_DIR, 'reports', 'performance_table.csv'), index=False)

print(f"Final Results (using deployed threshold {deployed_threshold}):")
print(f"  F1: {f1:.4f}")
print(f"  PR-AUC: {pr_auc:.4f}")
print(f"  Lift@20%: {lift_at_20:.2f}x")
print(f"  Brier Score: {brier:.4f}")
