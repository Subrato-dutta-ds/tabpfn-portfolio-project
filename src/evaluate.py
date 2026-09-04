import os, joblib, pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from src.data_loader import load_data

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json')) as f:
    metadata = json.load(f)
threshold = metadata['threshold']

# Load Data
df = load_data()
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Final evaluation on untouched test set
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

f1 = f1_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

k = int(len(y_test) * 0.20)
top_indices = np.argsort(y_proba)[::-1][:k]
precision_at_20 = y_test.iloc[top_indices].mean()

results = [{'Model': metadata['model'], 'F1': f1, 'ROC-AUC': auc, 'PR-AUC': pr_auc, 'Precision@20%': precision_at_20, 'Optimal Threshold': threshold}]
pd.DataFrame(results).to_csv(os.path.join(BASE_DIR, 'reports', 'performance_table.csv'), index=False)
print(f"Final Test Set -> F1: {f1:.4f}, P@20: {precision_at_20:.4f}")
