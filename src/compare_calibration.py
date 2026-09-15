import os, joblib, json, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, average_precision_score
from src.config import BASE_DIR

# Load the raw (uncalibrated) candidates and compare calibration methods on validation
from src.data_loader import load_data
from src.schema import TARGET_COLUMN, DROPPED_COLUMNS, RANDOM_STATE
from sklearn.model_selection import train_test_split

df = load_data()
X = df.drop(columns=[TARGET_COLUMN] + DROPPED_COLUMNS, errors='ignore')
y = df[TARGET_COLUMN].map({'yes': 1, 'no': 0})

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full)

with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json')) as f:
    meta = json.load(f)
winner = meta['model']
safe = winner.lower().replace(' ', '_')
raw = joblib.load(os.path.join(BASE_DIR, 'models', 'candidates', f'{safe}_raw.pkl'))

results = []

# 1. Raw
p_raw = raw.predict_proba(X_val)[:, 1]
results.append({'Method': 'Raw', 'Brier': round(brier_score_loss(y_val, p_raw), 4), 'PR_AUC': round(average_precision_score(y_val, p_raw), 4)})

# 2. Sigmoid (Platt)
cal_sig = CalibratedClassifierCV(raw, method='sigmoid', cv=5)
cal_sig.fit(X_train, y_train)
p_sig = cal_sig.predict_proba(X_val)[:, 1]
results.append({'Method': 'Sigmoid (Platt)', 'Brier': round(brier_score_loss(y_val, p_sig), 4), 'PR_AUC': round(average_precision_score(y_val, p_sig), 4)})

# 3. Isotonic
cal_iso = CalibratedClassifierCV(raw, method='isotonic', cv=5)
cal_iso.fit(X_train, y_train)
p_iso = cal_iso.predict_proba(X_val)[:, 1]
results.append({'Method': 'Isotonic', 'Brier': round(brier_score_loss(y_val, p_iso), 4), 'PR_AUC': round(average_precision_score(y_val, p_iso), 4)})

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(BASE_DIR, 'reports', 'calibration_comparison.csv'), index=False)
print(df_results.to_string(index=False))
print()
print(f'Winner for production: {winner}')
print(f'Calibration comparison saved to reports/calibration_comparison.csv')
