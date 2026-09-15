import os, joblib, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.data_loader import load_data
from src.config import BASE_DIR, get_models, get_param_grids
from src.schema import RANDOM_STATE, TARGET_COLUMN, DROPPED_COLUMNS, REVENUE_PER_SUBSCRIPTION, COST_PER_CONTACT

candidates_dir = os.path.join(BASE_DIR, 'models', 'candidates')
os.makedirs(candidates_dir, exist_ok=True)
reports_dir = os.path.join(BASE_DIR, 'reports')
os.makedirs(reports_dir, exist_ok=True)

df = load_data()
X = df.drop(columns=[TARGET_COLUMN] + DROPPED_COLUMNS, errors='ignore')
y = df[TARGET_COLUMN].map({'yes': 1, 'no': 0})

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full)

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

models = get_models()
grids = get_param_grids()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

comparison = []
best_score = -1
best_name = None
best_thresh = 0.5
best_calibrated = None
best_profit_thresh = 0.5

for name, model in models.items():
    print(f'Tuning {name}...')
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    if name in grids:
        search = RandomizedSearchCV(pipe, grids[name], n_iter=3, cv=cv, scoring='average_precision', n_jobs=-1, random_state=RANDOM_STATE)
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
    else:
        pipe.fit(X_train, y_train)

    safe = name.lower().replace(' ', '_')
    joblib.dump(pipe, os.path.join(candidates_dir, f'{safe}_raw.pkl'))

    # Calibration method = isotonic. Verified via src/compare_calibration.py (lowest validation Brier).
    try:
        calibrated = CalibratedClassifierCV(pipe, method='isotonic', cv=5)
        calibrated.fit(X_train, y_train)
        print(f'  Calibrated {name} with isotonic regression')
    except Exception as e:
        print(f'  WARNING: Calibration failed for {name}: {str(e)[:80]}')
        print(f'  Falling back to uncalibrated pipeline')
        calibrated = pipe
    joblib.dump(calibrated, os.path.join(candidates_dir, f'{safe}_calibrated.pkl'))

    val_proba = calibrated.predict_proba(X_val)[:, 1]
    val_pr_auc = average_precision_score(y_val, val_proba)

    best_f1, thresh_f1 = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        f1 = f1_score(y_val, (val_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, thresh_f1 = f1, t

    profit_per_thresh = []
    for t in np.arange(0.01, 0.99, 0.01):
        sel = val_proba >= t
        ep = float(np.sum(val_proba[sel] * REVENUE_PER_SUBSCRIPTION - COST_PER_CONTACT)) if sel.sum() else 0
        profit_per_thresh.append((t, ep))
    thresh_profit, best_profit = max(profit_per_thresh, key=lambda x: x[1])

    comparison.append({
        'Model': name,
        'Val_PR_AUC': round(val_pr_auc, 4),
        'Val_Brier': round(float(np.mean((val_proba - y_val) ** 2)), 4),
        'Val_F1_Best': round(best_f1, 4),
        'Threshold_F1': round(thresh_f1, 2),
        'Threshold_Profit': round(thresh_profit, 3),
        'Val_Profit': round(best_profit, 2),
    })
    print(f'  {name}: PR-AUC={val_pr_auc:.4f}, Profit@best=Rs {best_profit:,.0f}')

    if val_pr_auc > best_score:
        best_score = val_pr_auc
        best_name = name
        best_thresh = thresh_f1
        best_profit_thresh = thresh_profit
        best_calibrated = calibrated

joblib.dump(best_calibrated, os.path.join(BASE_DIR, 'models', 'production_model.pkl'))

comparison_df = pd.DataFrame(comparison).sort_values('Val_PR_AUC', ascending=False)
comparison_df.to_csv(os.path.join(reports_dir, 'model_comparison.csv'), index=False)

X_test.to_csv(os.path.join(reports_dir, 'X_test.csv'), index=False)
pd.DataFrame({'y': y_test}).to_csv(os.path.join(reports_dir, 'y_test.csv'), index=False)

import subprocess
from datetime import datetime

try:
    git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=BASE_DIR, stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    git_sha = 'unknown'

try:
    import sklearn, xgboost, pandas, numpy
    env_versions = {
        'scikit-learn': sklearn.__version__,
        'xgboost': xgboost.__version__,
        'pandas': pandas.__version__,
        'numpy': numpy.__version__,
    }
except Exception:
    env_versions = {}

with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json'), 'w') as f:
    json.dump({
        'model': best_name,
        'threshold_f1': float(best_thresh),
        'threshold_profit': float(best_profit_thresh),
        'primary_metric': 'PR-AUC (validation)',
        'revenue_per_subscription': REVENUE_PER_SUBSCRIPTION,
        'cost_per_contact': COST_PER_CONTACT,
        'calibration': 'isotonic (5-fold CV)',
        'model_version': git_sha,
        'git_sha': git_sha,
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'feature_schema_version': '1.0',
        'training_rows': int(len(X_train)),
        'validation_rows': int(len(X_val)),
        'test_rows': int(len(X_test)),
        'feature_count': int(X_train.shape[1]),
        'env_versions': env_versions,
    }, f, indent=4)

print(f'\nWinner: {best_name} (Val PR-AUC={best_score:.4f})')
print(f'F1 threshold: {best_thresh:.2f}, Profit threshold: {best_profit_thresh:.2f}')
print('Calibrated model saved to models/production_model.pkl')
