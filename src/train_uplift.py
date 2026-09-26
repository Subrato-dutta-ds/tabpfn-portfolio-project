import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.data_loader import load_data
from src.config import BASE_DIR
from src.schema import RANDOM_STATE, TARGET_COLUMN, DROPPED_COLUMNS

np.random.seed(123)
df = load_data()
X = df.drop(columns=[TARGET_COLUMN] + DROPPED_COLUMNS, errors='ignore')
y = df[TARGET_COLUMN].map({'yes': 1, 'no': 0}).values

# --- SIMULATED treatment/control (NOT real randomized data) ---
treatment = np.random.binomial(1, 0.5, len(X))
boost = np.where((X['age'].values < 40) & (treatment == 1), 0.15, 0.0)
y_sim = ((y + boost) > np.random.rand(len(X)) * 1.2).astype(int)

X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
    X, y_sim, treatment, test_size=0.2, random_state=RANDOM_STATE, stratify=treatment)

cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns


def make_preprocessor():
    """Fresh preprocessor per T-learner arm (avoids feature-shape mismatch)."""
    return ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])


# Treatment arm: separate pipeline AND preprocessor
mt = Pipeline([
    ('preprocessor', make_preprocessor()),
    ('classifier', XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE))
])

# Control arm: separate pipeline AND preprocessor
mc = Pipeline([
    ('preprocessor', make_preprocessor()),
    ('classifier', XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE))
])

mt.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
mc.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])

p_t = mt.predict_proba(X_te)[:, 1]
p_c = mc.predict_proba(X_te)[:, 1]
uplift = p_t - p_c

results = pd.DataFrame({
    'customer_id': [f'CUST_{i:06d}' for i in range(len(X_te))],
    'p_treatment': p_t,
    'p_control': p_c,
    'uplift': uplift,
}).sort_values('uplift', ascending=False).reset_index(drop=True)

results.to_csv(os.path.join(BASE_DIR, 'reports', 'uplift_scores.csv'), index=False)
joblib.dump(mt, os.path.join(BASE_DIR, 'models', 'uplift_treatment.pkl'))
joblib.dump(mc, os.path.join(BASE_DIR, 'models', 'uplift_control.pkl'))

print(f'Uplift trained on {len(X_tr)} rows')
print(f'Treatment group size: {(t_tr == 1).sum()}')
print(f'Control group size:   {(t_tr == 0).sum()}')
print(f'Mean uplift: {uplift.mean():.4f}')
print(f'Median uplift: {np.median(uplift):.4f}')
print(f'% positive uplift: {(uplift > 0).mean() * 100:.1f}%')
print(f'Top-100 mean uplift: {uplift[np.argsort(uplift)[::-1][:100]].mean():.4f}')
print()
print('Saved: reports/uplift_scores.csv')
print('Saved: models/uplift_treatment.pkl, models/uplift_control.pkl')
print()
print('CAVEAT: treatment is SYNTHETIC. Bank Marketing dataset is observational.')
