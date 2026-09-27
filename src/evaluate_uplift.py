import os, json, joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.config import BASE_DIR
from src.data_loader import load_data
from src.schema import RANDOM_STATE, TARGET_COLUMN, DROPPED_COLUMNS

REPORTS = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS, exist_ok=True)

np.random.seed(123)
df = load_data()
X = df.drop(columns=[TARGET_COLUMN] + DROPPED_COLUMNS, errors='ignore')
y = df[TARGET_COLUMN].map({'yes': 1, 'no': 0}).values
treatment = np.random.binomial(1, 0.5, len(X))
boost = np.where((X['age'].values < 40) & (treatment == 1), 0.15, 0.0)
y_sim = ((y + boost) > np.random.rand(len(X)) * 1.2).astype(int)

X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
    X, y_sim, treatment, test_size=0.2, random_state=RANDOM_STATE, stratify=treatment)

mt = joblib.load(os.path.join(BASE_DIR, 'models', 'uplift_treatment.pkl'))
mc = joblib.load(os.path.join(BASE_DIR, 'models', 'uplift_control.pkl'))
p_t = mt.predict_proba(X_te)[:, 1]
p_c = mc.predict_proba(X_te)[:, 1]
uplift = p_t - p_c

eval_df = pd.DataFrame({'uplift': uplift, 'y': y_te, 't': t_te})

# --- Uplift deciles ---
sorted_df = eval_df.sort_values('uplift', ascending=False).reset_index(drop=True)
n = len(sorted_df)
deciles = []
for i in range(10):
    lo, hi = i * n // 10, (i + 1) * n // 10
    seg = sorted_df.iloc[lo:hi]
    t_rate = seg[seg['t'] == 1]['y'].mean() if (seg['t'] == 1).sum() > 0 else 0
    c_rate = seg[seg['t'] == 0]['y'].mean() if (seg['t'] == 0).sum() > 0 else 0
    deciles.append({'Decile': i + 1, 'N': len(seg),
                    'Treatment_Rate': round(float(t_rate), 4),
                    'Control_Rate': round(float(c_rate), 4),
                    'Incremental_Uplift': round(float(t_rate - c_rate), 4)})
decile_df = pd.DataFrame(deciles)
decile_df.to_csv(os.path.join(REPORTS, 'uplift_deciles.csv'), index=False)
print(decile_df.to_string(index=False))


def qini_curve(y, t, scores):
    """Returns (qini_values, population_fraction) for a given score ordering."""
    order = np.argsort(scores)[::-1]
    y, t = y[order], t[order]
    n_t = (t == 1).sum()
    n_c = (t == 0).sum()
    if n_t == 0 or n_c == 0:
        return np.zeros(len(y)), np.linspace(0, 1, len(y))
    cum_t = np.cumsum(y * (t == 1))
    cum_c = np.cumsum(y * (t == 0))
    qini = cum_t - cum_c * (n_t / n_c)
    pop = np.arange(1, len(y) + 1) / len(y)
    return qini, pop


# Model Qini
qini_model, pop = qini_curve(y_te, t_te, uplift)
auuc_model = float(np.trapezoid(qini_model, pop))

# Random Qini (shuffle)
np.random.seed(42)
qini_random, _ = qini_curve(y_te, t_te, np.random.rand(len(y_te)))
auuc_random = float(np.trapezoid(qini_random, pop))

# Perfect Qini (order by true treatment effect — we know it because synthetic)
true_effect = np.where((X_te['age'].values < 40), 0.15, 0.0)
qini_perfect, _ = qini_curve(y_te, t_te, true_effect)
auuc_perfect = float(np.trapezoid(qini_perfect, pop))

# Normalized Qini
denom = auuc_perfect - auuc_random
qini_norm = (auuc_model - auuc_random) / denom if denom != 0 else 0.0

plt.figure(figsize=(8, 5))
plt.plot(pop, qini_model, label=f'Model (AUUC={auuc_model:.2f})', color='#00D9B8', linewidth=2)
plt.plot(pop, qini_random, label=f'Random (AUUC={auuc_random:.2f})', linestyle='--', color='gray')
plt.plot(pop, qini_perfect, label=f'Perfect (AUUC={auuc_perfect:.2f})', linestyle=':', color='orange')
plt.xlabel('Fraction of population targeted')
plt.ylabel('Cumulative incremental conversions')
plt.title('Qini Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, 'qini_curve.png'), dpi=100)
plt.close()


def uplift_at_k(df_, k_frac):
    k = int(len(df_) * k_frac)
    top = df_.head(k)
    t_rate = top[top['t'] == 1]['y'].mean() if (top['t'] == 1).sum() > 0 else 0
    c_rate = top[top['t'] == 0]['y'].mean() if (top['t'] == 0).sum() > 0 else 0
    return float(t_rate - c_rate)


summary = {
    'auuc_model': auuc_model,
    'auuc_random': auuc_random,
    'auuc_perfect': auuc_perfect,
    'qini_normalized': float(qini_norm),
    'uplift@10pct': uplift_at_k(sorted_df, 0.10),
    'uplift@20pct': uplift_at_k(sorted_df, 0.20),
    'uplift@30pct': uplift_at_k(sorted_df, 0.30),
}
pd.DataFrame([summary]).to_csv(os.path.join(REPORTS, 'uplift_summary.csv'), index=False)

# Patch metadata
meta_path = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    meta['auuc'] = auuc_model
    meta['qini_normalized'] = float(qini_norm)
    meta['uplift_at_20'] = float(summary['uplift@20pct'])
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print('Patched uplift metrics into model_metadata.json')

print()
print(f'AUUC (model):    {auuc_model:.4f}')
print(f'AUUC (random):   {auuc_random:.4f}')
print(f'AUUC (perfect):  {auuc_perfect:.4f}')
print(f'Normalized Qini: {qini_norm:.4f}')
print(f'Uplift@20%:      {summary["uplift@20pct"]:.4f}')
print()
print('CAVEAT: treatment is SYNTHETIC.')
