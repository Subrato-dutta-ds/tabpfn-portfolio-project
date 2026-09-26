import os, joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.config import BASE_DIR
from src.data_loader import load_data
from src.schema import RANDOM_STATE, TARGET_COLUMN, DROPPED_COLUMNS

REPORTS = os.path.join(BASE_DIR, 'reports')

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

eval_df = pd.DataFrame({'uplift': uplift, 'y': y_te, 't': t_te}).sort_values('uplift', ascending=False).reset_index(drop=True)

n = len(eval_df)
deciles = []
for i in range(10):
    lo, hi = i * n // 10, (i + 1) * n // 10
    seg = eval_df.iloc[lo:hi]
    t_rate = seg[seg['t'] == 1]['y'].mean() if (seg['t'] == 1).sum() > 0 else 0
    c_rate = seg[seg['t'] == 0]['y'].mean() if (seg['t'] == 0).sum() > 0 else 0
    deciles.append({'Decile': i + 1, 'N': len(seg),
                    'Treatment_Rate': round(float(t_rate), 4),
                    'Control_Rate': round(float(c_rate), 4),
                    'Incremental_Uplift': round(float(t_rate - c_rate), 4)})
decile_df = pd.DataFrame(deciles)
decile_df.to_csv(os.path.join(REPORTS, 'uplift_deciles.csv'), index=False)
print(decile_df.to_string(index=False))

qs = np.linspace(0, 1, 100)
n_t = (eval_df['t'] == 1).sum()
n_c = (eval_df['t'] == 0).sum()
cum_t = eval_df.loc[eval_df['t'] == 1, 'y'].cumsum().values
cum_c = eval_df.loc[eval_df['t'] == 0, 'y'].cumsum().values
qini = []
for q in qs:
    idx_t = int(q * n_t) - 1
    idx_c = int(q * n_c) - 1
    if idx_t < 0 or idx_c < 0:
        qini.append(0.0)
    else:
        qini.append(float(cum_t[min(idx_t, len(cum_t)-1)] - cum_c[min(idx_c, len(cum_c)-1)] * (n_t / max(n_c, 1))))
qini = np.array(qini)
auuc = float(np.trapezoid(qini, qs))

plt.figure(figsize=(8, 5))
plt.plot(qs, qini, label=f'Model (AUUC={auuc:.2f})', color='#00D9B8', linewidth=2)
plt.plot(qs, qs * qini[-1], label='Random', linestyle='--', color='gray')
plt.xlabel('Fraction of population targeted')
plt.ylabel('Cumulative incremental conversions')
plt.title('Qini Curve (SYNTHETIC treatment)')
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

summary = {'auuc': auuc,
           'uplift@10pct': uplift_at_k(eval_df, 0.10),
           'uplift@20pct': uplift_at_k(eval_df, 0.20),
           'uplift@30pct': uplift_at_k(eval_df, 0.30)}
pd.DataFrame([summary]).to_csv(os.path.join(REPORTS, 'uplift_summary.csv'), index=False)

print()
print(f'AUUC: {auuc:.4f}')
print(f'Uplift@10%: {summary["uplift@10pct"]:.4f}')
print(f'Uplift@20%: {summary["uplift@20pct"]:.4f}')
print(f'Uplift@30%: {summary["uplift@30pct"]:.4f}')
print()
print('Saved: reports/uplift_deciles.csv, qini_curve.png, uplift_summary.csv')
print('CAVEAT: treatment is SYNTHETIC.')
