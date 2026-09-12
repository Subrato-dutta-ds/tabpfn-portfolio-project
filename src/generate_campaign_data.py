import os, numpy as np, pandas as pd
from src.data_loader import load_data
from src.schema import FEATURE_SCHEMA, TARGET_COLUMN, DROPPED_COLUMNS
from src.config import BASE_DIR

np.random.seed(999)
df = load_data()
X = df.drop(columns=[TARGET_COLUMN] + DROPPED_COLUMNS, errors='ignore')

INTEGER_COLS = ['age', 'campaign', 'pdays', 'previous']
n = 2000

sampled = X.sample(n=n, replace=True, random_state=999).reset_index(drop=True)
source_ids = X.sample(n=n, replace=True, random_state=999).index.tolist()

for col in FEATURE_SCHEMA['numerical']:
    if col in sampled.columns:
        spec = FEATURE_SCHEMA['numerical'][col]
        span = spec['max'] - spec['min']
        noise = np.random.normal(0, 0.01 * span, n)
        noisy = np.clip(sampled[col].values + noise, spec['min'], spec['max'])
        if col in INTEGER_COLS:
            sampled[col] = np.round(noisy).astype(int)
        else:
            sampled[col] = np.round(noisy, 2)

sampled.insert(0, 'customer_id', [f'CUST_{i:06d}' for i in range(1, n + 1)])

out = os.path.join(BASE_DIR, 'data', 'synthetic_campaign_customers.csv')
sampled.to_csv(out, index=False)

n_unique = len(set(source_ids))
dup_pct = (1 - n_unique / n) * 100
print(f'Generated {len(sampled)} synthetic customers -> {out}')
print(f'Integer columns preserved: {INTEGER_COLS}')
print(f'Unique source rows: {n_unique}/{n} ({dup_pct:.1f}% duplicates from bootstrap)')
print(f'customer_id unique: {sampled["customer_id"].is_unique}')
