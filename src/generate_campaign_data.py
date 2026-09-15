import os, numpy as np, pandas as pd
from src.data_loader import load_data
from src.schema import FEATURE_SCHEMA, TARGET_COLUMN, DROPPED_COLUMNS
from src.config import BASE_DIR

np.random.seed(999)
df = load_data()
X = df.drop(columns=[TARGET_COLUMN] + DROPPED_COLUMNS, errors='ignore')

INTEGER_COLS = ['age', 'campaign', 'pdays', 'previous']
n = 2000

sampled = X.sample(n=n, replace=True, random_state=999)
source_ids = sampled.index.tolist()
sampled = sampled.reset_index(drop=True)

for col in FEATURE_SCHEMA['numerical']:
    if col in sampled.columns:
        spec = FEATURE_SCHEMA['numerical'][col]
        span = spec['max'] - spec['min']

        if col == 'pdays':
            # Preserve 999 special value
            original = sampled[col].values.copy()
            mask_999 = original == 999
            noise = np.random.normal(0, 0.01 * span, n)
            noisy = np.clip(original + noise, spec['min'], spec['max'])
            noisy = np.round(noisy).astype(int)
            noisy[mask_999] = 999
            # If noise pushed a non-999 value to 999, push it to 998
            noisy[(noisy == 999) & (~mask_999)] = 998
            sampled[col] = noisy
        else:
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
print(f'pdays=999 preserved: {(sampled["pdays"] == 999).sum()} rows')
print(f'Unique source rows: {n_unique}/{n} ({dup_pct:.1f}% bootstrap duplicates)')
print(f'customer_id unique: {sampled["customer_id"].is_unique}')
