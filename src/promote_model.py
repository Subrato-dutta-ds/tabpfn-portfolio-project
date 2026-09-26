import os, joblib, json
from src.config import BASE_DIR

MODELS = os.path.join(BASE_DIR, 'models')
CANDIDATE = os.path.join(MODELS, 'production_model.pkl')
ACTIVE = os.path.join(MODELS, 'production_model_active.pkl')
META = os.path.join(MODELS, 'model_metadata.json')
ACTIVE_META = os.path.join(MODELS, 'model_metadata_active.json')

def promote_check(cand, prod):
    reasons = []
    promote = True
    if cand['val_pr_auc'] < prod['val_pr_auc']:
        promote = False
        reasons.append(f"PR-AUC worse: {cand['val_pr_auc']:.4f} < {prod['val_pr_auc']:.4f}")
    if cand['val_brier'] > prod['val_brier'] * 1.05:
        promote = False
        reasons.append(f"Brier worse by >5%: {cand['val_brier']:.4f} vs {prod['val_brier']:.4f}")
    if promote:
        reasons.append("All criteria met")
    return promote, reasons

with open(META) as f:
    cand_meta = json.load(f)

cand = {'val_pr_auc': cand_meta.get('val_pr_auc', 0),
        'val_brier': cand_meta.get('val_brier', 1)}

if not os.path.exists(ACTIVE):
    print("No active production model. Promoting candidate immediately.")
    joblib.dump(joblib.load(CANDIDATE), ACTIVE)
    with open(ACTIVE_META, 'w') as f:
        json.dump(cand_meta, f, indent=4)
    print(f"Active model: {ACTIVE}")
    raise SystemExit(0)

with open(ACTIVE_META) as f:
    active_meta = json.load(f)

prod = {'val_pr_auc': active_meta.get('val_pr_auc', 0),
        'val_brier': active_meta.get('val_brier', 1)}

promote, reasons = promote_check(cand, prod)
print("Promotion check:")
for r in reasons:
    print(f"  - {r}")

if promote:
    joblib.dump(joblib.load(CANDIDATE), ACTIVE)
    with open(ACTIVE_META, 'w') as f:
        json.dump(cand_meta, f, indent=4)
    print(f"\nPromoted -> {ACTIVE}")
else:
    print(f"\nRejected. Keeping active model.")
