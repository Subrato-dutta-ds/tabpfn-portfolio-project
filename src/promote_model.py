import os, joblib, json
from src.config import BASE_DIR

MODELS = os.path.join(BASE_DIR, 'models')
CANDIDATE = os.path.join(MODELS, 'production_model.pkl')
ACTIVE = os.path.join(MODELS, 'production_model_active.pkl')
META = os.path.join(MODELS, 'model_metadata.json')
ACTIVE_META = os.path.join(MODELS, 'model_metadata_active.json')

TOLERANCES = {
    'val_brier': 1.05,
    'val_f1': 0.98,
    'val_profit': 0.98,
}


def _safe(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def gate(candidate, production):
    checks = []
    promote = True

    c_prauc = _safe(candidate.get('val_pr_auc'))
    p_prauc = _safe(production.get('val_pr_auc'))
    if c_prauc < p_prauc:
        promote = False
        checks.append(('FAIL', f"PR-AUC decreased: {c_prauc:.4f} < {p_prauc:.4f}"))
    else:
        checks.append(('PASS', f"PR-AUC: {c_prauc:.4f} >= {p_prauc:.4f}"))

    c_brier = _safe(candidate.get('val_brier'), 1.0)
    p_brier = _safe(production.get('val_brier'), 1.0)
    if c_brier > p_brier * TOLERANCES['val_brier']:
        promote = False
        checks.append(('FAIL', f"Brier worsened >5%: {c_brier:.4f} vs {p_brier:.4f}"))
    else:
        checks.append(('PASS', f"Brier within tolerance: {c_brier:.4f} vs {p_brier:.4f}"))

    c_f1 = _safe(candidate.get('val_f1'))
    p_f1 = _safe(production.get('val_f1'))
    if p_f1 > 0 and c_f1 < p_f1 * TOLERANCES['val_f1']:
        promote = False
        checks.append(('FAIL', f"F1 decreased >2%: {c_f1:.4f} vs {p_f1:.4f}"))
    else:
        checks.append(('PASS', f"F1: {c_f1:.4f} vs {p_f1:.4f}"))

    c_profit = _safe(candidate.get('val_profit'))
    p_profit = _safe(production.get('val_profit'))
    if p_profit > 0 and c_profit < p_profit * TOLERANCES['val_profit']:
        promote = False
        checks.append(('FAIL', f"Val profit decreased >2%: {c_profit:,.0f} vs {p_profit:,.0f}"))
    else:
        checks.append(('PASS', f"Val profit: {c_profit:,.0f} vs {p_profit:,.0f}"))

    return promote, checks


with open(META) as f:
    cand = json.load(f)

if not os.path.exists(ACTIVE):
    print("No active production model. Promoting candidate immediately.")
    joblib.dump(joblib.load(CANDIDATE), ACTIVE)
    with open(ACTIVE_META, 'w') as f:
        json.dump(cand, f, indent=4)
    print(f"Active model: {ACTIVE}")
    raise SystemExit(0)

with open(ACTIVE_META) as f:
    prod = json.load(f)

promote, checks = gate(cand, prod)

print("=" * 60)
print("PROMOTION GATE")
print("=" * 60)
for status, msg in checks:
    print(f"  [{status}] {msg}")
print("=" * 60)

if promote:
    joblib.dump(joblib.load(CANDIDATE), ACTIVE)
    with open(ACTIVE_META, 'w') as f:
        json.dump(cand, f, indent=4)
    print(f"\nPROMOTED candidate to {ACTIVE}")
else:
    print(f"\nREJECTED candidate. Keeping active model.")
