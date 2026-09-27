import os, joblib, json
from src.config import BASE_DIR

MODELS = os.path.join(BASE_DIR, 'models')
CANDIDATE = os.path.join(MODELS, 'production_model.pkl')
ACTIVE = os.path.join(MODELS, 'production_model_active.pkl')
META = os.path.join(MODELS, 'model_metadata.json')
ACTIVE_META = os.path.join(MODELS, 'model_metadata_active.json')

TOLERANCES = {'val_brier': 1.05, 'val_f1': 0.98, 'val_profit': 0.98}


def _safe(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def gate(candidate, production):
    """Pure function: returns (promote, checks). Safe to import in tests."""
    checks = []
    promote = True

    def _check_ge(key, label, tol=1.0):
        nonlocal promote
        c = _safe(candidate.get(key))
        p = _safe(production.get(key))
        if p > 0 and c < p * tol:
            promote = False
            checks.append(('FAIL', f"{label}: {c:.4f} < {p:.4f} (tol={tol})"))
        else:
            checks.append(('PASS', f"{label}: {c:.4f} vs {p:.4f}"))

    def _check_le(key, label, tol=1.0):
        nonlocal promote
        c = _safe(candidate.get(key), 1.0)
        p = _safe(production.get(key), 1.0)
        if c > p * tol:
            promote = False
            checks.append(('FAIL', f"{label}: {c:.4f} > {p:.4f} (tol={tol})"))
        else:
            checks.append(('PASS', f"{label}: {c:.4f} vs {p:.4f}"))

    _check_ge('val_pr_auc', 'PR-AUC', tol=1.0)
    _check_le('val_brier', 'Brier', tol=TOLERANCES['val_brier'])
    _check_ge('val_f1', 'F1', tol=TOLERANCES['val_f1'])
    _check_ge('val_profit', 'Val profit', tol=TOLERANCES['val_profit'])
    _check_ge('auuc', 'AUUC', tol=0.98)
    _check_ge('uplift_at_20', 'Uplift@20', tol=0.98)

    return promote, checks


def main():
    with open(META) as f:
        cand = json.load(f)

    if not os.path.exists(ACTIVE):
        print("No active production model. Promoting candidate immediately.")
        joblib.dump(joblib.load(CANDIDATE), ACTIVE)
        with open(ACTIVE_META, 'w') as f:
            json.dump(cand, f, indent=4)
        print(f"Active model: {ACTIVE}")
        return

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


if __name__ == "__main__":
    main()
