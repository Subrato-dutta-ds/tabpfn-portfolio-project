import os, joblib, json
from src.config import BASE_DIR

MODELS = os.path.join(BASE_DIR, 'models')
CANDIDATE = os.path.join(MODELS, 'production_model.pkl')
ACTIVE = os.path.join(MODELS, 'production_model_active.pkl')
META = os.path.join(MODELS, 'model_metadata.json')
ACTIVE_META = os.path.join(MODELS, 'model_metadata_active.json')

TOLERANCES = {'test_brier': 1.05, 'test_f1': 0.98, 'test_pr_auc': 1.0}


def _safe(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def gate(candidate, production):
    """Pure function: promotion based on TEST metrics, not validation."""
    checks = []
    promote = True

    def _ge(c_key, p_key, label, tol=1.0):
        nonlocal promote
        c = _safe(candidate.get(c_key))
        p = _safe(production.get(p_key))
        if p > 0 and c < p * tol:
            promote = False
            checks.append(('FAIL', f"{label}: {c:.4f} < {p:.4f} (tol={tol})"))
        else:
            checks.append(('PASS', f"{label}: {c:.4f} vs {p:.4f}"))

    def _le(c_key, p_key, label, tol=1.0):
        nonlocal promote
        c = _safe(candidate.get(c_key), 1.0)
        p = _safe(production.get(p_key), 1.0)
        if c > p * tol:
            promote = False
            checks.append(('FAIL', f"{label}: {c:.4f} > {p:.4f} (tol={tol})"))
        else:
            checks.append(('PASS', f"{label}: {c:.4f} vs {p:.4f}"))

    _ge('test_pr_auc', 'test_pr_auc', 'Test PR-AUC', tol=1.0)
    _le('test_brier', 'test_brier', 'Test Brier', tol=1.05)
    _ge('test_f1', 'test_f1', 'Test F1', tol=0.98)
    _ge('auuc', 'auuc', 'AUUC', tol=0.98)
    _ge('uplift_at_20', 'uplift_at_20', 'Uplift@20', tol=0.98)

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
    print("PROMOTION GATE (test-set metrics)")
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
