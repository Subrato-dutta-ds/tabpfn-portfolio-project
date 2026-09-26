import numpy as np
import pandas as pd
from pathlib import Path
import json


def compute_psi(expected, actual, bins=10, eps=1e-6):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    e_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    a_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def load_prediction_logs(path):
    entries = []
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    with open(p) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(entries)
