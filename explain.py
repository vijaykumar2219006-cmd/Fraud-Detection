"""Simple, model-agnostic explanation ("occlusion" / what-if analysis).

Idea: take the transaction being scored and, one feature at a time, replace that
feature with the customer's TYPICAL value (median of their previous transactions).
If the fraud probability drops a lot, that feature was a main reason for the alert.
"""
import numpy as np


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))

FEATURES = ["Amount", "Hour of day", "Distance from home", "Time since last txn", "Merchant category"]
_NUM_IDX = {"Amount": [0], "Hour of day": [1, 2], "Distance from home": [3], "Time since last txn": [4]}


def explain(score_fn, buf):
    """buf: sequence of (numeric_vector, category_id); the LAST item is the transaction being scored.
    Returns (fraud_probability, {feature: contribution}) where contribution is the drop in
    fraud log-odds when that feature is replaced by the customer's typical value.
    (Log-odds are used because they don't saturate when several features are suspicious at once.)"""
    items = list(buf)
    base = score_fn(items)
    hist = items[:-1]
    if not hist:
        return base, {f: 0.0 for f in FEATURES}
    vecs = np.stack([v for v, _ in hist])
    cats = [int(c) for _, c in hist]
    median = np.median(vecs, axis=0)
    usual_cat = max(set(cats), key=cats.count)
    last_v, last_c = items[-1]

    contrib = {}
    for name, idx in _NUM_IDX.items():
        v = last_v.copy()
        v[idx] = median[idx]
        contrib[name] = _logit(base) - _logit(score_fn(hist + [(v, last_c)]))
    contrib["Merchant category"] = _logit(base) - _logit(score_fn(hist + [(last_v, usual_cat)]))
    return base, contrib
