"""Pure numpy/pandas helpers (no torch): risk levels, batch CSV scoring, curves."""
from collections import deque
import numpy as np
import pandas as pd
from data import CATS, SEQ_LEN, encode, generate

REQUIRED = ["user", "timestamp", "amount", "category", "distance_km"]


def risk_level(p, thr):
    """High = above the alert threshold, Medium = above half of it, otherwise Low."""
    if p >= thr:
        return "🔴 High"
    if p >= thr * 0.5:
        return "🟡 Medium"
    return "🟢 Low"


# ------------------------------------------------------------------ batch CSV
def sample_csv():
    """A small example file in the exact format the Batch tab expects."""
    d = generate(n_users=6, txns_per_user=25, compromised_rate=0.5, seed=21)
    base = pd.Timestamp("2026-01-01")
    out = pd.DataFrame({
        "user": ["C" + str(u).zfill(3) for u in d.user],
        "timestamp": [base + pd.Timedelta(seconds=int(round(t))) for t in d.ts],
        "amount": d.amount,
        "category": [CATS[c] for c in d.cat],
        "distance_km": d.dist.round(1),
    })
    return out.sort_values("timestamp").reset_index(drop=True)


def prepare_batch(df):
    """Validate an uploaded file. Returns (clean_dataframe, list_of_warnings)."""
    cols = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Required columns: {', '.join(REQUIRED)}")
    d = df.rename(columns={cols[c]: c for c in REQUIRED})[REQUIRED].copy()
    warnings = []
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d["distance_km"] = pd.to_numeric(d["distance_km"], errors="coerce")
    bad = d[["timestamp", "amount", "distance_km"]].isna().any(axis=1)
    if bad.any():
        warnings.append(f"{int(bad.sum())} row(s) with invalid timestamp/amount/distance were skipped.")
        d = d[~bad]
    d["category"] = d["category"].astype(str).str.strip().str.lower()
    unknown = ~d["category"].isin(CATS)
    if unknown.any():
        warnings.append(f"{int(unknown.sum())} row(s) had an unknown category and were treated as 'online'. "
                        f"Known categories: {', '.join(CATS)}.")
        d.loc[unknown, "category"] = "online"
    d["user"] = d["user"].astype(str)
    d = d.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return d, warnings


def score_batch(d, score_fn, thr, explain_fn=None):
    """Score rows in time order, keeping a rolling history per customer.
    score_fn(buffer) -> probability ; explain_fn(buffer) -> {feature: contribution}."""
    bufs, last_ts, out = {}, {}, []
    for r in d.itertuples(index=False):
        # first transaction of a customer has no previous one: assume a typical 6 h gap
        dt = (r.timestamp - last_ts[r.user]).total_seconds() if r.user in last_ts else 6 * 3600.0
        last_ts[r.user] = r.timestamp
        hour = r.timestamp.hour + r.timestamp.minute / 60
        buf = bufs.setdefault(r.user, deque(maxlen=SEQ_LEN))
        buf.append((encode(r.amount, hour, r.distance_km, max(dt, 1.0)), CATS.index(r.category)))
        p = score_fn(buf)
        reason = ""
        if explain_fn is not None and p >= thr:
            contrib = explain_fn(buf)
            reason = max(contrib, key=contrib.get)
        out.append({"user": r.user, "timestamp": r.timestamp, "amount": r.amount, "category": r.category,
                    "distance_km": r.distance_km, "fraud_prob": round(float(p), 4),
                    "risk": risk_level(p, thr), "main_reason": reason})
    return pd.DataFrame(out)


# --------------------------------------------------------- curves for results
def curve_pack(y, probs):
    """probs: {model_name: scores}. Returns ROC and PR curves resampled on a common 0..1 grid."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    grid = np.linspace(0, 1, 101)
    roc, pr = {}, {}
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        roc[name] = np.interp(grid, fpr, tpr).round(4).tolist()
        prec, rec, _ = precision_recall_curve(y, p)
        pr[name] = np.interp(grid, rec[::-1], prec[::-1]).round(4).tolist()
    return {"grid": grid.round(2).tolist(), "roc": roc, "pr": pr}


def confusion(y, p, thr):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, p >= thr, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# ------------------------------------------------------------- drift monitoring
DRIFT_COLS = {"ratio": "Amount vs customer's norm", "hour": "Hour of day",
              "dist": "Distance from home", "gaplog": "Time between purchases"}


def stream_features(df_full, warm=12):
    """Monitoring features for every transaction after the warm-up history."""
    norm = df_full[df_full.step < warm].groupby("user").amount.mean()
    cur = df_full[df_full.step >= warm].copy()
    cur["ratio"] = cur.amount.values / norm.reindex(cur.user).values
    cur["hour"] = (cur.ts % 86400) / 3600
    cur["gaplog"] = np.log1p(cur.dt)
    return cur[list(DRIFT_COLS)]


def reference_features(n_seeds=20):
    """What 'normal' traffic looked like when the model was built (same demo set-up, many seeds)."""
    return pd.concat([stream_features(generate(12, 40, 0.5, seed=s)) for s in range(1000, 1000 + n_seeds)],
                     ignore_index=True)


def psi(ref, cur, bins=4):
    """Population Stability Index between a reference and a current sample of one feature.
    < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 significant drift."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3 or len(cur) == 0:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    a = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(v):
    return "🟢 stable" if v < 0.10 else ("🟡 moderate shift" if v < 0.25 else "🔴 significant drift")


def drift_report(ref, cur):
    rows = [{"feature": name, "PSI": round(psi(ref[c], cur[c]), 3)} for c, name in DRIFT_COLS.items()]
    out = pd.DataFrame(rows)
    out["status"] = out["PSI"].map(psi_status)
    return out


# ------------------------------------------- analyst feedback + cost-sensitive threshold
def suggest_threshold(probs, is_fraud, current, target_precision=0.8):
    """From analyst-reviewed alerts, find the lowest threshold (>= current) whose precision among the
    reviewed alerts reaches the target. Returns (threshold, precision, frauds_kept, frauds_total) or None.
    Caveat: only alerts get reviewed, so this can reduce false alarms but cannot find missed frauds."""
    p, f = np.asarray(probs, float), np.asarray(is_fraud, bool)
    total = int(f.sum())
    for t in np.unique(np.concatenate([[current], p[p >= current]])):
        keep = p >= t
        if keep.sum() == 0:
            break
        prec = f[keep].mean()
        if prec >= target_precision and f[keep].sum() > 0:
            return float(t), float(prec), int(f[keep].sum()), total
    return None


def cost_curve(y, p, amount, review_cost, grid=None):
    """Money-based view of the threshold: missed frauds lose their amount, every alert costs review time."""
    # fine steps near 1.0 too: class-weighted models push most probabilities towards 0 or 1
    grid = np.unique(np.concatenate([np.linspace(0.02, 0.98, 49), np.linspace(0.98, 0.999, 22)])) if grid is None else grid
    y, p, amount = np.asarray(y), np.asarray(p), np.asarray(amount, float)
    rows = []
    for t in grid:
        alert = p >= t
        missed = (y == 1) & ~alert
        rows.append((round(float(t), 3), float(amount[missed].sum() + review_cost * alert.sum()),
                     int(alert.sum()), int(missed.sum())))
    return pd.DataFrame(rows, columns=["threshold", "total_cost", "alerts", "missed_frauds"])
