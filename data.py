"""Synthetic card-transaction generator + sequence builder."""
import numpy as np
import pandas as pd

CATS = ["grocery", "fuel", "restaurant", "electronics", "travel", "online", "atm", "jewelry",
        "gift_card", "crypto", "money_transfer"]   # last three = high-risk merchant categories
RISKY = [8, 9, 10]
KINDS = ["big", "test", "travel", "takeover", "risky"]
PAD = len(CATS)          # padding token id for the categorical embedding
SEQ_LEN = 10             # the model looks at the last 10 transactions of a user
N_NUM = 5                # numeric features per transaction


def encode(amount, hour, dist, dt):
    """Turn raw values into the model's numeric features (works on scalars or arrays)."""
    amount, hour, dist, dt = map(np.asarray, (amount, hour, dist, dt))
    return np.stack([
        np.log1p(amount) / 10,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.log1p(dist) / 8,
        np.log1p(dt) / 12,
    ], axis=-1).astype(np.float32)


def _snap_to_habit(ts_next, h0, rng, width=6.0):
    """Normal customers shop around their own habitual hour h0. If the next purchase would fall
    outside their window (h0 +/- width hours), move it to the next time inside the window."""
    h = (ts_next % 86400) / 3600
    if abs((h - h0 + 12) % 24 - 12) <= width:
        return ts_next
    target = (ts_next // 86400) * 86400 + ((h0 + rng.normal(0, 2.5)) % 24) * 3600
    if target < ts_next:
        target += 86400
    return target


def _to_night(prev, nxt, rng):
    """Move a timestamp to the small hours (00:00-05:00) of its day (or the next day)."""
    nxt = (nxt // 86400) * 86400 + rng.uniform(0, 5) * 3600
    return nxt + 86400 if nxt <= prev else nxt


def generate(n_users=2000, txns_per_user=60, compromised_rate=0.2, seed=42):
    """Each customer has habits (typical amount, favourite categories, home area, habitual shopping hour).
    Compromised customers get a burst of ONE of five fraud types:
      big       : very large amount, far away, often at night
      test      : card testing - many tiny online purchases in rapid succession
      travel    : impossible travel - far from home only minutes after a normal purchase
      takeover  : account takeover - amounts escalate step by step, online/odd categories, odd hours
      risky     : cash-out at a high-risk merchant category (gift cards, crypto, money transfer)
    Legit noise is added (unusual-but-genuine purchases) so the task isn't trivial."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        fav = rng.choice([0, 1, 2, 6], 3, replace=False)
        avg = rng.uniform(300, 3000)
        h0 = rng.uniform(9, 20)                       # habitual hour of the day
        ts = rng.integers(0, 3) * 86400 + h0 * 3600
        bad = rng.random() < compromised_rate
        start = rng.integers(15, txns_per_user - 6) if bad else -1
        kind = rng.choice(KINDS) if bad else None
        blen = rng.integers(2, 6)
        for i in range(txns_per_user):
            fraud = bad and start <= i < start + blen
            prev = ts
            if fraud:
                j = i - start
                if kind == "big":
                    amount = avg * rng.uniform(4, 15)
                    cat = rng.choice([3, 4, 5, 7])
                    dist = rng.uniform(300, 3000) if rng.random() < 0.8 else rng.exponential(8)
                    nxt = ts + rng.uniform(60, 1800)
                    ts = _to_night(ts, nxt, rng) if rng.random() < 0.6 else nxt
                elif kind == "test":
                    amount = rng.uniform(10, 150)
                    cat = 5
                    dist = rng.uniform(0, 2000)
                    ts += rng.uniform(10, 120)
                elif kind == "travel":
                    amount = avg * rng.uniform(0.7, 2.5)       # looks like ordinary spending...
                    cat = rng.choice([*fav[:2], 5])
                    dist = rng.uniform(800, 5000)               # ...but thousands of km away
                    ts += rng.uniform(300, 2400)                # ...minutes after the last purchase
                elif kind == "takeover":
                    amount = avg * (1.3 + 0.9 * j) * rng.uniform(0.9, 1.2)   # escalating
                    cat = rng.choice([5, 3, 7, 8])
                    dist = rng.exponential(8) if rng.random() < 0.6 else rng.uniform(100, 1500)
                    nxt = ts + rng.uniform(600, 5400)
                    ts = _to_night(ts, nxt, rng) if rng.random() < 0.5 else nxt
                else:  # risky merchant category
                    amount = avg * rng.uniform(1.5, 6)
                    cat = rng.choice(RISKY)
                    dist = rng.exponential(10) if rng.random() < 0.7 else rng.uniform(100, 2000)
                    ts += rng.uniform(300, 3600)
                # camouflage: fraudsters sometimes imitate the customer, so each signal can be missing
                if rng.random() < 0.30: amount = avg * rng.lognormal(0, 0.45)
                if rng.random() < 0.25: dist = rng.exponential(6)
                if rng.random() < 0.25: cat = rng.choice(fav)
                if rng.random() < 0.20: ts = _snap_to_habit(prev + rng.exponential(5 * 3600) + 60, h0, rng)
            else:
                amount = avg * rng.lognormal(0, 0.45)
                cat = rng.choice(fav)
                dist = rng.exponential(6)
                if rng.random() < 0.05:               # legit travel / big purchase
                    amount *= rng.uniform(3, 8); cat = rng.choice([3, 4]); dist = rng.uniform(100, 800)
                if rng.random() < 0.015:              # legit small purchase at a high-risk category
                    cat = rng.choice(RISKY); amount = avg * rng.uniform(0.3, 1.2)
                r = rng.random()
                if r < 0.04:                          # legit quick successive purchases
                    ts += rng.uniform(120, 1800)
                elif r < 0.06:                        # legit late-night purchase
                    ts = _to_night(ts, ts + rng.exponential(5 * 3600) + 60, rng)
                else:
                    ts = _snap_to_habit(ts + rng.exponential(5 * 3600) + 60, h0, rng)
            rows.append((u, i, ts, round(float(amount), 2), int(cat), float(dist), float(ts - prev),
                         int(fraud), kind if fraud else ""))
    return pd.DataFrame(rows, columns=["user", "step", "ts", "amount", "cat", "dist", "dt", "label", "kind"])


def make_sequences(df, L=SEQ_LEN):
    """For every transaction build the window of that user's last L transactions
    (left-padded). Target = is the LAST transaction in the window fraud?"""
    Xn, Xc, Y, U = [], [], [], []
    for u, g in df.groupby("user", sort=False):
        num = encode(g.amount.values, (g.ts.values % 86400) / 3600, g.dist.values, g.dt.values)
        cat = g.cat.values
        T = len(g)
        pn = np.vstack([np.zeros((L - 1, N_NUM), np.float32), num])
        pc = np.concatenate([np.full(L - 1, PAD), cat])
        idx = np.arange(T)[:, None] + np.arange(L)[None, :]
        Xn.append(pn[idx]); Xc.append(pc[idx]); Y.append(g.label.values); U.append(np.full(T, u))
    return (np.concatenate(Xn), np.concatenate(Xc).astype(np.int64),
            np.concatenate(Y).astype(np.float32), np.concatenate(U))
