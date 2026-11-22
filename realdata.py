"""Loader for the public Kaggle 'Credit Card Fraud Detection' dataset (creditcard.csv).
Columns: Time, V1..V28 (anonymised PCA features), Amount, Class (1 = fraud). No customer IDs."""
import numpy as np
import pandas as pd

L = 10                                   # window = the last 10 consecutive transactions
V_COLS = [f"V{i}" for i in range(1, 29)]


def load_creditcard(path="creditcard.csv"):
    df = pd.read_csv(path)
    missing = [c for c in ["Time", "Amount", "Class"] + V_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}. Use the Kaggle 'creditcard.csv' file.")
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    gap = df["Time"].diff().fillna(0).clip(lower=0).values
    X = np.hstack([df[V_COLS].values,
                   (np.log1p(df["Amount"].values) / 5)[:, None],
                   (np.log1p(gap) / 3)[:, None]]).astype(np.float32)
    return X, df["Class"].values.astype(np.float32)


def pad_front(X, L=L):
    """Prepend L-1 zero rows so every transaction has a full-length window."""
    return np.vstack([np.zeros((L - 1, X.shape[1]), np.float32), X])


def windows(Xp, idx, L=L):
    """Windows (len(idx), L, F) of the last L transactions ending at each index in idx.
    Built per batch from the padded array so the full 10x dataset never sits in memory."""
    return Xp[np.asarray(idx)[:, None] + np.arange(L)[None, :]]


def temporal_split(n, train=0.70, val=0.15):
    a, b = int(n * train), int(n * (train + val))
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]
