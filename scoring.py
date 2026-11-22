"""Shared helpers used by the dashboard (app.py) and the REST API (api.py)."""
import json
import numpy as np
import torch
from data import PAD, SEQ_LEN, N_NUM
from model import TransFraud


def load_model(folder="artifacts"):
    meta = json.load(open(f"{folder}/meta.json"))
    m = TransFraud()
    m.load_state_dict(torch.load(f"{folder}/model.pt", map_location="cpu"))
    m.eval()
    return m, meta


def to_tensors(buf):
    """buf: sequence of (numeric_vector, category_id), oldest first. Left-pads to SEQ_LEN."""
    buf = list(buf)[-SEQ_LEN:]
    n = len(buf)
    xn = np.zeros((1, SEQ_LEN, N_NUM), np.float32)
    xc = np.full((1, SEQ_LEN), PAD, np.int64)
    for i, (v, c) in enumerate(buf):
        xn[0, SEQ_LEN - n + i] = v
        xc[0, SEQ_LEN - n + i] = c
    return torch.from_numpy(xn), torch.from_numpy(xc)


def score(model, buf):
    xn, xc = to_tensors(buf)
    with torch.no_grad():
        return torch.sigmoid(model(xn, xc)).item()


def score_many(model, xn, xc, batch=4096):
    """Fraud probabilities for many windows at once (xn: float32 (N,L,F), xc: int64 (N,L))."""
    out = []
    with torch.no_grad():
        for i in range(0, len(xn), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(xn[i:i + batch]), torch.from_numpy(xc[i:i + batch]))))
    return torch.cat(out).numpy()
