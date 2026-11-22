"""REST API for TransFraud.   Run:  uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser."""
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data import CATS, SEQ_LEN, encode
from explain import explain
from scoring import load_model, score
from utils import risk_level

app = FastAPI(title="TransFraud API", version="1.0")
model, meta = load_model()
THRESHOLD = float(meta["threshold"])


class Txn(BaseModel):
    amount: float
    hour: float                 # 0-24
    distance_km: float
    minutes_since_last: float   # gap since the customer's previous transaction
    category: str               # one of: grocery, fuel, restaurant, electronics, travel, online, atm, jewelry


class ScoreRequest(BaseModel):
    history: List[Txn] = []     # customer's previous transactions, oldest first (last 9 are used)
    transaction: Txn            # the new transaction to score


def to_item(t: Txn):
    cat = t.category.strip().lower()
    if cat not in CATS:
        raise HTTPException(400, f"Unknown category '{t.category}'. Use one of: {', '.join(CATS)}")
    return encode(t.amount, t.hour, t.distance_km, max(t.minutes_since_last, 1) * 60), CATS.index(cat)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}


@app.post("/score")
def score_transaction(req: ScoreRequest):
    buf = [to_item(t) for t in req.history[-(SEQ_LEN - 1):]] + [to_item(req.transaction)]
    p, contrib = explain(lambda b: score(model, b), buf)
    pos = {k: max(v, 0.0) for k, v in contrib.items()}
    total = sum(pos.values())
    return {
        "fraud_probability": round(p, 4),
        "risk": risk_level(p, THRESHOLD).split(" ", 1)[1],
        "is_fraud": bool(p >= THRESHOLD),
        "reason_share_percent": {k: round(v / total * 100, 1) if total > 0 else 0.0 for k, v in pos.items()},
    }
