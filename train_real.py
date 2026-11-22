"""Validate the transformer idea on REAL data (Kaggle credit-card fraud, creditcard.csv).

Put creditcard.csv in this folder, then run:   python train_real.py
The data has no customer IDs, so a 'sequence' here is the last 10 consecutive transactions in time order.
We split by TIME (train on the past, test on the future), which is how fraud models are used in practice."""
import json, os, sys, numpy as np, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, precision_recall_fscore_support)
from realdata import load_creditcard, pad_front, windows, temporal_split, L

PATH = sys.argv[1] if len(sys.argv) > 1 else "creditcard.csv"
if not os.path.exists(PATH):
    sys.exit(f"Cannot find {PATH}. Download 'creditcard.csv' from Kaggle (Credit Card Fraud Detection, ULB) "
             "and put it in this folder, or run: python train_real.py path/to/creditcard.csv")

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
X, y = load_creditcard(PATH)
Xp = pad_front(X)
tr, va, te = temporal_split(len(y))
print(f"{len(y):,} transactions, {int(y.sum())} frauds ({y.mean():.3%}). "
      f"Train {len(tr):,} | Val {len(va):,} ({int(y[va].sum())} frauds) | Test {len(te):,} ({int(y[te].sum())} frauds)")


class SeqTransformer(nn.Module):
    def __init__(self, n_feat, d=64, heads=4, layers=2, L=L):
        super().__init__()
        self.proj = nn.Linear(n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.encoder(self.proj(x) + self.pos)[:, -1]).squeeze(-1)


def predict(model, idx):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            xb = torch.from_numpy(windows(Xp, idx[i:i + 4096])).to(dev)
            out.append(torch.sigmoid(model(xb)).cpu())
    return torch.cat(out).numpy()


def best_threshold(yv, pv):
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    return float(th[np.argmax(f1[:-1])])


def report(name, p, yt, thr):
    P, R, F, _ = precision_recall_fscore_support(yt, p >= thr, average="binary", zero_division=0)
    return dict(model=name, roc_auc=float(roc_auc_score(yt, p)), pr_auc=float(average_precision_score(yt, p)),
                precision=float(P), recall=float(R), f1=float(F))


print("\nTraining transformer on windows of 10 consecutive transactions ...")
model = SeqTransformer(X.shape[1]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
pos_w = torch.tensor(float(min((1 - y[tr].mean()) / y[tr].mean(), 100.0))).to(dev)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
best_pr, best_state = -1, None
for ep in range(8):
    model.train(); perm = np.random.permutation(tr); total = 0
    for i in range(0, len(perm), 512):
        b = perm[i:i + 512]
        xb = torch.from_numpy(windows(Xp, b)).to(dev); yb = torch.from_numpy(y[b]).to(dev)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(b)
    pr_auc = average_precision_score(y[va], predict(model, va))
    print(f"  epoch {ep + 1}  loss {total / len(perm):.4f}  val PR-AUC {pr_auc:.4f}")
    if pr_auc > best_pr:
        best_pr, best_state = pr_auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)          # keep the best epoch (chosen on validation, not test)

print("\nTraining gradient-boosting baseline (current transaction only) ...")
gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(X[tr], y[tr])

thr_t = best_threshold(y[va], predict(model, va))
thr_g = best_threshold(y[va], gb.predict_proba(X[va])[:, 1])
results = [report("TransFraud transformer (last 10 txns)", predict(model, te), y[te], thr_t),
           report("Gradient Boosting (current txn only)", gb.predict_proba(X[te])[:, 1], y[te], thr_g)]

print("\nTEST RESULTS (future, unseen data)")
for r in results:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

os.makedirs("artifacts_real", exist_ok=True)
note = (f"Kaggle credit-card fraud dataset ({len(y):,} transactions, {int(y.sum())} frauds). Time-ordered split: "
        f"train on the first 70%, validate on the next 15%, test on the last 15% ({int(y[te].sum())} frauds). "
        "The data has no customer IDs, so the transformer sees the last 10 consecutive transactions overall, "
        "not one customer's history. With so few test frauds, differences between models can be noise.")
json.dump({"results": results, "note": note}, open("artifacts_real/results.json", "w"), indent=2)
print("\nSaved artifacts_real/results.json (it will appear in the dashboard's Model results tab).")
