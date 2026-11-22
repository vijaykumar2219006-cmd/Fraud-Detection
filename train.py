import json, os, numpy as np, torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, precision_recall_fscore_support)
from data import generate, make_sequences, CATS
from model import TransFraud, LSTMBaseline
from utils import curve_pack, confusion

torch.manual_seed(0); np.random.seed(0)
os.makedirs("artifacts", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

print("Generating data...")
df = generate()
Xn, Xc, y, users = make_sequences(df)
print(f"{len(y)} transactions, fraud rate {y.mean():.2%}")

# split by USER so no user's history leaks between train/val/test
uids = np.unique(users); np.random.shuffle(uids)
tr_u, va_u, te_u = np.split(uids, [int(.7 * len(uids)), int(.85 * len(uids))])
m_tr, m_va, m_te = (np.isin(users, s) for s in (tr_u, va_u, te_u))


def T(a, m): return torch.from_numpy(a[m])


tr = (T(Xn, m_tr), T(Xc, m_tr), T(y, m_tr))
pos_w = torch.tensor(float((1 - y[m_tr].mean()) / y[m_tr].mean())).clamp(max=30.0).to(dev)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)


class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples so training focuses on the rare, hard fraud cases."""

    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        a_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


def predict(model, m):
    model.eval(); out = []
    with torch.no_grad():
        xn, xc = T(Xn, m), T(Xc, m)
        for i in range(0, len(xn), 4096):
            out.append(torch.sigmoid(model(xn[i:i+4096].to(dev), xc[i:i+4096].to(dev))).cpu())
    return torch.cat(out).numpy()


def fit(model, name, epochs=8, crit=None):
    crit = crit or loss_fn
    print(f"\nTraining {name} ...")
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = len(tr[0])
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n); total = 0
        for i in range(0, n, 512):
            b = perm[i:i+512]
            xn, xc, yb = (t[b].to(dev) for t in tr)
            loss = crit(model(xn, xc), yb)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(b)
        p = predict(model, m_va)
        print(f"  epoch {ep+1}  loss {total/n:.4f}  val PR-AUC {average_precision_score(y[m_va], p):.4f}")
    return model


def best_threshold(yv, pv):
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    return float(th[np.argmax(f1[:-1])])


def report(name, p, yt, thr):
    P, R, F, _ = precision_recall_fscore_support(yt, p >= thr, average="binary", zero_division=0)
    return dict(model=name, roc_auc=roc_auc_score(yt, p), pr_auc=average_precision_score(yt, p),
                precision=P, recall=R, f1=F)


# ---- 1) TransFraud (transformer)  2) LSTM  -- both see the last 10 transactions
trans = fit(TransFraud(), "TransFraud (transformer)")
lstm = fit(LSTMBaseline(), "LSTM baseline")
focal = fit(TransFraud(), "TransFraud + focal loss", crit=FocalLoss())
thr = best_threshold(y[m_va], predict(trans, m_va))
thr_lstm = best_threshold(y[m_va], predict(lstm, m_va))
thr_focal = best_threshold(y[m_va], predict(focal, m_va))

# ---- 3) Random Forest  4) Gradient Boosting -- see ONLY the current transaction (no history)
def flat(m): return np.hstack([Xn[m][:, -1, :], np.eye(len(CATS) + 1)[Xc[m][:, -1]]])


print("\nTraining Random Forest and Gradient Boosting (no history) ...")
rf = RandomForestClassifier(200, class_weight="balanced", n_jobs=-1, random_state=0).fit(flat(m_tr), y[m_tr])
gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(flat(m_tr), y[m_tr])
thr_rf = best_threshold(y[m_va], rf.predict_proba(flat(m_va))[:, 1])
thr_gb = best_threshold(y[m_va], gb.predict_proba(flat(m_va))[:, 1])

# ---- test-set evaluation
yt = y[m_te]
NAMES = ["TransFraud (transformer)", "TransFraud + focal loss", "LSTM (history)",
         "Random Forest (no history)", "Gradient Boosting (no history)"]
probs = {NAMES[0]: predict(trans, m_te), NAMES[1]: predict(focal, m_te), NAMES[2]: predict(lstm, m_te),
         NAMES[3]: rf.predict_proba(flat(m_te))[:, 1], NAMES[4]: gb.predict_proba(flat(m_te))[:, 1]}
thrs = dict(zip(NAMES, [thr, thr_focal, thr_lstm, thr_rf, thr_gb]))

results = [report(n, probs[n], yt, thrs[n]) for n in NAMES]
print("\nTEST RESULTS")
for r in results:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

meta = {
    "threshold": thr,
    "results": results,
    "confusion": {n: confusion(yt, probs[n], thrs[n]) for n in NAMES},
    "curves": curve_pack(yt, probs),
    "data": {"transactions": int(len(y)), "fraud_rate": float(y.mean()),
             "test_transactions": int(m_te.sum()), "test_frauds": int(yt.sum())},
}
torch.save(trans.cpu().state_dict(), "artifacts/model.pt")
json.dump(meta, open("artifacts/meta.json", "w"), indent=2)
print(f"\nSaved model. Alert threshold = {thr:.3f}")
