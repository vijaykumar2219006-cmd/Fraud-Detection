"""Federated learning simulation (FedAvg).

Three 'banks' each hold their own customers. They never share transactions - only model weights.
Each round: every bank trains a copy of the global model on its own data, the server averages the weights
(weighted by data size), and the average becomes the new global model.
Compared with: a model trained by each bank alone, and a model trained on all data pooled together.

Run:  python federated.py      (takes a few minutes on CPU; results appear in the dashboard's Model results tab)"""
import copy, json, os, numpy as np, torch, torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from data import generate, make_sequences
from model import TransFraud

torch.manual_seed(0); np.random.seed(0)
os.makedirs("artifacts", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
N_BANKS, ROUNDS, LOCAL_EPOCHS = 3, 5, 2
TOTAL_EPOCHS = ROUNDS * LOCAL_EPOCHS            # same amount of training for every approach

print("Generating data...")
df = generate(n_users=1800, txns_per_user=60, seed=11)
Xn, Xc, y, users = make_sequences(df)
uids = np.unique(users); np.random.shuffle(uids)
test_users, train_users = uids[:300], uids[300:]
banks = np.array_split(train_users, N_BANKS)


def subset(mask):
    return torch.from_numpy(Xn[mask]), torch.from_numpy(Xc[mask]), torch.from_numpy(y[mask])


bank_data = [subset(np.isin(users, b)) for b in banks]
pooled = tuple(torch.cat(t) for t in zip(*bank_data))
Xte, Xcte, yte = subset(np.isin(users, test_users))
print(f"{N_BANKS} banks with {[len(d[0]) for d in bank_data]} transactions; test set {len(yte)} "
      f"({int(yte.sum())} frauds)")


def local_train(model, data, epochs):
    model.to(dev).train()
    xn, xc, yb = data
    pos = max(float(yb.mean()), 1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min((1 - pos) / pos, 30.0)).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = len(xn)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            loss = crit(model(xn[b].to(dev), xc[b].to(dev)), yb[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model):
    model.to(dev).eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 4096):
            out.append(torch.sigmoid(model(Xte[i:i + 4096].to(dev), Xcte[i:i + 4096].to(dev))).cpu())
    p = torch.cat(out).numpy()
    return float(average_precision_score(yte.numpy(), p)), float(roc_auc_score(yte.numpy(), p))


def fedavg(states, sizes):
    total = float(sum(sizes))
    return {k: sum(s[k].float() * (n / total) for s, n in zip(states, sizes)) for k in states[0]}


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---- federated training
print("\nFederated training (FedAvg) ...")
global_model = TransFraud()
rounds = []
for r in range(1, ROUNDS + 1):
    states, sizes = [], []
    for d in bank_data:
        local = local_train(copy.deepcopy(global_model), d, LOCAL_EPOCHS)
        states.append(cpu_state(local)); sizes.append(len(d[0]))
    global_model.load_state_dict(fedavg(states, sizes))
    pr, auc = evaluate(global_model)
    rounds.append({"round": r, "federated PR-AUC": round(pr, 4)})
    print(f"  round {r}: PR-AUC {pr:.4f}  ROC-AUC {auc:.4f}")
fed_pr, fed_auc = evaluate(global_model)

# ---- each bank training alone
print("\nBanks training alone ...")
local_scores = []
for i, d in enumerate(bank_data, 1):
    m = local_train(TransFraud(), d, TOTAL_EPOCHS)
    local_scores.append(evaluate(m)); print(f"  bank {i}: PR-AUC {local_scores[-1][0]:.4f}")

# ---- centralised (all data pooled - what federated learning tries to match without sharing data)
print("\nCentralised training on pooled data ...")
cen_pr, cen_auc = evaluate(local_train(TransFraud(), pooled, TOTAL_EPOCHS))

results = [
    {"model": f"Each bank alone (average of {N_BANKS})", "pr_auc": float(np.mean([s[0] for s in local_scores])),
     "roc_auc": float(np.mean([s[1] for s in local_scores]))},
    {"model": "Federated (FedAvg, no data shared)", "pr_auc": fed_pr, "roc_auc": fed_auc},
    {"model": "Centralised (all data pooled)", "pr_auc": cen_pr, "roc_auc": cen_auc},
]
print("\nRESULTS (same unseen test customers)")
for r in results:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

note = (f"Simulation with {N_BANKS} banks, {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs. Banks share only model weights, "
        "never transactions. Customers were split randomly between banks (IID); real banks differ more, "
        "which makes federated learning harder.")
json.dump({"results": results, "rounds": rounds, "note": note}, open("artifacts/federated.json", "w"), indent=2)
print("\nSaved artifacts/federated.json")
