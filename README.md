# 🛡️ TransFraud

**Real-time financial fraud detection with a Transformer that reads each customer's recent transaction history.**

TransFraud scores card transactions one by one. Instead of judging a transaction in isolation, a small Transformer
looks at the customer's last 10 transactions, so "₹25,000 at 2 AM, 1,200 km from home" can be flagged as unusual
*for this customer*. A responsive Streamlit dashboard wraps the model with explanations, drift monitoring, analyst
feedback and cost analysis.

> **Note:** the main experiments use **synthetic data** and a **simulated** real-time stream. The approach is also
> checked on the public Kaggle credit-card fraud dataset. This is a college demo project, not a production system.

## Features
- **Transformer model** over a customer's last 10 transactions (amount, hour, distance from home, time gap, merchant category)
- **Live stream demo** – transactions replayed in time order, scored with latency shown, risk levels 🔴 High / 🟡 Medium / 🟢 Low
- **Explainable alerts** – "why was this flagged?" what-if breakdown, plus an attention chart over past transactions
- **Five fraud types** in the synthetic data: big spend, card testing, impossible travel, account takeover, high-risk merchant cash-out
- **Batch CSV upload** with per-customer history and downloadable results
- **Drift monitor** (PSI) with a "simulate behaviour drift" switch
- **Analyst review** – confirm/reject alerts, save feedback, get a suggested threshold
- **Cost-sensitive threshold** – picks the cut-off that minimises money lost, not just F1
- **Model comparison** – Transformer, Transformer + focal loss, LSTM, Random Forest, Gradient Boosting (confusion matrix, ROC/PR curves)
- **Real-data validation** on the Kaggle credit-card fraud dataset (time-ordered split)
- **Federated learning simulation** (FedAvg across 3 banks)
- **REST API** (`POST /score`) and Docker files

## Quick start
```bash
pip install -r requirements.txt
python train.py             # trains the models, saves artifacts/ (a few minutes on CPU)
streamlit run app.py        # dashboard
```
Optional:
```bash
python train_real.py        # needs creditcard.csv from Kaggle ("Credit Card Fraud Detection") in this folder
python federated.py         # federated-learning experiment
uvicorn api:app --reload    # REST API docs at http://127.0.0.1:8000/docs
docker compose up --build   # dashboard :8501 and API :8000
```

## Project structure
| File | Purpose |
|---|---|
| `data.py` | Synthetic customer/transaction generator and sequence builder |
| `model.py` | TransFraud Transformer (with attention extraction) and LSTM baseline |
| `train.py` | Trains and compares 5 models; saves metrics, confusion matrices and curves |
| `explain.py` | What-if explanation of a flagged transaction |
| `scoring.py`, `utils.py` | Scoring helpers, risk levels, batch CSV, drift (PSI), feedback and cost logic |
| `app.py`, `style.py` | Streamlit dashboard and its responsive styling |
| `api.py` | FastAPI endpoint |
| `realdata.py`, `train_real.py` | Validation on the Kaggle dataset |
| `federated.py` | Federated learning simulation |

## Results
_Add your numbers and screenshots here after running `train.py`, `train_real.py` and `federated.py`._

| Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| TransFraud (Transformer) | 0.866 | 0.89 | 0.752 | 0.815 |
| LSTM | 0.869 |	0.909 |	0.743 |	0.817 |
| Random Forest (no history) |	0.875 |	0.953 | 	0.762 |	0.847 |
| Gradient Boosting (no history) |	0.861 | 	0.913 |	0.734 |	0.813 |

## Limitations
- The main data is synthetic, so strong scores show the model can learn the designed patterns, not that it would work on real bank data.
- "Real-time" is a replayed stream with per-transaction scoring, not a Kafka-style production pipeline.
- The Kaggle dataset has no customer IDs, so there the model reads the last 10 transactions overall, not one customer's history.
- Explanations are a simple what-if method, not SHAP. Analyst feedback only covers alerts, so it can cut false alarms but cannot find missed frauds.

## Future scope
Streaming with Kafka, graph-based fraud-ring detection, SHAP explanations, retraining from analyst feedback, production deployment and monitoring.

## Author
_Your name · College · Year_
