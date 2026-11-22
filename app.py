import json, math, os, time
from collections import deque
import numpy as np, pandas as pd, streamlit as st
from data import generate, encode, make_sequences, CATS, SEQ_LEN
from scoring import load_model, score, score_many, to_tensors
from explain import explain
from style import apply_style, hero
from utils import (risk_level, sample_csv, prepare_batch, score_batch, reference_features, drift_report,
                   suggest_threshold, cost_curve)

st.set_page_config(page_title="TransFraud", page_icon="🛡️", layout="wide")


@st.cache_resource
def load():
    return load_model()


def explain_buf(buf):
    return explain(lambda x: score(model, x), buf)


@st.cache_data(show_spinner="Building drift reference...")
def get_reference():
    return reference_features()


@st.cache_data(show_spinner="Scoring a fresh validation set (one-time)...")
def validation_scores():
    """Fresh synthetic customers (never used for training) scored by the model, for the cost analysis."""
    df = generate(n_users=400, seed=777)
    xn, xc, y, _ = make_sequences(df)
    return y, score_many(model, xn, xc), df.amount.values


def apply_thr(t):
    st.session_state["thr"] = float(min(max(math.ceil(t * 100) / 100, 0.05), 0.95))


model, meta = load()
apply_style()
hero("TransFraud", "Real-time transformer-based financial fraud detection",
     ["Transformer", "Real-time scoring", "Explainable alerts", "Drift monitoring"])

st.sidebar.header("Settings")
thr = st.sidebar.slider("Alert threshold (High risk)", 0.05, 0.95, float(meta["threshold"]), 0.01, key="thr")
st.sidebar.caption(f"🔴 High ≥ {thr:.2f}  |  🟡 Medium ≥ {thr * 0.5:.2f}  |  🟢 Low below that")
delay = st.sidebar.slider("Delay between transactions (s)", 0.0, 3.0, 0.5, 0.1)
n_events = st.sidebar.slider("Transactions to stream", 50, 336, 100, 50)
drift_on = st.sidebar.checkbox("Simulate behaviour drift (festive season)", value=False,
                               help="Amounts x1.8, distances x2 and later shopping hours, to see the drift monitor react.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📡 Live stream", "🔍 Check a transaction", "📁 Batch upload",
                                        "🧑‍💼 Analyst review", "📊 Model results"])

# ---------------------------------------------------------------- Live stream
with tab1:
    if st.button("▶ Start stream", type="primary"):
        # 12 customers, about half compromised. Each customer's first 12 transactions are the
        # history; the rest are replayed in time order as a live stream.
        full = generate(n_users=12, txns_per_user=40, compromised_rate=0.5, seed=85)
        norm = full[full.step < 12].groupby("user").amount.mean().to_dict()   # each customer's usual amount
        ref = get_reference()
        bufs = {}
        for u, g in full.groupby("user"):
            bufs[u] = deque(maxlen=SEQ_LEN)
            for r in g[g.step < 12].itertuples():
                bufs[u].append((encode(r.amount, (r.ts % 86400) / 3600, r.dist, r.dt), r.cat))
        stream = full[full.step >= 12].sort_values("ts").head(n_events)

        DRIFT_MIN = 100                       # events needed before the drift monitor reports
        rows, probs, lat, live = [], [], [], []
        high = caught = missed = 0
        ph = [c.empty() for c in st.columns(6)]
        chart = st.empty()
        table = st.empty()
        drift_ph = st.empty()

        for r in stream.itertuples():
            amt, dst, hr = r.amount, r.dist, (r.ts % 86400) / 3600
            if drift_on:                      # simulated change in customer behaviour
                amt, dst, hr = amt * 1.8, dst * 2.0, (hr + 3) % 24
            t0 = time.perf_counter()
            bufs[r.user].append((encode(amt, hr, dst, r.dt), r.cat))
            p = score(model, bufs[r.user])
            lat.append((time.perf_counter() - t0) * 1000)
            flag = p >= thr
            reason = ""
            if flag:   # explain only the alerts
                contrib = explain_buf(bufs[r.user])[1]
                reason = max(contrib, key=contrib.get)
            high += int(flag)
            caught += int(flag and r.label == 1)
            missed += int((not flag) and r.label == 1)
            probs.append(p)
            rows.append({"id": len(rows) + 1, "user": f"U{r.user:03d}", "hour": round(hr, 1),
                         "amount": round(amt, 2), "category": CATS[r.cat], "dist_km": round(dst, 1),
                         "fraud_prob": round(p, 3), "risk": risk_level(p, thr), "main_reason": reason,
                         "actual": f"fraud ({r.kind})" if r.label else "legit"})
            live.append({"ratio": amt / norm[r.user], "hour": hr, "dist": dst, "gaplog": float(np.log1p(r.dt))})
            ph[0].metric("Processed", len(probs))
            ph[1].metric("High-risk alerts", high)
            ph[2].metric("Frauds caught", caught)
            ph[3].metric("Frauds missed", missed)
            ph[4].metric("False alarms", high - caught)
            ph[5].metric("Avg latency", f"{np.mean(lat):.1f} ms")
            chart.line_chart(pd.Series(probs[-100:], name="fraud probability"), height=180)
            show = ["user", "amount", "category", "dist_km", "fraud_prob", "risk", "main_reason", "actual"]
            table.dataframe(pd.DataFrame(rows[::-1][:12])[show], hide_index=True)

            n_live = len(live)
            if n_live < DRIFT_MIN:
                drift_ph.caption(f"📉 Drift monitor: collecting data ({n_live}/{DRIFT_MIN} events)...")
            elif n_live % 10 == 0 or n_live == len(stream):
                rep = drift_report(ref, pd.DataFrame(live))
                with drift_ph.container():
                    st.markdown("**📉 Drift monitor** (PSI: <0.10 stable, 0.10-0.25 moderate, >0.25 significant)")
                    st.dataframe(rep, hide_index=True)
                    if (rep["PSI"] > 0.25).any():
                        st.warning("Data drift detected: customer behaviour has moved away from what the model was "
                                   "trained on, so accuracy may fall. Recommended action: retrain on recent data.")
                    else:
                        st.success("No significant drift: incoming traffic looks like the training data.")
            time.sleep(delay)
        st.session_state["stream_rows"] = rows
        st.session_state.pop("review_df", None)
        st.session_state.pop("review_editor", None)
        st.success("Stream finished.")
    else:
        st.info("Press **Start stream** to replay simulated card transactions and watch the model score each one.")

    saved = st.session_state.get("stream_rows")
    if saved:
        sdf = pd.DataFrame(saved)
        alerts = sdf[sdf["risk"].str.contains("High")]
        st.write(f"**Last run:** {len(sdf)} transactions scored, {len(alerts)} flagged as high risk.")
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Download alerts (CSV)", alerts.to_csv(index=False).encode("utf-8"),
                           "transfraud_alerts.csv", "text/csv")
        c2.download_button("⬇ Download all scored transactions (CSV)", sdf.to_csv(index=False).encode("utf-8"),
                           "transfraud_all_transactions.csv", "text/csv")

# ------------------------------------------------------- Check a transaction
with tab2:
    st.write("The model compares your transaction with a simulated history of this customer's normal behaviour.")
    PRESETS = {
        "— enter my own values —": None,
        "Normal grocery purchase": dict(typical=1000, amount=900, cat="grocery", hour=14, dist=3, mins=300),
        "Big spend far away at night": dict(typical=1000, amount=25000, cat="electronics", hour=2, dist=1200, mins=10),
        "Impossible travel (2,500 km in 20 min)": dict(typical=1000, amount=1500, cat="restaurant", hour=15, dist=2500, mins=20),
        "Suspicious online purchase, odd hour": dict(typical=1000, amount=4500, cat="online", hour=3, dist=5, mins=25),
        "High-risk merchant (gift-card cash-out)": dict(typical=1000, amount=6000, cat="gift_card", hour=13, dist=4, mins=40),
    }
    for k, v in dict(chk_typical=1000, chk_amount=25000, chk_cat="electronics",
                     chk_hour=2, chk_dist=1200, chk_mins=10).items():
        st.session_state.setdefault(k, v)

    def load_preset():
        p = PRESETS[st.session_state["preset"]]
        if p:
            for k, v in p.items():
                st.session_state[f"chk_{k}"] = v

    st.selectbox("Load an example scenario", list(PRESETS), key="preset", on_change=load_preset)
    a_col, b_col = st.columns(2)
    typical = a_col.number_input("Customer's typical purchase (₹)", min_value=100, max_value=50000, key="chk_typical")
    amount = a_col.number_input("This transaction amount (₹)", min_value=1, max_value=500000, key="chk_amount")
    cat = a_col.selectbox("Merchant category", CATS, key="chk_cat")
    hour = b_col.slider("Hour of day", 0, 23, key="chk_hour")
    dist = b_col.number_input("Distance from home (km)", min_value=0, max_value=5000, key="chk_dist")
    mins = b_col.number_input("Minutes since previous transaction", min_value=1, max_value=5000, key="chk_mins")

    if st.button("Check transaction", type="primary"):
        rng = np.random.default_rng(1)
        buf = deque(maxlen=SEQ_LEN)
        for _ in range(SEQ_LEN - 1):
            buf.append((encode(typical * rng.lognormal(0, 0.45), rng.uniform(9, 20),
                               rng.exponential(5), rng.exponential(5 * 3600) + 60),
                        int(rng.choice([0, 1, 2]))))
        buf.append((encode(float(amount), float(hour), float(dist), float(mins) * 60), CATS.index(cat)))

        p, contrib = explain_buf(buf)
        risk = risk_level(p, thr)
        st.metric("Fraud probability", f"{p:.1%}")
        if p >= thr:
            st.error(f"{risk} risk – flagged as FRAUD")
        elif p >= thr * 0.5:
            st.warning(f"{risk} risk – needs manual review")
        else:
            st.success(f"{risk} risk – looks legitimate")

        st.subheader("Why this decision?")
        st.caption("Each feature is swapped with the customer's typical value and the model is re-run. "
                   "The bigger the drop in fraud score, the more that feature contributed.")
        pos = {k: max(v, 0.0) for k, v in contrib.items()}
        total = sum(pos.values())
        share = {k: (v / total * 100 if total > 0 else 0.0) for k, v in pos.items()}
        st.bar_chart(pd.Series({k: round(v, 1) for k, v in share.items()}, name="share of suspicion (%)"))

        details = {
            "Amount": f"₹{amount:,} is {amount / typical:.1f}× the customer's typical ₹{typical:,}",
            "Hour of day": f"purchase made at {hour:02d}:00 (this customer normally shops 09:00-20:00)",
            "Distance from home": f"{dist:,} km from the customer's home area (normally within a few km)",
            "Time since last txn": f"only {mins} min after the previous transaction (normally several hours)",
            "Merchant category": f"'{cat}' is not one of the customer's usual categories "
                                 f"({', '.join(CATS[i] for i in (0, 1, 2))})",
        }
        top = [k for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v >= 0.3]
        if top:
            st.write("**Main reasons:**")
            for k in top:
                st.write(f"- **{k}** ({share[k]:.0f}%) – {details[k]}")
        else:
            st.write("No single feature stood out as unusual, so this looks like normal behaviour.")

        try:   # attention view: which past transactions did the transformer look at?
            xn, xc = to_tensors(buf)
            att = model.attention_to_last(xn, xc)[0].numpy()
            names = [f"{i + 1:02d} (t-{SEQ_LEN - 1 - i})" for i in range(SEQ_LEN - 1)] + [f"{SEQ_LEN} (new)"]
            st.subheader("What the transformer looked at")
            st.caption("Attention of the new transaction over the customer's last 10 transactions "
                       "(taller bar = that past transaction mattered more for the decision).")
            st.bar_chart(pd.Series(att, index=names, name="attention"))
        except Exception as e:
            st.caption(f"(Attention view unavailable: {e})")

# --------------------------------------------------------------- Batch upload
with tab3:
    st.write("Upload a CSV of transactions and the model will score every row, keeping a separate history "
             "for each customer.")
    st.markdown("**Required columns:** `user`, `timestamp`, `amount`, `category`, `distance_km`  \n"
                f"**Categories:** {', '.join(CATS)}")
    st.download_button("⬇ Download a sample CSV", sample_csv().to_csv(index=False).encode("utf-8"),
                       "sample_transactions.csv", "text/csv")
    up = st.file_uploader("Upload transactions CSV", type=["csv"])
    if up is not None:
        d = None
        try:
            d, warns = prepare_batch(pd.read_csv(up))
        except Exception as e:
            st.error(f"Could not read the file: {e}")
        if d is not None:
            for w in warns:
                st.warning(w)
            if len(d) > 3000:
                st.warning("Only the first 3,000 rows (by time) are scored.")
                d = d.head(3000)
            with st.spinner("Scoring transactions..."):
                res = score_batch(d, lambda x: score(model, x), thr, lambda x: explain_buf(x)[1])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Transactions", len(res))
            m2.metric("Customers", res["user"].nunique())
            m3.metric("🔴 High risk", int(res["risk"].str.contains("High").sum()))
            m4.metric("🟡 Medium risk", int(res["risk"].str.contains("Medium").sum()))
            res = res.sort_values("fraud_prob", ascending=False).reset_index(drop=True)
            st.dataframe(res, hide_index=True)
            st.download_button("⬇ Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                               "transfraud_batch_results.csv", "text/csv")
            st.caption("A customer's first transactions have little history, so scores for them are less reliable.")

# -------------------------------------------------------------- Analyst review
with tab4:
    st.write("**Human-in-the-loop:** an analyst reviews the model's high-risk alerts. Their verdicts are saved as "
             "feedback (future training labels) and are used here to re-tune the alert threshold.")
    saved_rows = st.session_state.get("stream_rows")
    if not saved_rows:
        st.info("Run the live stream first - its high-risk alerts will appear here for review.")
    else:
        VERDICTS = ["unreviewed", "confirmed fraud", "false alarm"]
        sdf = pd.DataFrame(saved_rows)
        if "review_df" not in st.session_state:
            al = sdf[sdf["risk"].str.contains("High")][["id", "user", "amount", "category", "dist_km",
                                                        "fraud_prob", "main_reason"]].copy()
            al["analyst_verdict"] = "unreviewed"
            st.session_state["review_df"] = al.reset_index(drop=True)
        if st.button("🤖 Simulate an analyst reviewing all alerts (demo shortcut)"):
            truth = dict(zip(sdf["id"], sdf["actual"]))
            rv = st.session_state["review_df"].copy()
            rv["analyst_verdict"] = ["confirmed fraud" if truth[i].startswith("fraud") else "false alarm"
                                     for i in rv["id"]]
            st.session_state["review_df"] = rv
            st.session_state.pop("review_editor", None)

        if len(st.session_state["review_df"]) == 0:
            st.info("The last run produced no high-risk alerts.")
        else:
            cols = list(st.session_state["review_df"].columns)
            edited = st.data_editor(
                st.session_state["review_df"], key="review_editor", hide_index=True,
                disabled=[c for c in cols if c != "analyst_verdict"],
                column_config={"analyst_verdict": st.column_config.SelectboxColumn(
                    "analyst verdict", options=VERDICTS, required=True)})
            rev = edited[edited["analyst_verdict"] != "unreviewed"]
            if len(rev) == 0:
                st.caption("Set a verdict for each alert (or use the demo shortcut above).")
            else:
                is_fraud = (rev["analyst_verdict"] == "confirmed fraud").values
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Alerts reviewed", len(rev))
                m2.metric("Confirmed fraud", int(is_fraud.sum()))
                m3.metric("False alarms", int((~is_fraud).sum()))
                m4.metric("Alert precision", f"{is_fraud.mean():.0%}")
                target = st.slider("Target precision for alerts", 0.5, 1.0, 0.8, 0.05)
                sug = suggest_threshold(rev["fraud_prob"].values, is_fraud, thr, target)
                if sug is None:
                    st.info("No higher threshold reaches that precision with the reviewed alerts "
                            "(or no confirmed frauds would remain).")
                else:
                    t_new, prec, kept, tot = sug
                    st.write(f"Suggested threshold: **{t_new:.2f}** (now {thr:.2f}). Among the reviewed alerts, "
                             f"precision would be {prec:.0%} while keeping {kept} of {tot} confirmed frauds.")
                    st.button("Apply suggested threshold", on_click=apply_thr, args=(t_new,))
                st.caption("Caveat: only alerts get reviewed, so feedback can cut false alarms but cannot reveal "
                           "frauds the model missed.")
                if st.button("💾 Save reviewed alerts as feedback"):
                    os.makedirs("feedback", exist_ok=True)
                    path = "feedback/feedback_log.csv"
                    out = rev.copy()
                    out.insert(0, "saved_at", pd.Timestamp.now().isoformat(timespec="seconds"))
                    out.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                    st.success(f"Saved {len(out)} reviewed alert(s) to {path}.")
            st.download_button("⬇ Download review table (CSV)", edited.to_csv(index=False).encode("utf-8"),
                               "analyst_review.csv", "text/csv")

# -------------------------------------------------------------- Model results
with tab5:
    st.write(f"Test-set results on customers the model never saw. Alert threshold chosen on validation data: "
             f"**{meta['threshold']:.3f}**")
    st.dataframe(pd.DataFrame(meta["results"]).round(3), hide_index=True)
    info = meta.get("data")
    if info:
        st.caption(f"Dataset: {info['transactions']:,} synthetic transactions ({info['fraud_rate']:.2%} fraud). "
                   f"Test set: {info['test_transactions']:,} transactions, {info['test_frauds']} frauds. "
                   "Models marked 'no history' only see the current transaction.")

    conf, curves = meta.get("confusion"), meta.get("curves")
    if conf and curves:
        st.subheader("Confusion matrix")
        name = st.selectbox("Model", list(conf.keys()))
        c = conf[name]
        st.dataframe(pd.DataFrame([[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
                                  index=["Actual: legit", "Actual: fraud"],
                                  columns=["Predicted: legit", "Predicted: fraud"]))
        st.caption(f"Caught {c['tp']} of {c['tp'] + c['fn']} frauds, with {c['fp']} false alarms "
                   f"out of {c['tn'] + c['fp']} legitimate transactions.")
        x = pd.Index(curves["grid"])
        g1, g2 = st.columns(2)
        g1.subheader("ROC curve")
        g1.line_chart(pd.DataFrame(curves["roc"], index=x.rename("false positive rate")))
        g2.subheader("Precision-Recall curve")
        g2.line_chart(pd.DataFrame(curves["pr"], index=x.rename("recall")))
        st.caption("ROC: y-axis = fraud caught (true positive rate). PR: y-axis = precision. "
                   "Curves closer to the top-left (ROC) / top-right (PR) are better.")
    else:
        st.info("Run `python train.py` again to generate the confusion matrices and curves.")

    if os.path.exists("artifacts_real/results.json"):
        real = json.load(open("artifacts_real/results.json"))
        st.subheader("Validation on real data")
        st.dataframe(pd.DataFrame(real["results"]).round(3), hide_index=True)
        st.caption(real["note"])

    st.subheader("💰 Cost-sensitive threshold")
    st.write("Instead of maximising F1, pick the threshold that minimises money lost: a missed fraud costs its "
             "amount, and every alert costs analyst review time.")
    review_cost = st.number_input("Cost of manually reviewing one alert (₹)", min_value=1, max_value=10000,
                                  value=100, key="review_cost")
    yv, pv, av = validation_scores()
    cc = cost_curve(yv, pv, av, review_cost)
    best = cc.loc[cc["total_cost"].idxmin()]
    now = cost_curve(yv, pv, av, review_cost, grid=[thr]).iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Cost-optimal threshold", f"{best['threshold']:.2f}")
    k2.metric("Total cost at that threshold", f"₹{best['total_cost']:,.0f}")
    k3.metric(f"Cost at current threshold ({thr:.2f})", f"₹{now['total_cost']:,.0f}")
    st.caption(f"Without any model, all {int(yv.sum())} frauds in this validation set would go through "
               f"(₹{float(av[yv == 1].sum()):,.0f} lost). Costs are on 400 fresh synthetic customers.")
    st.line_chart(cc.set_index("threshold")["total_cost"])

    if os.path.exists("artifacts/federated.json"):
        fed = json.load(open("artifacts/federated.json"))
        st.subheader("🏦 Federated learning experiment")
        st.dataframe(pd.DataFrame(fed["results"]).round(3), hide_index=True)
        st.line_chart(pd.DataFrame(fed["rounds"]).set_index("round"))
        st.caption(fed["note"])
