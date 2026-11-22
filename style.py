"""Look & feel + responsive layout for the Streamlit dashboard.
Works in light and dark mode (uses translucent greys instead of fixed backgrounds)."""
import streamlit as st

CSS = """
<style>
:root { --tf-radius: 14px; --tf-border: rgba(128,128,128,.28); --tf-card: rgba(128,128,128,.08); }

/* page frame */
.block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1280px; }   /* clears the top bar */
footer { visibility: hidden; }   /* the top bar (sidebar arrow, Deploy, 3-dot menu) stays visible */

/* hero banner */
.tf-hero { background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#0ea5e9 100%); color:#fff;
  border-radius:18px; padding: clamp(14px,3vw,26px) clamp(16px,3.5vw,32px); margin-bottom:1rem; }
.tf-title { font-size: clamp(1.5rem,4.5vw,2.3rem); font-weight:800; line-height:1.15; color:#fff; }
.tf-sub { margin:.35rem 0 0; opacity:.93; font-size: clamp(.85rem,2.3vw,1.05rem); color:#fff; }
.tf-badges { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.8rem; }
.tf-badge { background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35); color:#fff;
  border-radius:999px; padding:.15rem .7rem; font-size:.78rem; }

/* cards, tables, alerts, buttons */
[data-testid="stMetric"] { background:var(--tf-card); border:1px solid var(--tf-border);
  border-radius:var(--tf-radius); padding:.7rem .9rem; }
[data-testid="stMetricLabel"] { font-size:.8rem; opacity:.85; }
[data-testid="stMetricValue"] { font-size: clamp(1.15rem,2.6vw,1.7rem); }
[data-testid="stDataFrame"] { border:1px solid var(--tf-border); border-radius:12px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:var(--tf-radius); }
.stButton > button, .stDownloadButton > button { border-radius:10px; min-height:2.6rem; font-weight:600; }

/* tabs scroll sideways instead of wrapping */
[data-baseweb="tab-list"] { gap:.25rem; overflow-x:auto; flex-wrap:nowrap; scrollbar-width:thin; }
[data-baseweb="tab"] { white-space:nowrap; padding:.5rem .8rem; }

/* tablets: rows of metrics wrap, 3 per row */
@media (max-width: 1100px) {
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:.75rem; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="column"],
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
    flex:1 1 calc(33.333% - 1rem); min-width:calc(33.333% - 1rem); }
}

/* phones: forms stack to one column, metric cards 2 per row, full-width buttons */
@media (max-width: 640px) {
  .block-container { padding-left:.8rem; padding-right:.8rem; padding-top:4rem; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"],
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100%; min-width:100%; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="column"],
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
    flex:1 1 calc(50% - .75rem); min-width:calc(50% - .75rem); }
  .stButton > button, .stDownloadButton > button { width:100%; }
  [data-baseweb="tab"] { padding:.45rem .6rem; font-size:.85rem; }
}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, badges=()):
    # one line per tag and no blank lines/indentation, so Markdown doesn't treat it as a code block
    chips = "".join(f'<span class="tf-badge">{b}</span>' for b in badges)
    st.markdown(f'<div class="tf-hero"><div class="tf-title">🛡️ {title}</div>'
                f'<p class="tf-sub">{subtitle}</p><div class="tf-badges">{chips}</div></div>',
                unsafe_allow_html=True)
