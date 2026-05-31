"""
Churn Radar — upload your order CSV, get customer churn scores instantly.
"""
import io
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_PATH = Path("data/processed/churn_model_clean.pkl")

FEATURES = ["frequency", "monetary", "avg_score",
            "avg_freight", "avg_items", "late_delivery_rate"]

COLORS = {
    "High risk":   "#E24B4A",
    "Medium risk": "#EF9F27",
    "Low risk":    "#378ADD",
    "Loyal":       "#1D9E75",
}
SEG_ORDER = ["High risk", "Medium risk", "Low risk", "Loyal"]

HINTS = {
    "customer_id":   ["email", "customer", "id", "user"],
    "order_date":    ["date", "paid", "created", "timestamp"],
    "order_value":   ["total", "value", "amount", "price", "revenue"],
    "review_score":  ["review", "rating", "score", "star"],
    "freight_value": ["freight", "shipping", "delivery"],
    "item_count":    ["item", "quantity", "qty", "count"],
}

NONE_OPT = "(not in file)"

st.set_page_config(page_title="Churn Radar", page_icon="📡", layout="wide")


# ── Cached model load (once per session) ───────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Cached RFM computation (keyed on file bytes + mapping) ─────────────────
@st.cache_data(show_spinner=False)
def compute_rfm(file_bytes: bytes, mapping: tuple) -> pd.DataFrame:
    mapping_dict = dict(mapping)
    df = pd.read_csv(io.BytesIO(file_bytes))

    rename = {v: k for k, v in mapping_dict.items()
              if v and v != NONE_OPT and v in df.columns}
    df = df.rename(columns=rename)

    df["order_date"]  = pd.to_datetime(df["order_date"],  errors="coerce")
    df["order_value"] = pd.to_numeric(df["order_value"],  errors="coerce")

    for col, default in [("review_score", 3.0), ("freight_value", 0.0), ("item_count", 1.0)]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rfm = (
        df.groupby("customer_id", as_index=False)
        .agg(
            frequency=("order_value",    "count"),
            monetary=("order_value",     "sum"),
            avg_score=("review_score",   "mean"),
            avg_freight=("freight_value","mean"),
            avg_items=("item_count",     "mean"),
        )
    )

    rfm["avg_score"]         = rfm["avg_score"].fillna(3.0)
    rfm["avg_freight"]       = rfm["avg_freight"].fillna(0.0)
    rfm["avg_items"]         = rfm["avg_items"].fillna(1.0)
    rfm["late_delivery_rate"] = 0.0

    return rfm


# ── Helpers ────────────────────────────────────────────────────────────────
def best_required(field: str, cols: list) -> str:
    for col in cols:
        for hint in HINTS.get(field, []):
            if hint in col.lower():
                return col
    return cols[0]


def best_optional(field: str, cols: list) -> str:
    for col in cols:
        for hint in HINTS.get(field, []):
            if hint in col.lower():
                return col
    return NONE_OPT


def assign_segments(probs: pd.Series, threshold: float) -> pd.Series:
    low_floor = threshold * 0.50
    def seg(p):
        if p >= 0.80:          return "High risk"
        elif p >= threshold:   return "Medium risk"
        elif p >= low_floor:   return "Low risk"
        else:                  return "Loyal"
    return probs.apply(seg)


def render_charts(scored: pd.DataFrame) -> None:
    counts = scored["segment"].value_counts().reindex(SEG_ORDER).fillna(0).astype(int)
    rev    = scored.groupby("segment")["revenue_at_risk"].sum().reindex(SEG_ORDER).fillna(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("none")
    ax1.set_facecolor("none")
    ax2.set_facecolor("none")

    # Left — customer count
    ax1.bar(counts.index, counts.values,
            color=[COLORS[s] for s in counts.index], edgecolor="none")
    ax1.set_title("Customers by segment", fontsize=12, color="#555555")
    ax1.tick_params(colors="#777777", labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color("#DDDDDD")
    max_c = counts.max() if counts.max() > 0 else 1
    for i, v in enumerate(counts.values):
        if v > 0:
            ax1.text(i, v + max_c * 0.02, f"{v:,}",
                     ha="center", va="bottom", fontsize=9, color="#555555")

    # Right — revenue at risk
    ax2.barh(rev.index, rev.values,
             color=[COLORS[s] for s in rev.index], edgecolor="none")
    ax2.set_title("Revenue at risk by segment", fontsize=12, color="#555555")
    ax2.tick_params(colors="#777777", labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#DDDDDD")
    max_r = rev.max() if rev.max() > 0 else 1
    for i, (_, v) in enumerate(rev.items()):
        if v > 0:
            ax2.text(v + max_r * 0.02, i, f"${v:,.0f}",
                     va="center", fontsize=9, color="#555555")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Churn Radar")
    st.caption("Turn order data into actionable churn scores.")
    st.divider()

    threshold = st.slider(
        "Churn probability threshold",
        min_value=0.10, max_value=0.90,
        value=0.55, step=0.05,
        key="threshold",
    )
    st.caption(
        f"Customers with churn probability ≥ **{threshold:.0%}** are flagged Medium risk "
        "or higher. Drag lower to catch more churners; drag higher to be more conservative."
    )

    st.divider()
    st.markdown("**Expected format**")
    st.markdown("**Required**")
    st.markdown("- `customer_id`\n- `order_date`\n- `order_value`")
    st.markdown("**Optional** (defaults applied if missing)")
    st.markdown("- `review_score` → 3.0\n- `freight_value` → 0\n- `item_count` → 1")


# ── Main ──────────────────────────────────────────────────────────────────
st.title("📡 Churn Radar")
st.markdown("Upload your customer order history and get churn risk scores in seconds.")
st.divider()

# STEP 1 — Upload
uploaded = st.file_uploader("Upload orders CSV", type="csv",
                             label_visibility="collapsed")

if uploaded is None:
    left, right = st.columns([1.2, 1])
    with left:
        st.info("👆 Upload a CSV file above to get started.")
        st.markdown("""
**What you'll get:**
- Churn probability (0–1) for every customer
- Risk segment: High / Medium / Low / Loyal
- Revenue at risk per customer and segment
- Downloadable CSV ready for Mailchimp or Klaviyo
        """)
    with right:
        st.markdown("**Sample format**")
        st.dataframe(pd.DataFrame({
            "customer_id":   ["cust_001", "cust_002", "cust_003"],
            "order_date":    ["2024-01-15", "2024-03-02", "2023-11-20"],
            "order_value":   [120.50, 89.99, 245.00],
            "review_score":  [5, 3, 1],
            "freight_value": [9.90, 0.00, 14.50],
            "item_count":    [2, 1, 3],
        }), use_container_width=True, hide_index=True)
    st.stop()

# Read bytes once; Streamlit provides a fresh object on each run
uploaded.seek(0)
file_bytes = uploaded.read()
file_id    = uploaded.file_id  # stable per upload, changes when user uploads a new file

try:
    preview_df = pd.read_csv(io.BytesIO(file_bytes))
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

all_cols     = list(preview_df.columns)
optional_opts = [NONE_OPT] + all_cols

# STEP 2 — Column mapper
st.markdown("### 1 — Preview")
st.dataframe(preview_df.head(3), use_container_width=True, hide_index=True)

st.markdown("### 2 — Map columns")
st.caption("Match your column names to the fields the model expects.  (* = required)")

def opt_idx(field: str) -> int:
    match = best_optional(field, all_cols)
    return optional_opts.index(match)

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

with c1:
    m_customer = st.selectbox("customer_id *", all_cols,
        index=all_cols.index(best_required("customer_id", all_cols)))
with c2:
    m_date = st.selectbox("order_date *", all_cols,
        index=all_cols.index(best_required("order_date", all_cols)))
with c3:
    m_value = st.selectbox("order_value *", all_cols,
        index=all_cols.index(best_required("order_value", all_cols)))
with c4:
    m_score   = st.selectbox("review_score (optional)",  optional_opts, index=opt_idx("review_score"))
with c5:
    m_freight = st.selectbox("freight_value (optional)", optional_opts, index=opt_idx("freight_value"))
with c6:
    m_items   = st.selectbox("item_count (optional)",    optional_opts, index=opt_idx("item_count"))

mapping = {
    "customer_id":   m_customer,
    "order_date":    m_date,
    "order_value":   m_value,
    "review_score":  m_score,
    "freight_value": m_freight,
    "item_count":    m_items,
}
mapping_tuple = tuple(sorted(mapping.items()))

run = st.button("▶  Confirm mapping and run analysis",
                type="primary", use_container_width=True)

if run:
    with st.spinner("Running analysis..."):
        try:
            rfm   = compute_rfm(file_bytes, mapping_tuple)
            model = load_model()
            X     = rfm[FEATURES].fillna(rfm[FEATURES].median())

            # Warn if any feature has > 50% nulls after mapping
            null_frac = X.isnull().mean()
            for col in null_frac[null_frac > 0.5].index:
                st.warning(
                    f"**{col}** has {null_frac[col]:.0%} null values after mapping — "
                    "check the column assignment above. The default fill value is being used."
                )

            # Score once; store raw probabilities in session state
            st.session_state["raw_probs"] = model.predict_proba(X)[:, 1]
            st.session_state["rfm"]       = rfm
            st.session_state["file_id"]   = file_id

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

# Clear results if a new file was uploaded
if st.session_state.get("file_id") != file_id:
    for key in ("raw_probs", "rfm"):
        st.session_state.pop(key, None)

# STEP 3 — Results dashboard (re-renders live on slider change)
if "raw_probs" in st.session_state and st.session_state.get("file_id") == file_id:

    raw_probs = st.session_state["raw_probs"]
    rfm       = st.session_state["rfm"]

    scored = rfm.copy()
    scored["churn_prob"]       = raw_probs
    scored["segment"]          = assign_segments(scored["churn_prob"], threshold)
    scored["revenue_at_risk"]  = scored["churn_prob"] * scored["monetary"]

    st.divider()
    st.markdown("### 3 — Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total customers", f"{len(scored):,}")
    m2.metric("High risk",       f"{(scored['segment'] == 'High risk').sum():,}")
    m3.metric("Revenue at risk", f"${scored['revenue_at_risk'].sum():,.0f}")
    m4.metric("Avg churn prob",  f"{scored['churn_prob'].mean():.2f}")

    render_charts(scored)

    # Filterable table
    selected_segs = st.multiselect(
        "Filter by segment", SEG_ORDER, default=SEG_ORDER,
        label_visibility="collapsed",
    )
    display = (
        scored[scored["segment"].isin(selected_segs)]
        [["customer_id", "segment", "churn_prob", "monetary", "revenue_at_risk"]]
        .rename(columns={"churn_prob": "churn_probability", "monetary": "total_spend"})
        .sort_values("churn_probability", ascending=False)
        .assign(churn_probability=lambda d: d["churn_probability"].round(2))
        .reset_index(drop=True)
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    # STEP 4 — Downloads
    st.divider()
    st.markdown("### 4 — Download")

    summary = (
        scored.groupby("segment")
        .agg(
            customers=("customer_id",      "count"),
            avg_spend=("monetary",         "mean"),
            avg_churn_prob=("churn_prob",  "mean"),
            total_revenue_at_risk=("revenue_at_risk", "sum"),
        )
        .sort_values("total_revenue_at_risk", ascending=False)
        .round(2)
    )

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇  Download scored CSV",
        data=(scored[["customer_id", "segment", "churn_prob", "monetary", "revenue_at_risk"]]
              .sort_values("churn_prob", ascending=False)
              .to_csv(index=False)),
        file_name="churn_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "⬇  Download summary report",
        data=summary.to_csv(),
        file_name="churn_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
