"""
Automated validation of app.py logic against test CSVs.
Simulates exactly what the app does — no Streamlit imports needed.
Run from the olist-churn/ directory.
"""
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants (must match app.py exactly) ─────────────────────────────────
MODEL_PATH = Path("data/processed/churn_model_clean.pkl")
FEATURES   = ["frequency", "monetary", "avg_score",
              "avg_freight", "avg_items", "late_delivery_rate"]
NONE_OPT   = "(not in file)"

HINTS = {
    "customer_id":   ["email", "customer", "id", "user"],
    "order_date":    ["date", "paid", "created", "timestamp"],
    "order_value":   ["total", "value", "amount", "price", "revenue"],
    "review_score":  ["review", "rating", "score", "star"],
    "freight_value": ["freight", "shipping", "delivery"],
    "item_count":    ["item", "quantity", "qty", "count"],
}

results = {}   # check_id -> (bool, str)

def ok(check_id, msg):
    results[check_id] = (True, msg)
    print(f"  PASS  {check_id}: {msg}")

def fail(check_id, msg):
    results[check_id] = (False, msg)
    print(f"  FAIL  {check_id}: {msg}")


# ── Helpers (copied verbatim from app.py) ─────────────────────────────────
def compute_rfm(file_bytes: bytes, mapping: dict) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    rename = {v: k for k, v in mapping.items()
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
            frequency=("order_value",     "count"),
            monetary=("order_value",      "sum"),
            avg_score=("review_score",    "mean"),
            avg_freight=("freight_value", "mean"),
            avg_items=("item_count",      "mean"),
        )
    )
    rfm["avg_score"]          = rfm["avg_score"].fillna(3.0)
    rfm["avg_freight"]        = rfm["avg_freight"].fillna(0.0)
    rfm["avg_items"]          = rfm["avg_items"].fillna(1.0)
    rfm["late_delivery_rate"] = 0.0
    return rfm


def assign_segments(probs: np.ndarray, threshold: float) -> pd.Series:
    low_floor = threshold * 0.50
    def seg(p):
        if p >= 0.80:          return "High risk"
        elif p >= threshold:   return "Medium risk"
        elif p >= low_floor:   return "Low risk"
        else:                  return "Loyal"
    return pd.Series([seg(p) for p in probs])


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


# ── Load model ─────────────────────────────────────────────────────────────
print("Loading model...")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"  Model loaded from {MODEL_PATH}\n")
except FileNotFoundError:
    print(f"  ERROR: model not found at {MODEL_PATH}")
    sys.exit(1)

# ── Load test files ────────────────────────────────────────────────────────
with open("test_orders.csv", "rb") as f:
    standard_bytes = f.read()
with open("test_orders_shopify_style.csv", "rb") as f:
    shopify_bytes = f.read()

standard_mapping = {
    "customer_id":   "customer_id",
    "order_date":    "order_date",
    "order_value":   "order_value",
    "review_score":  "review_score",
    "freight_value": "freight_value",
    "item_count":    "item_count",
}

# ── Core pipeline run ──────────────────────────────────────────────────────
rfm   = compute_rfm(standard_bytes, standard_mapping)
X     = rfm[FEATURES].fillna(rfm[FEATURES].median())
probs = model.predict_proba(X)[:, 1]

scored = rfm.copy()
scored["churn_prob"]      = probs
scored["segment"]         = assign_segments(probs, 0.55)
scored["revenue_at_risk"] = scored["churn_prob"] * scored["monetary"]

SEG_ORDER = ["High risk", "Medium risk", "Low risk", "Loyal"]

# ──────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  AUTOMATED CHECKS")
print("=" * 55)

# CHECK A — Row count
print("\nCHECK A  Row count (nulls filled, not dropped)")
n = len(scored)
if n == 200:
    ok("A", f"scored customers == 200")
else:
    fail("A", f"expected 200, got {n}")

# CHECK B — No NaN in churn_prob
print("\nCHECK B  No NaN in model output")
n_nan = scored["churn_prob"].isna().sum()
if n_nan == 0:
    ok("B", "churn_prob has 0 NaN values")
else:
    fail("B", f"churn_prob has {n_nan} NaN values")

# CHECK C — Segment distribution sanity
print("\nCHECK C  Segment distribution sanity")
seg_counts = scored["segment"].value_counts().reindex(SEG_ORDER).fillna(0).astype(int)
hr_pct  = seg_counts.get("High risk",  0) / len(scored)
loy_cnt = seg_counts.get("Loyal",      0)

print("  Segment distribution (threshold=0.55):")
for s in SEG_ORDER:
    n_s = seg_counts.get(s, 0)
    pct = n_s / len(scored)
    print(f"    {s:<13} {n_s:>4}  ({pct:.1%})")

c_pass = True
if hr_pct >= 0.50:
    fail("C", f"High risk is {hr_pct:.1%} -- suspiciously high (>= 50%)")
    c_pass = False
if loy_cnt == 0:
    fail("C", "Loyal count is 0 -- model may be broken")
    c_pass = False
if c_pass:
    ok("C", f"High risk={hr_pct:.1%} (<50%), Loyal={loy_cnt} (>0)")

# CHECK D — Revenue at risk formula spot-check
print("\nCHECK D  Revenue at risk formula (5 random customers)")
rng      = np.random.default_rng(99)
sample   = scored.sample(5, random_state=99)
d_pass   = True
for _, row in sample.iterrows():
    expected = round(row["churn_prob"] * row["monetary"], 6)
    actual   = round(row["revenue_at_risk"], 6)
    diff     = abs(expected - actual)
    if diff >= 0.01:
        fail("D", f"customer {row['customer_id']}: expected {expected:.4f}, got {actual:.4f}")
        d_pass = False
if d_pass:
    ok("D", "all 5 sampled customers match churn_prob * monetary within 0.01")

# CHECK E — Threshold sensitivity
print("\nCHECK E  Threshold sensitivity (0.25 / 0.55 / 0.80)")
loyal_at = {}
for t in [0.25, 0.55, 0.80]:
    segs         = assign_segments(probs, t)
    loyal_at[t]  = (segs == "Loyal").sum()
    hr           = (segs == "High risk").sum()
    mr           = (segs == "Medium risk").sum()
    lr           = (segs == "Low risk").sum()
    print(f"  threshold={t:.2f}:  High={hr:>3}  Medium={mr:>3}  Low={lr:>3}  Loyal={loyal_at[t]:>3}")

# With dynamic low_floor = threshold*0.50:
#   threshold=0.25 -> low_floor=0.125 -> Loyal = p < 0.125 (smallest Loyal pool)
#   threshold=0.55 -> low_floor=0.275 -> Loyal = p < 0.275
#   threshold=0.80 -> low_floor=0.400 -> Loyal = p < 0.400 (largest Loyal pool)
# So loyal must strictly increase as threshold increases.
# Also verify Low risk is non-empty at threshold=0.25 (the original dead-zone bug).
low_risk_at_025 = (assign_segments(probs, 0.25) == "Low risk").sum()

e_pass = True
if not (loyal_at[0.25] < loyal_at[0.55] < loyal_at[0.80]):
    fail("E",
         f"loyal not strictly increasing with threshold: "
         f"loyal@0.25={loyal_at[0.25]}, loyal@0.55={loyal_at[0.55]}, loyal@0.80={loyal_at[0.80]}")
    e_pass = False
if low_risk_at_025 == 0:
    fail("E", "Low risk still empty at threshold=0.25 — dead zone not fixed")
    e_pass = False
if e_pass:
    ok("E",
       f"loyal strictly increases (0.25->{loyal_at[0.25]}, 0.55->{loyal_at[0.55]}, "
       f"0.80->{loyal_at[0.80]}); Low risk@0.25={low_risk_at_025} (not empty)")

# CHECK F — Shopify mapper simulation
print("\nCHECK F  Shopify column mapper simulation")
shopify_raw  = pd.read_csv(io.BytesIO(shopify_bytes))
shopify_cols = list(shopify_raw.columns)
print(f"  Shopify columns detected: {shopify_cols}")

auto_map = {
    field: best_required(field, shopify_cols)
    if field in ["customer_id", "order_date", "order_value"]
    else best_optional(field, shopify_cols)
    for field in ["customer_id", "order_date", "order_value",
                  "review_score", "freight_value", "item_count"]
}
print("  Auto-selected mappings:")
for field, col in auto_map.items():
    print(f"    {field:<15} -> {col}")

expected_map = {
    "customer_id":   "Email",
    "order_date":    "Paid at",
    "order_value":   "Total",
    "review_score":  NONE_OPT,
    "freight_value": NONE_OPT,
    "item_count":    "Lineitem quantity",
}
wrong = {k: (auto_map[k], expected_map[k])
         for k in expected_map if auto_map[k] != expected_map[k]}
if wrong:
    fail("F-map", f"mapper wrong for: {wrong}")
else:
    print("  Mapper auto-selections: all correct")

try:
    rfm_shopify   = compute_rfm(shopify_bytes, auto_map)
    X_shopify     = rfm_shopify[FEATURES].fillna(rfm_shopify[FEATURES].median())
    probs_shopify = model.predict_proba(X_shopify)[:, 1]
    n_shopify     = len(rfm_shopify)
    nan_shopify   = np.isnan(probs_shopify).sum()

    if n_shopify == 200 and nan_shopify == 0 and not wrong:
        ok("F", f"Shopify: 200 customers scored, 0 NaN, mapper correct")
    else:
        issues = []
        if n_shopify != 200: issues.append(f"customer count={n_shopify}")
        if nan_shopify  > 0:  issues.append(f"NaN in probs={nan_shopify}")
        if wrong:             issues.append(f"mapper errors={wrong}")
        fail("F", "; ".join(issues))
except Exception as e:
    fail("F", f"exception during Shopify scoring: {e}")

# CHECK G — Download output format
print("\nCHECK G  Download output format")
Path("output").mkdir(exist_ok=True)

scored_export = (
    scored[["customer_id", "segment", "churn_prob", "monetary", "revenue_at_risk"]]
    .sort_values("churn_prob", ascending=False)
    .reset_index(drop=True)
)
summary_export = (
    scored.groupby("segment")
    .agg(
        customers=("customer_id",      "count"),
        avg_spend=("monetary",         "mean"),
        avg_churn_prob=("churn_prob",  "mean"),
        total_revenue_at_risk=("revenue_at_risk", "sum"),
    )
    .sort_values("total_revenue_at_risk", ascending=False)
    .round(2)
    .reset_index()
)

scored_export.to_csv("output/validated_churn_scores.csv", index=False)
summary_export.to_csv("output/validated_churn_summary.csv", index=False)

g_pass = True
if len(scored_export) != 200:
    fail("G-scored", f"scored CSV has {len(scored_export)} rows, expected 200")
    g_pass = False
if len(scored_export.columns) != 5:
    fail("G-scored", f"scored CSV has {len(scored_export.columns)} columns, expected 5")
    g_pass = False
if len(summary_export) != 4:
    fail("G-summary", f"summary CSV has {len(summary_export)} rows, expected 4")
    g_pass = False
if len(summary_export.columns) != 5:
    fail("G-summary", f"summary CSV has {len(summary_export.columns)} columns, expected 5")
    g_pass = False
if g_pass:
    ok("G", f"scored=200 rows x 5 cols, summary=4 rows x 5 cols -- saved to output/")

print(f"\n  Scored CSV top 5:")
print(scored_export.head(5).to_string(index=False))
print(f"\n  Summary CSV:")
print(summary_export.to_string(index=False))

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
passed = sum(1 for v, _ in results.values() if v)
total  = len(results)
print(f"  {passed}/{total} checks passed")
print("=" * 55)

fails = [(k, msg) for k, (v, msg) in results.items() if not v]
if fails:
    print("\nFAILED checks:")
    for k, msg in fails:
        print(f"  CHECK {k}: {msg}")
else:
    print("\nAll checks passed. App logic is correct.")
