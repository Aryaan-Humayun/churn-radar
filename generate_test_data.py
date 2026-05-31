"""
Generates test_orders.csv and test_orders_shopify_style.csv for app testing.
Run once from the olist-churn/ directory.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

# ── Customer groups ────────────────────────────────────────────────────────
N_SINGLE      = 120   # 60% — one purchase, skew low scores
N_REPEAT_LOW  = 40    # 20% — 2–3 orders, mid scores
N_REPEAT_HIGH = 40    # 20% — 4+ orders, skew high scores

START = pd.Timestamp("2022-01-01")
END   = pd.Timestamp("2025-01-01")
SPAN  = (END - START).days


def rand_date() -> str:
    return (START + pd.Timedelta(days=int(rng.integers(0, SPAN)))).strftime("%Y-%m-%d")

def rand_value(mu=120, sigma=60) -> float:
    return round(float(np.clip(rng.normal(mu, sigma), 20, 500)), 2)

def rand_freight() -> float:
    return round(float(np.clip(rng.normal(14, 7), 0, 80)), 2)


rows = []

# Single-purchase customers (likely churners)
for i in range(N_SINGLE):
    cust = f"cust_{i+1:04d}"
    score = int(rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.35, 0.15, 0.10]))
    rows.append({
        "customer_id":   cust,
        "order_date":    rand_date(),
        "order_value":   rand_value(110, 55),
        "review_score":  score,
        "freight_value": rand_freight(),
        "item_count":    int(rng.integers(1, 4)),
    })

# 2–3 order customers
for i in range(N_REPEAT_LOW):
    cust   = f"cust_{N_SINGLE + i + 1:04d}"
    n_ord  = int(rng.integers(2, 4))
    for _ in range(n_ord):
        score = int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.25, 0.35, 0.25]))
        rows.append({
            "customer_id":   cust,
            "order_date":    rand_date(),
            "order_value":   rand_value(120, 55),
            "review_score":  score,
            "freight_value": rand_freight(),
            "item_count":    int(rng.integers(1, 5)),
        })

# 4+ order customers (loyal)
for i in range(N_REPEAT_HIGH):
    cust  = f"cust_{N_SINGLE + N_REPEAT_LOW + i + 1:04d}"
    n_ord = int(rng.integers(4, 9))
    for _ in range(n_ord):
        score = int(rng.choice([1, 2, 3, 4, 5], p=[0.02, 0.05, 0.10, 0.33, 0.50]))
        rows.append({
            "customer_id":   cust,
            "order_date":    rand_date(),
            "order_value":   rand_value(145, 65),
            "review_score":  score,
            "freight_value": rand_freight(),
            "item_count":    int(rng.integers(1, 6)),
        })

df = pd.DataFrame(rows)

# Introduce missing values
df.loc[rng.random(len(df)) < 0.15, "review_score"]  = np.nan   # 15% nulls
df.loc[rng.random(len(df)) < 0.10, "freight_value"] = np.nan   # 10% nulls

df.to_csv("test_orders.csv", index=False)
print(f"Saved test_orders.csv  ({len(df):,} rows)")

# ── Shopify-style CSV ──────────────────────────────────────────────────────
shopify = df.rename(columns={
    "customer_id":   "Email",
    "order_date":    "Paid at",
    "order_value":   "Total",
    "item_count":    "Lineitem quantity",
}).drop(columns=["review_score", "freight_value"])

shopify.to_csv("test_orders_shopify_style.csv", index=False)
print(f"Saved test_orders_shopify_style.csv  ({len(shopify):,} rows)")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  TEST DATA SUMMARY")
print("="*55)

# Customer-level stats
cust = (
    df.groupby("customer_id")
    .agg(orders=("order_value", "count"),
         total_spend=("order_value", "sum"),
         avg_score=("review_score", "mean"))
    .reset_index()
)

print(f"\nCustomers:     {cust['customer_id'].nunique():,}")
print(f"Total rows:    {len(df):,}")
print(f"Date range:    {df['order_date'].min()}  to  {df['order_date'].max()}")
print(f"\nOrder value:   min={df['order_value'].min():.2f}  "
      f"mean={df['order_value'].mean():.2f}  "
      f"max={df['order_value'].max():.2f}")
print(f"\nMissing values:")
print(f"  review_score  : {df['review_score'].isna().sum():>4} rows "
      f"({df['review_score'].isna().mean():.1%})")
print(f"  freight_value : {df['freight_value'].isna().sum():>4} rows "
      f"({df['freight_value'].isna().mean():.1%})")

print(f"\nCustomers by order frequency:")
freq_bins = pd.cut(cust["orders"], bins=[0,1,3,99],
                   labels=["1 order (single)", "2–3 orders", "4+ orders"])
print(cust.groupby(freq_bins, observed=True)["customer_id"].count().to_string())

# ── Predict expected segment distribution ─────────────────────────────────
MODEL_PATH = Path("data/processed/churn_model_clean.pkl")
FEATURES   = ["frequency", "monetary", "avg_score",
              "avg_freight", "avg_items", "late_delivery_rate"]

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

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
    rfm["avg_score"]          = rfm["avg_score"].fillna(3.0)
    rfm["avg_freight"]        = rfm["avg_freight"].fillna(0.0)
    rfm["avg_items"]          = rfm["avg_items"].fillna(1.0)
    rfm["late_delivery_rate"] = 0.0

    X = rfm[FEATURES].fillna(rfm[FEATURES].median())
    probs = model.predict_proba(X)[:, 1]

    def seg(p, t=0.55):
        if p >= 0.80: return "High risk"
        if p >= t:    return "Medium risk"
        if p >= 0.30: return "Low risk"
        return "Loyal"

    segments = pd.Series([seg(p) for p in probs])
    order    = ["High risk", "Medium risk", "Low risk", "Loyal"]

    print("\nExpected segment distribution (threshold=0.55):")
    for s in order:
        n    = (segments == s).sum()
        pct  = n / len(segments)
        bar  = "#" * int(pct * 30)
        print(f"  {s:<13} {n:>4}  ({pct:.1%})  {bar}")

    print(f"\nAvg churn probability:  {probs.mean():.3f}")
    print(f"Revenue at risk (total): ${(probs * rfm['monetary']).sum():,.0f}")

except FileNotFoundError:
    print("\n(Model not found — run 03_train_model.py first to see segment preview)")

print("\n" + "="*55)
