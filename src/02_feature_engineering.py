"""
Step 2: Collapse order-level data into one row per customer with RFM
features plus late delivery rate. Define churn label (recency > 180 days).
"""
import pandas as pd
import numpy as np
import os

PROCESSED = "data/processed/"
os.makedirs(PROCESSED, exist_ok=True)

df = pd.read_parquet(f"{PROCESSED}orders_merged.parquet")

# Snapshot date: one day after the last order in the dataset
SNAPSHOT_DATE = pd.Timestamp("2018-10-01")

# --- Late delivery flag at order level ---
df["delivered_late"] = (
    df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
).astype(int)

# --- Aggregate to customer level ---
# customer_unique_id tracks the real person; customer_id changes per order
rfm = (
    df
    .groupby("customer_unique_id")
    .agg(
        recency=("order_purchase_timestamp",
                 lambda x: (SNAPSHOT_DATE - x.max()).days),
        frequency=("order_id", "count"),
        monetary=("total_payment", "sum"),
        avg_score=("review_score", "mean"),
        avg_freight=("freight_value", "mean"),
        avg_items=("item_count", "mean"),
        late_delivery_rate=("delivered_late", "mean"),
        customer_state=("customer_state", "first"),
        last_payment_type=("payment_type", "last"),
    )
    .reset_index()
)

# --- Fill missing review scores with neutral 3 ---
rfm["avg_score"] = rfm["avg_score"].fillna(3.0)

# --- Churn label: no purchase in last 180 days before snapshot ---
rfm["churned"] = (rfm["recency"] > 180).astype(int)

print("Churn rate breakdown:")
print(rfm["churned"].value_counts(normalize=True).round(3))
print(f"\nTotal customers: {len(rfm):,}")
print("\nFeature summary:")
print(rfm[["recency", "frequency", "monetary", "avg_score",
           "late_delivery_rate"]].describe().round(2))

rfm.to_parquet(f"{PROCESSED}rfm_features.parquet", index=False)
print("\nDone. Saved: data/processed/rfm_features.parquet")
