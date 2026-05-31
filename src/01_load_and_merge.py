"""
Step 1: Load all 5 relevant Olist CSVs, filter to delivered orders,
aggregate payments/reviews/items to order level, and merge into one
wide table saved as a parquet file.
"""
import pandas as pd
import os

RAW = "data/raw/"
PROCESSED = "data/processed/"
os.makedirs(PROCESSED, exist_ok=True)

# --- Load ---
orders    = pd.read_csv(f"{RAW}olist_orders_dataset.csv")
customers = pd.read_csv(f"{RAW}olist_customers_dataset.csv")
payments  = pd.read_csv(f"{RAW}olist_order_payments_dataset.csv")
reviews   = pd.read_csv(f"{RAW}olist_order_reviews_dataset.csv")
items     = pd.read_csv(f"{RAW}olist_order_items_dataset.csv")

print(f"Orders loaded: {len(orders):,}")

# --- Filter to delivered orders only ---
orders = orders[orders["order_status"] == "delivered"].copy()
print(f"Delivered orders: {len(orders):,}")

# --- Parse timestamps ---
ts_cols = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for col in ts_cols:
    orders[col] = pd.to_datetime(orders[col])

# --- Aggregate: payments per order (some orders have split payments) ---
pay_agg = (
    payments
    .groupby("order_id")
    .agg(
        total_payment=("payment_value", "sum"),
        installments=("payment_installments", "max"),
        payment_type=("payment_type", "first"),
    )
    .reset_index()
)

# --- Aggregate: reviews per order ---
rev_agg = (
    reviews
    .groupby("order_id")
    .agg(review_score=("review_score", "mean"))
    .reset_index()
)

# --- Aggregate: items per order ---
item_agg = (
    items
    .groupby("order_id")
    .agg(
        item_count=("order_item_id", "count"),
        freight_value=("freight_value", "sum"),
        avg_price=("price", "mean"),
    )
    .reset_index()
)

# --- Merge all into one table ---
df = (
    orders
    .merge(customers, on="customer_id", how="left")
    .merge(pay_agg,   on="order_id",   how="left")
    .merge(rev_agg,   on="order_id",   how="left")
    .merge(item_agg,  on="order_id",   how="left")
)

print(f"Merged table shape: {df.shape}")
print(f"Null review scores: {df['review_score'].isna().sum():,}")

df.to_parquet(f"{PROCESSED}orders_merged.parquet", index=False)
print("Done. Saved: data/processed/orders_merged.parquet")
