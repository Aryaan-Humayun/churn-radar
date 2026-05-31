"""
Step 5: Score all customers and export the final client deliverable CSV
sorted by churn probability descending.
"""
import pandas as pd
import pickle
import os

PROCESSED = "data/processed/"
OUTPUT = "output/"
os.makedirs(OUTPUT, exist_ok=True)

rfm = pd.read_parquet(f"{PROCESSED}rfm_features.parquet")

FEATURES = [
    "recency", "frequency", "monetary",
    "avg_score", "avg_freight", "avg_items",
    "late_delivery_rate",
]

with open(f"{PROCESSED}churn_model.pkl", "rb") as f:
    model = pickle.load(f)

X = rfm[FEATURES].fillna(rfm[FEATURES].median())
rfm["churn_prob"] = model.predict_proba(X)[:, 1]

def assign_segment(p):
    if p >= 0.80:   return "High risk"
    elif p >= 0.55: return "Medium risk"
    elif p >= 0.30: return "Low risk"
    else:           return "Loyal"

rfm["segment"] = rfm["churn_prob"].apply(assign_segment)
rfm["revenue_at_risk"] = rfm["churn_prob"] * rfm["monetary"]

EXPORT_COLS = [
    "customer_unique_id",
    "recency",
    "frequency",
    "monetary",
    "avg_score",
    "churn_prob",
    "segment",
    "revenue_at_risk",
]

out = rfm[EXPORT_COLS].sort_values("churn_prob", ascending=False)
out.to_csv(f"{OUTPUT}churn_scores_by_customer.csv", index=False)

print(f"Exported: output/churn_scores_by_customer.csv")
print(f"Total customers scored: {len(out):,}")
print("\nTop 10 highest-risk customers:")
print(out.head(10).to_string(index=False))
print("\nDone.")
