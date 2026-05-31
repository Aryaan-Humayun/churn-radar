"""
Step 4: Score every customer with the trained model, assign risk segments,
calculate revenue at risk, and produce the segment summary charts.
"""
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

PROCESSED = "data/processed/"
OUTPUT = "output/"
os.makedirs(f"{OUTPUT}charts", exist_ok=True)

rfm = pd.read_parquet(f"{PROCESSED}rfm_features.parquet")

FEATURES = [
    "recency", "frequency", "monetary",
    "avg_score", "avg_freight", "avg_items",
    "late_delivery_rate",
]

with open(f"{PROCESSED}churn_model.pkl", "rb") as f:
    model = pickle.load(f)

X = rfm[FEATURES].fillna(rfm[FEATURES].median())

# --- Score every customer ---
rfm["churn_prob"] = model.predict_proba(X)[:, 1]

# --- Assign segment ---
def assign_segment(p):
    if p >= 0.80:
        return "High risk"
    elif p >= 0.55:
        return "Medium risk"
    elif p >= 0.30:
        return "Low risk"
    else:
        return "Loyal"

rfm["segment"] = rfm["churn_prob"].apply(assign_segment)
rfm["revenue_at_risk"] = rfm["churn_prob"] * rfm["monetary"]

# --- Segment summary table ---
summary = (
    rfm
    .groupby("segment")
    .agg(
        customers=("customer_unique_id", "count"),
        avg_monetary=("monetary", "mean"),
        avg_churn_prob=("churn_prob", "mean"),
        total_revenue_at_risk=("revenue_at_risk", "sum"),
    )
    .sort_values("total_revenue_at_risk", ascending=False)
    .round(2)
)
print("\nSegment summary:")
print(summary.to_string())
summary.to_csv(f"{OUTPUT}segment_summary.csv")
print(f"\nSaved: output/segment_summary.csv")

# --- Charts ---
COLORS = {
    "High risk":   "#E24B4A",
    "Medium risk": "#EF9F27",
    "Low risk":    "#378ADD",
    "Loyal":       "#1D9E75",
}
SEG_ORDER = ["High risk", "Medium risk", "Low risk", "Loyal"]

counts = rfm["segment"].value_counts().reindex(SEG_ORDER).fillna(0).astype(int)
rev    = rfm.groupby("segment")["revenue_at_risk"].sum().reindex(SEG_ORDER).fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Customer Churn Risk Analysis", fontsize=15, fontweight="bold", y=1.02)

# Left: customer count per segment
axes[0].bar(
    counts.index, counts.values,
    color=[COLORS[s] for s in counts.index],
    edgecolor="white", linewidth=0.5,
)
axes[0].set_title("Customers by risk segment", fontsize=12)
axes[0].set_ylabel("Number of customers")
for i, (seg, val) in enumerate(counts.items()):
    axes[0].text(i, val + 200, f"{val:,}", ha="center", fontsize=10)

# Right: revenue at risk per segment
axes[1].barh(
    rev.index, rev.values,
    color=[COLORS[s] for s in rev.index],
    edgecolor="white",
)
axes[1].set_title("Revenue at risk by segment (BRL)", fontsize=12)
axes[1].set_xlabel("Revenue at risk (BRL)")
for i, (seg, val) in enumerate(rev.items()):
    axes[1].text(val + 5000, i, f"R${val:,.0f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT}charts/segment_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart saved: output/charts/segment_analysis.png")
print("Done.")
