"""
Step 3: Train an XGBoost churn classifier on RFM features with 5-fold
stratified cross-validation. Save the model and a SHAP importance chart.
"""
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

PROCESSED = "data/processed/"
OUTPUT = "output/charts/"
os.makedirs(OUTPUT, exist_ok=True)

rfm = pd.read_parquet(f"{PROCESSED}rfm_features.parquet")

FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_score",
    "avg_freight",
    "avg_items",
    "late_delivery_rate",
]

X = rfm[FEATURES].fillna(rfm[FEATURES].median())
y = rfm["churned"]

# --- Class weight to handle 97% churn imbalance ---
neg = (y == 0).sum()
pos = (y == 1).sum()
scale = neg / pos
print(f"Class imbalance ratio: {scale:.1f}  (scale_pos_weight)")

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

# --- 5-fold stratified cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"\nCross-validated ROC-AUC: {roc_scores.mean():.3f} +/- {roc_scores.std():.3f}")

# --- Fit on all data ---
model.fit(X, y)

with open(f"{PROCESSED}churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved: data/processed/churn_model.pkl")

# --- SHAP feature importance chart ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, X,
    feature_names=FEATURES,
    plot_type="bar",
    show=False,
)
plt.title("Feature Importance (SHAP values)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT}shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart saved: output/charts/shap_importance.png")

# --- Quick classification report on training data (informational) ---
y_pred = model.predict(X)
print("\nClassification report (training data — informational only):")
print(classification_report(y, y_pred))
print("Done.")
