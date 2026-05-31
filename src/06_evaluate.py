import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.calibration import CalibrationDisplay
from xgboost import XGBClassifier

PROCESSED = "data/processed/"
OUTPUT    = "output/charts/"

rfm = pd.read_parquet(f"{PROCESSED}rfm_features.parquet")

# --- FIX: remove recency from features ---
# recency IS the label definition (recency > 180 = churned)
# including it is like giving the model the answer key
FEATURES = [
    "frequency",
    "monetary",
    "avg_score",
    "avg_freight",
    "avg_items",
    "late_delivery_rate",
]

X = rfm[FEATURES].fillna(rfm[FEATURES].median())
y = rfm["churned"]

print(f"Class balance — loyal: {(y==0).sum():,}  churned: {(y==1).sum():,}")

# --- Proper hold-out split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Retrain clean model on train split only ---
scale = (y_train==0).sum() / (y_train==1).sum()

model = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale, subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", use_label_encoder=False, random_state=42
)
model.fit(X_train, y_train)

# --- Evaluate on unseen test set ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)

print("=" * 50)
print(f"ROC-AUC (no leakage): {auc:.3f}")
print("=" * 50)
print(classification_report(y_test, y_pred,
      target_names=["Loyal", "Churned"]))

# --- Save model ---
with open(f"{PROCESSED}churn_model_clean.pkl", "wb") as f:
    pickle.dump(model, f)

# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Evaluation (no leakage)", fontsize=14, fontweight="bold")

RocCurveDisplay.from_predictions(
    y_test, y_prob, ax=axes[0], name=f"XGBoost (AUC={auc:.2f})")
axes[0].plot([0,1],[0,1],"k--",lw=0.8,label="Random")
axes[0].set_title("ROC curve")
axes[0].legend()

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Loyal","Churned"],
    ax=axes[1], colorbar=False)
axes[1].set_title("Confusion matrix")

CalibrationDisplay.from_predictions(
    y_test, y_prob, n_bins=10, ax=axes[2], name="XGBoost")
axes[2].set_title("Calibration curve\n(perfect = diagonal)")

plt.tight_layout()
plt.savefig(f"{OUTPUT}model_evaluation_clean.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved: {OUTPUT}model_evaluation_clean.png")

# --- Threshold sensitivity ---
threshold = 0.35
y_pred_adjusted = (y_prob >= threshold).astype(int)
print(f"\nWith threshold = {threshold}:")
print(classification_report(y_test, y_pred_adjusted,
      target_names=["Loyal", "Churned"]))