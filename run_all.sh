#!/usr/bin/env bash
set -e

echo "============================================"
echo "  Olist Churn Prediction Pipeline"
echo "============================================"

echo ""
echo "[1/5] Loading and merging raw data..."
python src/01_load_and_merge.py

echo ""
echo "[2/5] Engineering RFM features..."
python src/02_feature_engineering.py

echo ""
echo "[3/5] Training XGBoost model..."
python src/03_train_model.py

echo ""
echo "[4/5] Segmenting customers by risk..."
python src/04_segment_customers.py

echo ""
echo "[5/5] Exporting client deliverable..."
python src/05_export_deliverable.py

echo ""
echo "============================================"
echo "  Project complete. Check the output/ folder."
echo "============================================"
