# Churn Radar

> Predict which customers are about to disappear — and how much revenue is at risk.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://churn-radar-az3zojkywalmjoiqweqhjt.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)

A machine learning app that scores every customer in your e-commerce store
by churn probability. Upload a CSV export from Shopify or WooCommerce,
get every customer scored in under 30 seconds, and download a risk-ranked
list ready to upload into Mailchimp or Klaviyo as a win-back campaign.

Built on 96,000 real orders from the Olist Brazilian E-Commerce dataset.
Validated at AUC 0.82 on a held-out test set with no data leakage.

---

## Live demo

https://churn-radar-az3zojkywalmjoiqweqhjt.streamlit.app

Upload the included test_orders.csv to see a full demo with 200 customers.

---

## What it does

1. You upload your order history as a CSV (Shopify, WooCommerce, or any format)
2. You map your column names to the expected fields using dropdown menus
3. The model scores every customer with a churn probability (0 to 1)
4. Every customer is placed into a segment: High risk, Medium risk, Low risk, or Loyal
5. Revenue at risk is calculated per customer (churn probability x total spend)
6. You download a scored CSV ready to import into any email marketing platform

---

## How the model works

- Dataset: Olist Brazilian E-Commerce (96,000 delivered orders, 2016 to 2018)
- Model: XGBoost binary classifier
- Features: purchase frequency, total spend, average review score,
  average freight value, average items per order, late delivery rate
- Validation: 5-fold stratified cross-validation, AUC 0.82 on held-out test set
- Leakage fix: recency was removed from features after AUC 1.00 was detected
  on first run — the full diagnosis and fix is documented in src/06_evaluate.py

---

## Segment logic

| Segment | Condition |
|---|---|
| High risk | churn probability >= 0.80 |
| Medium risk | churn probability >= threshold (default 0.55) |
| Low risk | churn probability >= threshold x 0.50 |
| Loyal | below low floor |

The threshold is adjustable via a sidebar slider. Segments update live
without re-running the model.

---

## Project structure

Config and setup:
- .gitignore
- .streamlit/config.toml
- requirements.txt
- requirements_app.txt
- run_all.sh

Pipeline (src/):
- 01_load_and_merge.py
- 02_feature_engineering.py
- 03_train_model.py
- 04_segment_customers.py
- 05_export_deliverable.py
- 06_evaluate.py
- 07_validate_app.py

App:
- app.py

---

## Run locally

```bash
pip install -r requirements_app.txt
streamlit run app.py
```

App opens at http://localhost:8501

---

## Run the full pipeline

Requires the Olist dataset CSVs placed in data/raw/
Download from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

```bash
bash run_all.sh
```

---

## Deploy to Streamlit Cloud (free)

1. Fork this repo
2. Go to https://share.streamlit.io
3. Connect your GitHub account
4. Select this repo, branch main, main file app.py
5. Click Deploy

---

## Column mapping

The app auto-detects your column names. Supported export formats:

Shopify: Email, Paid at, Total, Lineitem quantity
WooCommerce: billing_email, date_created, order_total, shipping_total
Generic: any CSV with customer ID, order date, and order value columns

Optional columns (review_score, freight_value, item_count) are filled
with neutral defaults if not present.

---

## Built with

- Python 3.11
- XGBoost 2.0
- Streamlit 1.32
- pandas, scikit-learn, matplotlib, SHAP
- Olist Brazilian E-Commerce Dataset (CC BY-NC-SA 4.0)
