# Churn Radar — Streamlit App

Upload a customer order CSV and get churn risk scores in seconds.

---

## Run locally

```bash
# From the olist-churn/ directory, with your venv active:
pip install -r requirements_app.txt
streamlit run app.py
```

The app opens at http://localhost:8501

---

## What the threshold slider does

The sidebar slider sets the **minimum churn probability** to be flagged as Medium risk.

| Slider value | Effect |
|---|---|
| 0.35 (lower) | Wider net — more customers flagged, fewer churners missed |
| 0.55 (default) | Balanced — flags customers the model is moderately confident about |
| 0.75 (higher) | Conservative — only flags customers the model is very confident about |

High risk (≥ 0.80) is always fixed regardless of the slider.
Segments update live as you drag — the model does not re-run.

---

## What columns your CSV needs

### Required
| Your column | Maps to | Example |
|---|---|---|
| Unique customer identifier | `customer_id` | email address, user ID |
| Date of purchase | `order_date` | `2024-03-15` |
| Order total | `order_value` | `89.99` |

### Optional (app fills defaults if missing)
| Your column | Maps to | Default |
|---|---|---|
| Star rating 1–5 | `review_score` | 3.0 |
| Shipping cost | `freight_value` | 0 |
| Items per order | `item_count` | 1 |

The column mapper in Step 2 lets you match your column names — you do not need to rename anything.

---

## Shopify export

Orders → Export → All orders → Export orders (CSV for Excel)

| Shopify column | Map to |
|---|---|
| Email | customer_id |
| Paid at | order_date |
| Total | order_value |
| Shipping | freight_value |
| Lineitem quantity | item_count |

## WooCommerce export

Use WooCommerce built-in export or the free plugin "Customer / Order / Coupon Export".

| WooCommerce column | Map to |
|---|---|
| billing_email | customer_id |
| date_created | order_date |
| order_total | order_value |
| shipping_total | freight_value |

---

## Deploy free on Streamlit Community Cloud

1. Push your project to a **public** GitHub repo. Include:
   - `app.py`
   - `requirements_app.txt`
   - `data/processed/churn_model_clean.pkl`

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **New app** → select your repo → set main file to `app.py` → **Deploy**.

4. Your app gets a permanent URL:  
   `https://your-username-churn-radar.streamlit.app`

Share this URL with clients. They upload their CSV, get scores, download results — no code required.

---

## Charging for the app

| Offer | Price |
|---|---|
| One-off audit (includes app) | $300–$500 |
| Monthly retainer (updated model + hosting) | $200–$350/month |
| White-label (client logo + colors) | +$200 one-time |
| Direct Shopify API integration (no CSV) | +$500 one-time |

---

## Deploy in 5 minutes

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Churn Radar v1.0"
```

Create a new repo on github.com (do not initialize with README), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/churn-radar.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

- Go to share.streamlit.io
- Sign in with GitHub
- Click "New app"
- Select your repo, branch: main, main file: app.py
- Click Deploy
- Wait 2 to 3 minutes
- Your app is live at: `https://YOUR_USERNAME-churn-radar.streamlit.app`

### Step 3: Send this to your client

> "Here is your live churn dashboard: [URL]
> Export your orders as CSV from Shopify or WooCommerce,
> upload it, and you will have every customer scored in under 30 seconds."
