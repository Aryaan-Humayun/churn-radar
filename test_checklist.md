# Churn Radar — Pre-Client Testing Checklist

Run through every check below before showing the app to a client.
App must be running at http://localhost:8501

---

## Test File Reference

| File | Rows | Customers | Purpose |
|---|---|---|---|
| `test_orders.csv` | 462 | 200 | Primary test — all 6 columns, nulls included |
| `test_orders_shopify_style.csv` | 462 | 200 | Column mapper test — Shopify column names |

**Expected segment distribution at threshold=0.55 (test_orders.csv):**
| Segment | Customers | % |
|---|---|---|
| High risk | ~2 | ~1% |
| Medium risk | ~43 | ~21% |
| Low risk | ~68 | ~34% |
| Loyal | ~87 | ~44% |

Note: the clean model excludes recency, so distribution will skew lower-risk than an
Olist-trained model with recency. This is expected and correct.

---

## CHECK 1 — Landing state (no file uploaded)

**What to do:** Open the app. Do not upload anything.

**PASS:**
- Sample format table is visible
- "What you'll get" bullet list is visible
- No charts, no tables, no metric cards are shown
- Sidebar shows the threshold slider at 0.55

**FAIL:**
- Any results or empty containers are shown
- App throws an error on load

---

## CHECK 2 — File upload and preview

**What to do:** Upload `test_orders.csv`

**PASS:**
- First 3 rows of the CSV appear in a preview table immediately
- 6 column-mapping dropdowns appear below the preview
- "Confirm mapping and run analysis" button is visible
- No results appear yet (button not clicked)

**FAIL:**
- Error on upload
- App shows results before the button is clicked
- Preview shows more than 3 rows

---

## CHECK 3 — Column mapper auto-detection

**What to do:** Inspect the 6 dropdowns after uploading `test_orders.csv`

**PASS — each dropdown should pre-select:**
| Dropdown | Expected auto-selection |
|---|---|
| customer_id | `customer_id` |
| order_date | `order_date` |
| order_value | `order_value` |
| review_score | `review_score` |
| freight_value | `freight_value` |
| item_count | `item_count` |

All 6 should match exactly since column names are standard.

**FAIL:**
- Any required dropdown shows the wrong column
- Optional dropdowns default to `(not in file)` when the column exists

---

## CHECK 4 — Run analysis with standard CSV

**What to do:** Click "Confirm mapping and run analysis" with `test_orders.csv`

**PASS:**
- Spinner appears while processing
- 4 metric cards appear: Total customers=200, High risk ~2, Revenue at risk ~$16,300, Avg churn prob ~0.34
- 2 charts appear side by side: left=bar chart by segment, right=horizontal bar by revenue
- Filtered table appears sorted by churn_probability descending
- All 4 segment filter pills are selected by default
- 2 download buttons appear at the bottom

**FAIL:**
- Error during scoring
- Metric cards show 0 or NaN
- Charts are blank or missing colors
- Table does not appear

---

## CHECK 5 — Null handling (review_score and freight_value)

**What to do:** After running analysis, check that results produced without error
despite ~16.5% null review_score and ~9.3% null freight_value in the file.

**PASS:**
- No error thrown
- Total customers is exactly 200 (nulls filled, not dropped)
- If a warning banner appears it should say which column has >50% nulls
  (neither column exceeds 50% here, so no warning is expected)

**FAIL:**
- Customer count is less than 200 (nulls incorrectly dropped)
- Error about NaN in model input
- App crashes silently and shows nothing

---

## CHECK 6 — Segment filter

**What to do:** In the multiselect above the table, deselect "High risk" and "Loyal"

**PASS:**
- Table instantly updates to show only Medium risk and Low risk customers
- Row count updates accordingly
- Metric cards do NOT change (they show totals, not filtered view)
- Charts do NOT change

**FAIL:**
- Table does not update
- App re-runs the model when filter changes
- Metric cards reset

---

## CHECK 7 — Threshold slider (live re-segmentation)

**What to do:** With results showing, drag the sidebar slider from 0.55 down to 0.25,
then back up to 0.80

**PASS at 0.25:**
- More customers move into Medium risk (lower bar to enter)
- High risk count stays the same or increases (fixed at >=0.80)
- Loyal count decreases
- Charts update immediately
- Model is NOT re-run (no spinner)

**PASS at 0.80:**
- Medium risk count drops significantly (bar raised to 0.80)
- More customers fall into Low risk and Loyal
- High risk may show 0

**FAIL:**
- Spinner appears when slider changes (model re-running unnecessarily)
- Charts do not update
- Metric cards do not update
- App crashes

---

## CHECK 8 — Revenue at risk calculation

**What to do:** Look at the "Revenue at risk" metric card and the table

**Manual spot-check formula:** For any customer in the table:
```
revenue_at_risk = churn_probability * total_spend
```

Pick 2 customers from the table. Multiply their churn_probability by total_spend.
The result should match revenue_at_risk within rounding.

**PASS:** Values match the formula for both checked customers.

**FAIL:** revenue_at_risk values don't match or are all 0.

---

## CHECK 9 — Download buttons

**What to do:** Click both download buttons

**PASS — "Download scored CSV":**
- File downloads as `churn_scores.csv`
- Opens in Excel/Numbers: columns are customer_id, segment, churn_prob, monetary, revenue_at_risk
- Sorted by churn_prob descending
- Row count matches total customers (200)

**PASS — "Download summary report":**
- File downloads as `churn_summary.csv`
- 4 rows (one per segment)
- Columns: segment, customers, avg_spend, avg_churn_prob, total_revenue_at_risk

**FAIL:**
- Download button does nothing
- File is empty or has wrong columns
- Row count is wrong

---

## CHECK 10 — Shopify column mapper

**What to do:** Upload `test_orders_shopify_style.csv`

Shopify columns in this file: `Email`, `Paid at`, `Total`, `Lineitem quantity`

**PASS — expected auto-selections:**
| Dropdown | Expected auto-selection |
|---|---|
| customer_id | `Email` |
| order_date | `Paid at` |
| order_value | `Total` |
| review_score | `(not in file)` |
| freight_value | `(not in file)` |
| item_count | `Lineitem quantity` |

Click run. Results should appear for 200 customers.
review_score fills to 3.0 and freight_value fills to 0.0 (defaults).

**PASS:**
- Column mapper pre-selects the correct Shopify columns automatically
- Analysis runs without error
- 200 customers scored
- Avg churn probability may differ slightly from test_orders.csv because
  review_score and freight_value are filled with defaults

**FAIL:**
- Mapper cannot find Email, Paid at, or Total
- Analysis errors on missing columns
- Customer count is wrong

---

## CHECK 11 — New file clears old results

**What to do:** With Shopify results showing, upload `test_orders.csv` again

**PASS:**
- Column mapper resets to the new file's columns
- Old results disappear until the button is clicked again
- Metric cards do not show stale data from the previous file

**FAIL:**
- Results from the Shopify file are still visible with the new file loaded
- Column mapper shows Shopify column names when standard CSV is loaded

---

## Summary: Quick Smoke Test (3 minutes)

For a fast pre-call check, run only these:

1. Upload `test_orders.csv` — verify 200 customers load
2. Click run — verify metric cards show reasonable numbers
3. Drag slider to 0.25 — verify charts update without spinner
4. Click "Download scored CSV" — verify file downloads with 200 rows
5. Upload `test_orders_shopify_style.csv` — verify Email/Paid at/Total auto-detected
6. Click run — verify 200 customers scored with defaults applied

If all 6 pass, the app is ready to demo.
