# Sample Data

Anonymized data for running the Risk Optimization Console end to end.

Customer names, employee names, SKUs, and document numbers have been replaced with sequential identifiers; ZIP codes and memo notes removed; cost and price fields scaled by a random ±15% factor. Row counts, structure, date ranges, and dimensional relationships are unchanged, so the model produces comparable output.

---

## `warranty_raw_sample.csv` — input

7,995 warranty transactions (invoices and credit memos) spanning Jan 2022 – May 2025, across 359 customers, 1,083 SKUs, and 5 fulfillment locations in the US and Canada.

| Column | Description |
|---|---|
| `txn_date` | Transaction date |
| `txn_type` | Invoice or Credit Memo |
| `doc_num` | Document identifier |
| `created_by` | User who entered the transaction |
| `customer_name` | Service center / customer |
| `customer_category` | Customer segment (10 categories) |
| `ship_state` | Destination state or province |
| `ship_method` | Carrier and service (UPS Ground, LTL, expedited air, etc.) |
| `ship_tier` | Service tier grouping |
| `fulfillment_loc` | Fulfillment location code (HREP, HSTK, HFLD, CREP, CSTK) |
| `fulfillment_country` | US or Canada, derived from location code |
| `item_category` / `item_class` / `item_type` | Product hierarchy (13 / 48 / 4 levels) |
| `item_sku` | Item identifier |
| `item_rate` | Unit price |
| `item_qty` | Quantity (negative on credit memos) |
| `txn_amount` | Transaction amount |
| `hstk_std_cost` | Standard cost per unit |

Derived during cleaning: `net_cost_impact`, `unit_margin_impact`, `margin_efficiency_ratio`, `margin_bleed_ratio`, `margin_loss_intensity`, `refund_per_unit`, `GL_tag`.

---

## `safety_stock_output_sample.csv` — model output

1,412 SKU-location pairs with computed safety stock and reorder points.

### Inputs to the model

| Column | Description |
|---|---|
| `P_f` | Failure probability — failure transactions ÷ total transactions for that SKU-location |
| `sigma_demand_qty` | Standard deviation of daily failure quantity, capped at the 99th percentile to limit outlier influence |
| `F` | Failure frequency, normalized against the highest-frequency pair |
| `H` | Shannon entropy of a SKU's failure distribution across locations — high values indicate failures spread broadly rather than concentrated |
| `L_t` | Lead time in days (7 domestic, 14 cross-border) |
| `sigma_LT` | Estimated lead-time standard deviation (25% of `L_t`) |
| `D` | Average daily demand, from year-to-date failure quantity |
| `avg_margin_loss_intensity` | Mean margin loss per unit for that SKU-location |
| `avg_unit_margin_impact` | Mean unit margin (price less standard cost) |

### Scoring and service level

| Column | Description |
|---|---|
| `S_score` | Composite criticality score — weighted blend of failure probability, demand variability, frequency, entropy, and margin impact (0.18 / 0.18 / 0.18 / 0.18 / 0.14 / 0.14) |
| `item_specific_z` | Service-level Z assigned by `S_score` quartile: 1.28, 1.65, 1.96, or 2.33 (≈90%, 95%, 97.5%, 99%). Higher-criticality items get higher coverage |

### Results

| Column | Description |
|---|---|
| `safety_stock_qty` | `Z × √(L_t · σ_demand² + D² · σ_LT²)`, rounded up |
| `Reorder_Point` | Expected demand during lead time (`D × L_t`) plus safety stock |
| `total_safety_stock_value` | `safety_stock_qty × hstk_std_cost` |

---

## Assumptions

Lead times are set by country rather than derived from historical receipt data, and lead-time variability is estimated as a fixed proportion of lead time. Both would be computed from actuals where receipt history is available.
