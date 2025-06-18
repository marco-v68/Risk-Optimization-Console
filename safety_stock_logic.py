    safety_stock_logic.py
    import numpy as np
    from typing import Optional
    import pandas as pd
    from datetime import datetime

    # IMPORTANT: There should NOT be any 'from safety_stock_logic import ...' here.
    # This file defines the function, it doesn't import it from itself.

    def generate_safety_stock_table(cleaned_df: pd.DataFrame,
                                    safety_csv_path: str,
                                    as_of: Optional[datetime] = None) -> pd.DataFrame:
        if as_of is None:
            as_of = datetime.now()

        # 2) Pull Credit Memos (CMs)
        cm = cleaned_df[
            (cleaned_df["txn_type"]=="Credit Memo")
        ].copy()

        # 3) Pull zero-cost Invoices
        inv = cleaned_df[
            (cleaned_df["txn_type"]=="Invoice") &
            (cleaned_df["txn_amount"]==0)
        ].copy()

        # Combine all failure-related transactions for common processing
        all_failures_df = pd.concat([cm, inv])

        # 4) Total 'failure' transactions QUANTITY by SKU x Loc (now summing absolute item_qty)
        tot_failure_qty = all_failures_df.groupby(["item_sku","fulfillment_loc"])["item_qty"].apply(lambda x: x.abs().sum())

        # 5) Build base table - MODIFIED to include 'item_class' and 'hstk_std_cost'
        base_indices = list(tot_failure_qty.index)

        # --- MODIFIED: Ensure unique item_sku/fulfillment_loc for item_class and hstk_std_cost ---
        # We drop duplicates on 'item_sku', 'fulfillment_loc' and keep the first encountered item_class/hstk_std_cost
        temp_df_item_attributes = cleaned_df.drop_duplicates(
            subset=['item_sku', 'fulfillment_loc'], keep='first'
        )[['item_sku', 'fulfillment_loc', 'item_class', 'hstk_std_cost']]

        table = pd.DataFrame(base_indices, columns=["item_sku","fulfillment_loc"])

        table = pd.merge(table, temp_df_item_attributes, on=['item_sku', 'fulfillment_loc'], how='left')
        table['item_class'] = table['item_class'].fillna('UNKNOWN')
        table['hstk_std_cost'] = table['hstk_std_cost'].fillna(0)


        # 6) Keep only specified core fulfillment locations
        keep_locs = ["HREP","CREP","HSTK","CSTK"]
        table = table[table["fulfillment_loc"].isin(keep_locs)].reset_index(drop=True)

        if table.empty:
            print("⚠️ No relevant item_sku/fulfillment_loc combinations found after filtering.")
            return pd.DataFrame()

        skus_to_exclude = ["SCLAB", "SCMIL"]
        table = table[~table["item_sku"].isin(skus_to_exclude)].reset_index(drop=True)

        if table.empty:
            print(f"⚠️ No relevant item_sku/fulfillment_loc combinations remaining after excluding {skus_to_exclude}.")
            return pd.DataFrame()


        # 7) Failure probability P_f
        all_transactions_for_pf_denominator = cleaned_df[
            cleaned_df["txn_type"].isin(["Credit Memo", "Invoice"])
        ].groupby(["item_sku","fulfillment_loc"]).size()

        cm_cnt = cm.groupby(["item_sku","fulfillment_loc"]).size()
        inv_cnt = inv.groupby(["item_sku","fulfillment_loc"]).size()
        tot_failure_transaction_cnt = cm_cnt.add(inv_cnt, fill_value=0)

        table["P_f"] = table.apply(
            lambda r: tot_failure_transaction_cnt.get((r["item_sku"],r["fulfillment_loc"]),0) /
                      all_transactions_for_pf_denominator.get((r["item_sku"],r["fulfillment_loc"]),1),
            axis=1
        )

        # 8) sigma_demand_qty: Standard Deviation of Daily Failure QUANTITY (corrected)
        all_failures_df['txn_date'] = pd.to_datetime(all_failures_df['txn_date'])

        daily_failure_qty_series = all_failures_df.groupby(["item_sku", "fulfillment_loc", all_failures_df['txn_date'].dt.date])["item_qty"].apply(lambda x: x.abs().sum())

        std_dev_daily_qty = daily_failure_qty_series.groupby(["item_sku", "fulfillment_loc"]).std().fillna(0)

        cap_std_dev = std_dev_daily_qty.quantile(0.99)
        std_dev_daily_qty_capped = std_dev_daily_qty.clip(upper=cap_std_dev).to_dict()

        table["sigma_demand_qty"] = table.apply(
            lambda r: std_dev_daily_qty_capped.get((r["item_sku"],r["fulfillment_loc"]),0), axis=1
        )

        # 9) Frequency F (using total failure transaction count for consistency)
        max_tot_failure_transaction_cnt = tot_failure_transaction_cnt.max()
        table["F"] = table.apply(
            lambda r: tot_failure_transaction_cnt.get((r["item_sku"],r["fulfillment_loc"]),0)/max_tot_failure_transaction_cnt,
            axis=1
        )

        # 10) Entropy H
        def sku_entropy(sku):
            sub = cm[cm["item_sku"]==sku]
            if sub.empty: return 0.0
            p = sub.groupby("fulfillment_loc").size()/len(sub)
            return - (p * np.log2(p + 1e-9)).sum()
        table["H"] = table["item_sku"].apply(sku_entropy)

        # 11) Lead time L_t by country
        combined_txns = pd.concat([cm, inv])
        loc2country = combined_txns.set_index("fulfillment_loc")["fulfillment_country"].drop_duplicates().to_dict()

        table["L_t"] = table["fulfillment_loc"].apply(
            lambda loc: 7 if loc2country.get(loc,"US")=="US" else 14
        )

        # 11.5) Estimate Standard Deviation of Lead Time (sigma_LT)
        table["sigma_LT"] = table["L_t"] * 0.25
        table["sigma_LT"] = table["sigma_LT"].fillna(0)

        # 12) Daily failures D (now average daily QUANTITY of failures)
        start = as_of.replace(month=1,day=1)
        days_ytd = max((as_of - start).days, 1)

        table["D"] = table.apply(
            lambda r, days_ytd=days_ytd: tot_failure_qty.get((r["item_sku"],r["fulfillment_loc"]),0)/days_ytd,
            axis=1
        )

        # 13) Inject margin KPIs
        group_all = cleaned_df.groupby(["item_sku","fulfillment_loc"])

        table["avg_margin_loss_intensity"] = table.apply(
            lambda r: group_all.get_group((r["item_sku"],r["fulfillment_loc"]))["margin_loss_intensity"].mean()
            if (r["item_sku"],r["fulfillment_loc"]) in group_all.groups else 0.0,
            axis=1
        ).fillna(0.0)

        table["avg_unit_margin_impact"] = table.apply(
            lambda r: group_all.get_group((r["item_sku"],r["fulfillment_loc"]))["unit_margin_impact"].mean()
            if (r["item_sku"],r["fulfillment_loc"]) in group_all.groups else 0.0,
            axis=1
        ).fillna(0.0)

        # 14) Composite Safety Score (transformed margin KPIs)
        max_abs_margin_loss_intensity = table["avg_margin_loss_intensity"].abs().max() or 1
        max_abs_unit_margin_impact = table["avg_unit_margin_impact"].abs().max() or 1

        table["S_score"] = (
            0.18*table["P_f"] +
            0.18*(table["sigma_demand_qty"]/(table["sigma_demand_qty"].max() or 1)) +
            0.18*table["F"] +
            0.18*(table["H"]/(table["H"].max() or 1)) +
            0.14*(table["avg_margin_loss_intensity"].abs()/max_abs_margin_loss_intensity) +
            0.14*(table["avg_unit_margin_impact"].abs()/max_abs_unit_margin_impact)
        )
        table["S_score"] = table["S_score"].fillna(0)

        # 14.5) Assign Item-Specific Z-score based on S_score for Business Criticality
        s_score_quantiles = table["S_score"].quantile([0.25, 0.50, 0.75]).tolist()

        def get_item_specific_z(s_score, quantiles):
            if len(quantiles) < 3: # Handle case where not enough data for 3 quantiles
                return 1.65

            q1, q2, q3 = quantiles[0], quantiles[1], quantiles[2]

            if s_score >= q3:
                return 2.33
            elif s_score >= q2:
                return 1.96
            elif s_score >= q1:
                return 1.65
            else:
                return 1.28

        table["item_specific_z"] = table["S_score"].apply(lambda s: get_item_specific_z(s, s_score_quantiles))


        # 15) Safety Stock and Reorder Point Calculation
        demand_variance_term = table["L_t"] * (table["sigma_demand_qty"]**2)
        lead_time_variance_term = (table["D"]**2) * (table["sigma_LT"]**2)

        combined_variance = np.maximum(0, demand_variance_term + lead_time_variance_term)

        table["safety_stock_qty"] = table["item_specific_z"] * np.sqrt(combined_variance)

        table["safety_stock_qty"] = np.ceil(table["safety_stock_qty"])
        table["safety_stock_qty"] = table["safety_stock_qty"].fillna(0)

        expected_demand_during_lt = table["D"] * table["L_t"]
        expected_demand_during_lt = np.maximum(0, expected_demand_during_lt)

        table["Reorder_Point"] = expected_demand_during_lt + table["safety_stock_qty"]
        table["Reorder_Point"] = np.maximum(0, table["Reorder_Point"])

        # 15.5) Calculate Total Safety Stock Value
        table["total_safety_stock_value"] = table["safety_stock_qty"] * table["hstk_std_cost"]
        table["total_safety_stock_value"] = table["total_safety_stock_value"].fillna(0)


        # 16) Save & return
        table.to_csv(safety_csv_path, index=False)
        print(f"Wrote safety stock table: {safety_csv_path} ({len(table)} rows)")
        return table
