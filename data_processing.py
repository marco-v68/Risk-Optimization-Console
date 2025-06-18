data_processing.py
import pandas as pd
from datetime import datetime, timedelta
import numpy as np # Import numpy as it's used in some functions
from typing import Optional # <--- ADDED THIS LINE

def clean_currency(val):
    """Strip $ and commas, convert to float."""
    if pd.isna(val):
        return 0.0
    return float(str(val).replace('$','').replace(',','').strip())

def avg_days_between(dates: pd.Series) -> float:
    """Average days between successive dates in a Series."""
    # Ensure dates are datetime objects before calculating diff
    dates = pd.to_datetime(dates, errors='coerce').dropna()
    if dates.empty or len(dates) < 2:
        return 0.0
    diffs = dates.sort_values().diff().dt.days.dropna()
    return diffs.mean() if not diffs.empty else 0.0

def load_and_clean(input_csv_path: str) -> pd.DataFrame:
    """Loads raw CSV, cleans it, and adds initial derived columns."""
    df = pd.read_csv(input_csv_path, parse_dates=["txn_date"])
    df = df.drop_duplicates()
    # fulfillment_country
    df["fulfillment_country"] = (
        df["fulfillment_loc"].astype(str).str.upper().str.strip()
        .apply(lambda x: "US" if x.startswith("H") else "Canada")
    )
    # clean numeric
    df["item_rate"]     = df["item_rate"].apply(clean_currency)
    df["item_qty"]      = pd.to_numeric(df["item_qty"], errors="coerce").fillna(0)
    df["txn_amount"]    = df["txn_amount"].apply(clean_currency)
    df["hstk_std_cost"] = df["hstk_std_cost"].apply(clean_currency)
    # GL tag
    df["GL_tag"] = df["txn_amount"].abs().gt(0.01).map({True:"GL", False:"No GL"})
    # per-line metrics
    df["cogs_per_unit"]   = df["hstk_std_cost"]
    df["refund_per_unit"] = df.apply(
        lambda r: (r["txn_amount"]/r["item_qty"]) if r["item_qty"] else 0.0, axis=1
    )
    df["net_cost_impact"] = df["cogs_per_unit"]*df["item_qty"] + df["txn_amount"]
    # extra KPIs for lines
    df["unit_margin_impact"]      = df["item_rate"] - df["hstk_std_cost"]
    df["margin_efficiency_ratio"] = df.apply(
        lambda r: (r["unit_margin_impact"]/r["item_rate"]) if r["item_rate"] else 0.0,
        axis=1
    )
    df["margin_bleed_ratio"]      = df.apply(
        lambda r: abs(r["net_cost_impact"])/(r["unit_margin_impact"]*r["item_qty"])
                     if (r["unit_margin_impact"]*r["item_qty"]) else 0.0,
        axis=1
    )
    df["margin_loss_intensity"]   = df.apply(
        lambda r: abs(r["net_cost_impact"])/r["item_qty"] if r["item_qty"] else 0.0,
        axis=1
    )
    return df

def generate_sku_kpi_table(cleaned_df: pd.DataFrame,
                           kpi_csv_path: str,
                           as_of: Optional[datetime]=None) -> pd.DataFrame:
    if as_of is None:
        as_of = datetime.now()
    start = as_of.replace(month=1, day=1)
    last90 = as_of - timedelta(days=90)
    days_ytd = max((as_of - start).days, 1)

    df_ytd = cleaned_df[cleaned_df["txn_date"] >= start]
    df_90d = cleaned_df[cleaned_df["txn_date"] >= last90]

    def agg(df, label):
        return df.groupby("item_sku").agg(
            **{f"{label}_claims": ("doc_num","count"),
               f"{label}_net_cost": ("net_cost_impact","sum")}
        )

    kpi = agg(df_ytd, "YTD").join(agg(df_90d, "90D"), how="outer").fillna(0)

    stats = pd.DataFrame(index=cleaned_df["item_sku"].unique(), dtype=float)
    stats.index.name = "item_sku"
    stats["sku_share_of_loss"]           = cleaned_df.groupby("item_sku")["net_cost_impact"].sum() / cleaned_df["net_cost_impact"].sum()
    stats["sku_transaction_count"]       = cleaned_df.groupby("item_sku").size()
    stats["sku_gl_transaction_count"]    = cleaned_df[cleaned_df["GL_tag"]=="GL"].groupby("item_sku").size()
    stats["sku_credit_memo_count"]     = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"].groupby("item_sku").size()
    stats["sku_invoice_count"]         = cleaned_df[cleaned_df["txn_type"]=="Invoice"].groupby("item_sku").size()
    stats["total_sku_cost"]              = cleaned_df.groupby("item_sku")["net_cost_impact"].sum()
    stats["credit_memo_velocity_gl_only"]= stats["sku_credit_memo_count"] / days_ytd
    stats["invoice_velocity_gl_only"]    = stats["sku_invoice_count"]       / days_ytd
    stats["loss_per_touchpoint"]         = stats["total_sku_cost"] / stats["sku_transaction_count"]
    stats["sku_customer_recurrence"]     = cleaned_df.groupby("item_sku")["customer_name"].nunique()

    cm = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"]
    stats["avg_credit_memo_cost_per_sku"] = cm.groupby("item_sku")["net_cost_impact"].mean().fillna(0)
    stats["avg_days_between_failures_per_sku"] = cm.groupby("item_sku")["txn_date"].apply(avg_days_between)
    stats["failure_cost_per_unit_failed"] = cm.groupby("item_sku").apply(
        lambda g: g["net_cost_impact"].sum()/g["item_qty"].sum() if g["item_qty"].sum() else 0
    )
    cpi = cm.groupby(["item_sku","customer_name"])["doc_num"].count().pow(2)
    stats["customer_pain_index"] = cpi.groupby("item_sku").sum()
    stats["gl_bleed_density"]    = stats["total_sku_cost"] / stats["sku_gl_transaction_count"]

    td = cleaned_df.copy()
    td["days_since"] = (as_of - td["txn_date"]).dt.days
    stats["time_weighted_sku_cost_velocity"] = td.groupby("item_sku").apply(
        lambda g: (g["net_cost_impact"] * g["days_since"]).sum() / days_ytd
    )
    total_qty = cleaned_df.groupby("item_sku")["item_qty"].sum()
    stats["cost_per_unit_moved"] = stats["total_sku_cost"] / total_qty

    kpi = kpi.reset_index().merge(stats.reset_index(), on="item_sku", how="left")
    kpi.to_csv(kpi_csv_path, index=False)
    return kpi

def generate_customer_kpi_table(cleaned_df: pd.DataFrame,
                                  cust_csv_path: str,
                                  as_of: Optional[datetime]=None) -> pd.DataFrame:
    if as_of is None:
        as_of = datetime.now()
    start = as_of.replace(month=1, day=1)
    last90 = as_of - timedelta(days=90)

    df_ytd = cleaned_df[cleaned_df["txn_date"] >= start]
    df_90d = cleaned_df[cleaned_df["txn_date"] >= last90]

    def agg(df, label):
        return df.groupby("customer_name").agg(
            **{f"{label}_claims": ("doc_num","count"),
               f"{label}_net_cost":("net_cost_impact","sum")}
        )

    cust = agg(df_ytd, "YTD").join(agg(df_90d, "90D"), how="outer").fillna(0)

    stats = pd.DataFrame(index=cleaned_df["customer_name"].unique(), dtype=float)
    stats.index.name = "customer_name"
    stats["total_customer_cost"]           = cleaned_df.groupby("customer_name")["net_cost_impact"].sum()
    stats["total_units_serviced"]          = cleaned_df.groupby("customer_name")["item_qty"].sum()
    stats["customer_transaction_count"]    = cleaned_df.groupby("customer_name").size()
    stats["customer_gl_transaction_count"] = cleaned_df[cleaned_df["GL_tag"]=="GL"].groupby("customer_name").size()
    stats["customer_credit_memo_count"]    = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"].groupby("customer_name").size()
    stats["customer_invoice_count"]        = cleaned_df[cleaned_df["txn_type"]=="Invoice"].groupby("customer_name").size()
    stats["customer_avg_cost_per_unit"]    = stats["total_customer_cost"] / stats["total_units_serviced"].replace(0,1)
    stats["customer_avg_margin_loss"]      = cleaned_df.groupby("customer_name")["margin_loss_intensity"].mean()

    cm = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"]
    stats["avg_days_between_failures_per_customer"] = cm.groupby("customer_name")["txn_date"].apply(avg_days_between)
    stats["failure_count_per_customer"]    = cm.groupby("customer_name").size()
    stats["customer_sku_diversity"]        = cleaned_df.groupby("customer_name")["item_sku"].nunique()

    sku_counts = cleaned_df.groupby(["customer_name","item_sku"]).size()
    repeats = sku_counts[sku_counts>1].groupby("customer_name").size()
    stats["customer_repeat_rate"]          = repeats.div(stats["customer_sku_diversity"]).fillna(0)

    # bleed score
    max_cost  = stats["total_customer_cost"].abs().max()  or 1 # Use abs() for max cost
    max_mloss = stats["customer_avg_margin_loss"].abs().max() or 1 # Use abs() for max margin loss
    stats["customer_bleed_score"] = (
        (stats["total_customer_cost"].abs()/max_cost) + # Use abs() here too
        stats["customer_repeat_rate"] +
        (stats["customer_avg_margin_loss"].abs()/max_mloss) # Use abs() here too
    ) / 3

    us = cleaned_df[cleaned_df["fulfillment_country"]=="US"].groupby("customer_name").size()
    stats["customer_fulfillment_mix"] = us.div(stats["customer_transaction_count"]).fillna(0)

    state_bleed = cleaned_df.groupby(["customer_name","ship_state"])["net_cost_impact"].sum()
    stats["customer_state_bleed"] = state_bleed.groupby("customer_name").idxmax().apply(lambda x: x[1])

    cust = cust.reset_index().merge(stats.reset_index(), on="customer_name", how="left")
    cust.to_csv(cust_csv_path, index=False)
    return cust

def generate_location_kpi_table(cleaned_df: pd.DataFrame,
                                  loc_csv_path: str,
                                  as_of: Optional[datetime]=None) -> pd.DataFrame:
    if as_of is None:
        as_of = datetime.now()
    start = as_of.replace(month=1, day=1)
    last90 = as_of - timedelta(days=90)
    days_ytd = max((as_of - start).days, 1)

    df_ytd = cleaned_df[cleaned_df["txn_date"] >= start]
    df_90d = cleaned_df[cleaned_df["txn_date"] >= last90]

    def agg(df, label):
        return df.groupby("fulfillment_loc").agg(
            **{f"{label}_claims": ("doc_num","count"),
               f"{label}_net_cost":("net_cost_impact","sum")}
        )

    loc = agg(df_ytd, "YTD").join(agg(df_90d, "90D"), how="outer").fillna(0)

    stats = pd.DataFrame(index=cleaned_df["fulfillment_loc"].unique(), dtype=float)
    stats.index.name = "fulfillment_loc"
    stats["total_fulfillment_cost"]      = cleaned_df.groupby("fulfillment_loc")["net_cost_impact"].sum()
    stats["total_units_fulfilled"]       = cleaned_df.groupby("fulfillment_loc")["item_qty"].sum()
    stats["gl_transaction_count_by_loc"] = cleaned_df[cleaned_df["GL_tag"]=="GL"].groupby("fulfillment_loc").size()
    stats["avg_margin_loss_per_unit_by_loc"] = cleaned_df.groupby("fulfillment_loc")["margin_loss_intensity"].mean()
    tx_count = cleaned_df.groupby("fulfillment_loc").size()
    stats["bleed_density_by_loc"]        = stats["total_fulfillment_cost"] / tx_count.replace(0,1)
    stats["fulfillment_to_state_map"]    = cleaned_df.groupby("fulfillment_loc")["ship_state"].nunique()
    # delivery risk
    state_cost = cleaned_df.groupby("ship_state")["net_cost_impact"].sum()
    state_risk = (state_cost.abs()/state_cost.abs().max()).to_dict()
    method_counts = cleaned_df["ship_method"].value_counts()
    method_risk = (method_counts/method_counts.max()).to_dict()
    # Ensure 'delivery_risk_score' is calculated on 'cleaned_df' for proper mean aggregation later
    cleaned_df_copy_for_risk = cleaned_df.copy() # Avoid modifying original cleaned_df for later steps
    cleaned_df_copy_for_risk["delivery_risk_score"] = (
        cleaned_df_copy_for_risk["ship_state"].map(state_risk).fillna(0) + # Added .fillna(0)
        cleaned_df_copy_for_risk["ship_method"].map(method_risk).fillna(0) # Added .fillna(0)
    ) / 2
    stats["avg_delivery_risk_score"]     = cleaned_df_copy_for_risk.groupby("fulfillment_loc")["delivery_risk_score"].mean()
    stats["fulfillment_customer_footprint"] = cleaned_df.groupby("fulfillment_loc")["customer_name"].nunique()
    stats["location_cost_velocity"]      = stats["total_fulfillment_cost"] / days_ytd
    inv = cleaned_df[cleaned_df["txn_type"]=="Invoice"].copy()
    inv["margin_recovery"] = inv["unit_margin_impact"] * inv["item_qty"]
    rec = inv.groupby("fulfillment_loc")["margin_recovery"].sum()
    stats["fulfillment_margin_recovery_rate"] = rec / stats["total_fulfillment_cost"].abs().replace(0,1)
    stats["fulfillment_bleed_index"]     = stats["total_fulfillment_cost"].abs() / stats["total_units_fulfilled"].replace(0,1)

    loc = loc.reset_index().merge(stats.reset_index(), on="fulfillment_loc", how="left")
    loc.to_csv(loc_csv_path, index=False)
    return loc

def generate_state_kpi_table(cleaned_df: pd.DataFrame,
                                  state_csv_path: str,
                                  as_of: Optional[datetime]=None) -> pd.DataFrame:
    if as_of is None:
        as_of = datetime.now()
    start = as_of.replace(month=1, day=1)
    last90 = as_of - timedelta(days=90)

    df_ytd = cleaned_df[cleaned_df["txn_date"] >= start]
    df_90d = cleaned_df[cleaned_df["txn_date"] >= last90]

    def agg(df, label):
        return df.groupby("ship_state").agg(
            **{f"{label}_claims":("doc_num","count"),
               f"{label}_net_cost":("net_cost_impact","sum")}
        )

    state = agg(df_ytd, "YTD").join(agg(df_90d, "90D"), how="outer").fillna(0)

    stats = pd.DataFrame(index=cleaned_df["ship_state"].unique(), dtype=float)
    stats.index.name = "ship_state"
    stats["state_transaction_count"]    = cleaned_df.groupby("ship_state").size()
    stats["state_gl_transaction_count"] = cleaned_df[cleaned_df["GL_tag"]=="GL"].groupby("ship_state").size()
    stats["state_credit_memo_count"]    = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"].groupby("ship_state").size()
    stats["state_invoice_count"]        = cleaned_df[cleaned_df["txn_type"]=="Invoice"].groupby("ship_state").size()
    stats["total_state_cost"]           = cleaned_df.groupby("ship_state")["net_cost_impact"].sum()
    shipped = cleaned_df.groupby("ship_state")["item_qty"].sum().replace(0,1)
    stats["avg_cost_per_unit_by_state"] = stats["total_state_cost"] / shipped
    stats["bleed_density_by_state"]     = stats["total_state_cost"] / stats["state_transaction_count"].replace(0,1)
    fails = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"].groupby("ship_state")["item_qty"].sum()
    ship  = cleaned_df.groupby("ship_state")["item_qty"].sum().replace(0,1) # Recalculate ship for this scope
    stats["unit_failure_rate_by_state"] = fails.div(ship)
    stats["state_customer_count"]       = cleaned_df.groupby("ship_state")["customer_name"].nunique()
    stats["avg_margin_loss_by_state"]   = cleaned_df.groupby("ship_state")["margin_loss_intensity"].mean()
    cust_claims = cleaned_df.groupby(["ship_state","customer_name"])["doc_num"].count()
    repeats = cust_claims[cust_claims>1].groupby("ship_state").size()
    stats["repeat_claim_rate_by_state"] = repeats.div(stats["state_customer_count"].replace(0,1)).fillna(0)
    thresh = stats["bleed_density_by_state"].mean() + stats["bleed_density_by_state"].std()
    stats["high_risk_state_flag"]       = stats["bleed_density_by_state"] > thresh

    state = state.reset_index().merge(stats.reset_index(), on="ship_state", how="left")
    state.to_csv(state_csv_path, index=False)
    return state

def generate_monthly_kpi_table(cleaned_df: pd.DataFrame,
                                monthly_csv_path: str,
                                as_of: Optional[datetime]=None) -> pd.DataFrame:
    df = cleaned_df.copy()
    if as_of is None:
        as_of = datetime.now()
    
    # Ensure 'txn_date' is datetime type
    df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')
    # Drop rows where 'txn_date' could not be parsed to avoid errors
    df = df.dropna(subset=['txn_date'])

    if df.empty:
        print("DEBUG: generate_monthly_kpi_table received an empty DataFrame after cleaning txn_date.")
        # Create an empty DataFrame with expected columns if input is empty
        empty_monthly_df = pd.DataFrame(columns=[
            'month', 'monthly_transaction_count', 'monthly_gl_transaction_count',
            'monthly_credit_memo_count', 'monthly_invoice_count', 'monthly_total_cost',
            'monthly_total_units', 'monthly_avg_cost_per_unit',
            'monthly_margin_bleed_rate', 'monthly_repeat_customers',
            'monthly_high_risk_state_events', 'monthly_sku_bleed_leader',
            'monthly_fulfillment_risk_score'
        ])
        empty_monthly_df.to_csv(monthly_csv_path, index=False)
        return empty_monthly_df

    df["month"] = df["txn_date"].dt.to_period("M").astype(str)
    df["value_transacted"] = df["item_rate"] * df["item_qty"]

    monthly = df.groupby("month").agg(
        monthly_transaction_count=("doc_num","count"),
        monthly_gl_transaction_count=("GL_tag",lambda x:(x=="GL").sum()),
        monthly_credit_memo_count=("txn_type",lambda x:(x=="Credit Memo").sum()),
        monthly_invoice_count=("txn_type",lambda x:(x=="Invoice").sum()),
        monthly_total_cost=("net_cost_impact","sum"),
        monthly_total_units=("item_qty","sum"),
        total_value_transacted=("value_transacted","sum")
    )
    monthly["monthly_avg_cost_per_unit"] = monthly["monthly_total_cost"] / monthly["monthly_total_units"].replace(0,1)
    monthly["monthly_margin_bleed_rate"] = monthly["monthly_total_cost"] / monthly["total_value_transacted"].replace(0,1)

    repeat = df.groupby(["month","customer_name"]).size().groupby("month").apply(lambda x: (x>1).sum())
    monthly["monthly_repeat_customers"] = repeat.reindex(monthly.index).fillna(0) # Use reindex to align with monthly index

    state_cost = df.groupby("ship_state")["net_cost_impact"].sum()
    state_tx   = df.groupby("ship_state").size()
    bleed_den  = state_cost / state_tx.replace(0,1)
    thresh     = bleed_den.mean() + bleed_den.std()
    flags      = bleed_den[bleed_den>thresh].index
    high_risk  = df[df["ship_state"].isin(flags)].groupby("month").size()
    monthly["monthly_high_risk_state_events"] = high_risk.reindex(monthly.index).fillna(0) # Use reindex to align

    sku_loss = df.groupby(["month","item_sku"])["net_cost_impact"].sum()
    # Handle cases where a month might have no top bleed sku (e.g., all costs are 0 or only one sku)
    leader = sku_loss.groupby("month").apply(
        lambda x: x.idxmax()[1] if not x.empty and x.max() > 0 else "" # Get sku part of index, handle empty/zero
    )
    monthly["monthly_sku_bleed_leader"] = leader.reindex(monthly.index).fillna("") # Use reindex to align

    state_risk_dict    = (state_cost.abs()/state_cost.abs().max()).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict() # Handle inf and fillna
    method_risk_dict = (df["ship_method"].value_counts()/df["ship_method"].value_counts().max()).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict() # Handle inf and fillna
    
    # Calculate delivery_risk_score on the full df first, then aggregate by month
    df["delivery_risk_score"] = (df["ship_state"].map(state_risk_dict).fillna(0) + df["ship_method"].map(method_risk_dict).fillna(0)) / 2 
    monthly["monthly_fulfillment_risk_score"] = df.groupby("month")["delivery_risk_score"].mean().reindex(monthly.index).fillna(0) # Use reindex to align

    # Final fillna for all monthly KPIs before saving
    monthly = monthly.fillna({
        "monthly_transaction_count": 0,
        "monthly_gl_transaction_count": 0,
        "monthly_credit_memo_count": 0,
        "monthly_invoice_count": 0,
        "monthly_total_cost": 0,
        "monthly_total_units": 0,
        "monthly_avg_cost_per_unit": 0,
        "monthly_margin_bleed_rate": 0,
        "monthly_repeat_customers":0,
        "monthly_high_risk_state_events":0,
        "monthly_sku_bleed_leader":"",
        "monthly_fulfillment_risk_score":0
    }).drop(columns=["total_value_transacted"]) # total_value_transacted is an intermediate calculation

    # Debugging: Print columns just before returning
    print(f"DEBUG: Columns in monthly DataFrame just before return in data_processing.py: {monthly.columns.tolist()}")

    monthly.reset_index().to_csv(monthly_csv_path, index=False)
    return monthly.reset_index() # ENSURE 'month' IS A COLUMN WHEN RETURNED


def generate_category_kpi_table(cleaned_df: pd.DataFrame,
                                  category_csv_path: str) -> pd.DataFrame:
    stats = pd.DataFrame(index=cleaned_df["customer_category"].unique(), dtype=float)
    stats.index.name = "customer_category"
    stats["category_transaction_count"]    = cleaned_df.groupby("customer_category").size()
    stats["category_gl_transaction_count"] = cleaned_df[cleaned_df["GL_tag"]=="GL"].groupby("customer_category").size()
    stats["total_category_cost"]           = cleaned_df.groupby("customer_category")["net_cost_impact"].sum()
    stats["avg_cost_per_unit_by_category"] = stats["total_category_cost"] / cleaned_df.groupby("customer_category")["item_qty"].sum().replace(0,1)
    stats["avg_margin_loss_by_category"]   = cleaned_df.groupby("customer_category")["margin_loss_intensity"].mean()
    stats["category_bleed_rate"]           = stats["total_category_cost"] / stats["category_transaction_count"].replace(0,1)
    stats["avg_units_per_claim"]           = cleaned_df.groupby("customer_category")["item_qty"].sum() / stats["category_transaction_count"].replace(0,1)
    cc = cleaned_df.groupby(["customer_category","customer_name"])["doc_num"].count()
    repeats = cc[cc>1].groupby("customer_category").size()
    stats["repeat_claim_rate_by_category"] = repeats.div(cleaned_df.groupby("customer_category")["customer_name"].nunique()).fillna(0)
    stats["category_customer_count"]       = cleaned_df.groupby("customer_category")["customer_name"].nunique()
    us = cleaned_df[cleaned_df["fulfillment_country"]=="US"].groupby("customer_category").size()
    stats["category_fulfillment_mix"]      = us.div(stats["category_transaction_count"]).fillna(0)
    sku_cost = cleaned_df.groupby(["customer_category","item_sku"])["net_cost_impact"].sum()
    leader = sku_cost.groupby("customer_category").idxmax().apply(lambda x:x[1])
    stats["category_top_bleed_sku"]        = leader

    stats.reset_index().to_csv(category_csv_path, index=False)
    return stats.reset_index() # ENSURE 'customer_category' IS A COLUMN WHEN RETURNED

def generate_item_class_kpi_table(cleaned_df: pd.DataFrame,
                                   class_csv_path: str) -> pd.DataFrame:
    stats = pd.DataFrame(index=cleaned_df["item_class"].unique(), dtype=float)
    stats.index.name = "item_class"
    stats["item_class_transaction_count"]   = cleaned_df.groupby("item_class").size()
    stats["item_class_gl_transaction_count"]= cleaned_df[cleaned_df["GL_tag"]=="GL"].groupby("item_class").size()
    stats["item_class_credit_memo_count"]   = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"].groupby("item_class").size()
    stats["item_class_invoice_count"]       = cleaned_df[cleaned_df["txn_type"]=="Invoice"].groupby("item_class").size()
    stats["item_class_total_cost"]          = cleaned_df.groupby("item_class")["net_cost_impact"].sum()
    stats["item_class_avg_cost_per_unit"]   = stats["item_class_total_cost"] / cleaned_df.groupby("item_class")["item_qty"].sum().replace(0,1)
    fails = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"].groupby("item_class")["item_qty"].sum()
    ship  = cleaned_df.groupby("item_class")["item_qty"].sum().replace(0,1)
    stats["item_class_unit_failure_rate"]   = fails.div(ship)
    stats["item_class_bleed_density"]       = stats["item_class_total_cost"] / stats["item_class_transaction_count"].replace(0,1)
    stats["item_class_margin_efficiency"]   = cleaned_df.groupby("item_class")["margin_efficiency_ratio"].mean()
    sku_loss = cleaned_df.groupby(["item_class","item_sku"])["net_cost_impact"].sum()
    stats["item_class_top_bleed_sku"]       = sku_loss.groupby("item_class").idxmax().apply(lambda x:x[1])
    cm = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"]
    stats["item_class_avg_days_to_failure"] = cm.groupby("item_class")["txn_date"].apply(avg_days_between)
    cc = cleaned_df.groupby(["item_class","customer_name"])["doc_num"].count()
    reps = cc[cc>1].groupby("item_class").size()
    stats["item_class_repeat_customer_count"]= reps
    us = cleaned_df[cleaned_df["fulfillment_country"]=="US"].groupby("item_class").size()
    stats["item_class_fulfillment_mix"]      = us.div(stats["item_class_transaction_count"]).fillna(0)

    stats.reset_index().to_csv(class_csv_path, index=False)
    return stats.reset_index() # ENSURE 'item_class' IS A COLUMN WHEN RETURNED

def generate_item_type_kpi_table(cleaned_df: pd.DataFrame,
                                   type_csv_path: str) -> pd.DataFrame:
    stats = pd.DataFrame(index=cleaned_df["item_type"].unique(), dtype=float)
    stats.index.name = "item_type"
    stats["item_type_transaction_count"]    = cleaned_df.groupby("item_type").size()
    stats["item_type_gl_transaction_count"]= cleaned_df[cleaned_df["GL_tag"]=="GL"].groupby("item_type").size()
    stats["item_type_credit_memo_count"]    = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"].groupby("item_type").size()
    stats["item_type_invoice_count"]        = cleaned_df[cleaned_df["txn_type"]=="Invoice"].groupby("item_type").size()
    stats["item_type_total_cost"]           = cleaned_df.groupby("item_type")["net_cost_impact"].sum()
    stats["item_type_avg_cost_per_unit"]    = stats["item_type_total_cost"] / cleaned_df.groupby("item_type")["item_qty"].sum().replace(0,1)
    fails = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"].groupby("item_type")["item_qty"].sum()
    ship  = cleaned_df.groupby("item_type")["item_qty"].sum().replace(0,1)
    stats["item_type_unit_failure_rate"]    = fails.div(ship)
    stats["item_type_bleed_density"]        = stats["item_type_total_cost"] / stats["item_type_transaction_count"].replace(0,1)
    stats["item_type_margin_efficiency"]    = cleaned_df.groupby("item_type")["margin_efficiency_ratio"].mean()
    sku_loss = cleaned_df.groupby(["item_type","item_sku"])["net_cost_impact"].sum()
    stats["item_type_top_bleed_sku"]        = sku_loss.groupby("item_type").idxmax().apply(lambda x:x[1])
    cm = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"]
    stats["item_type_avg_days_to_failure"] = cm.groupby("item_type")["txn_date"].apply(avg_days_between)
    cc = cleaned_df.groupby(["item_type","customer_name"])["doc_num"].count()
    reps = cc[cc>1].groupby("item_type").size()
    stats["item_type_repeat_customer_count"] = reps
    us = cleaned_df[cleaned_df["fulfillment_country"]=="US"].groupby("item_type").size()
    stats["item_type_fulfillment_mix"]       = us.div(stats["item_type_transaction_count"]).fillna(0)

    stats.reset_index().to_csv(type_csv_path, index=False)
    return stats.reset_index() # ENSURE 'item_type' IS A COLUMN WHEN RETURNED

def generate_service_center_kpi_table(cleaned_df: pd.DataFrame,
                                      center_csv_path: str) -> pd.DataFrame:
    stats = pd.DataFrame(index=cleaned_df["customer_name"].unique(), dtype=float)
    stats.index.name = "customer_name"
    stats["service_center_transaction_count"]    = cleaned_df.groupby("customer_name").size()
    stats["service_center_gl_transaction_count"] = cleaned_df[cleaned_df["GL_tag"]=="GL"].groupby("customer_name").size()
    stats["service_center_credit_memo_count"]    = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"].groupby("customer_name").size()
    stats["service_center_invoice_count"]        = cleaned_df[cleaned_df["txn_type"]=="Invoice"].groupby("customer_name").size()
    stats["total_service_center_cost"]           = cleaned_df.groupby("customer_name")["net_cost_impact"].sum()
    # labor/mileage SKUs
    labor = cleaned_df[cleaned_df["item_sku"]=="SCLAB"].groupby("customer_name")["net_cost_impact"].sum()
    mile  = cleaned_df[cleaned_df["item_sku"]=="SCMIL"].groupby("customer_name")["net_cost_impact"].sum()
    # Use reindex and fillna(0) to ensure these columns exist for all customers in 'stats'
    # This prevents KeyError if 'SCLAB' or 'SCMIL' are not present for any customer or at all
    stats["total_labor_cost"]    = labor.reindex(stats.index).fillna(0)
    stats["total_mileage_cost"]  = mile.reindex(stats.index).fillna(0)

    stats["avg_cost_per_visit"] = stats["total_service_center_cost"] / stats["service_center_transaction_count"].replace(0,1)
    total_units = cleaned_df.groupby("customer_name")["item_qty"].sum()
    stats["avg_units_serviced"]          = total_units / stats["service_center_transaction_count"].replace(0,1)
    stats["bleed_per_unit"]              = stats["total_service_center_cost"] / total_units.replace(0,1)
    stats["avg_margin_loss_per_visit"]   = cleaned_df.groupby("customer_name")["margin_loss_intensity"].mean()
    stats["service_center_sku_diversity"]= cleaned_df.groupby("customer_name")["item_sku"].nunique()
    cc = cleaned_df.groupby(["customer_name","item_sku"])["doc_num"].count()
    stats["service_center_customer_repeats"] = cc.groupby("customer_name").apply(lambda g:(g>1).sum())
    cm = cleaned_df[cleaned_df["txn_type"]=="Credit Memo"]
    stats["return_window_avg_days"] = cm.groupby("customer_name")["txn_date"].apply(avg_days_between)
    us = cleaned_df[cleaned_df["fulfillment_country"]=="US"].groupby("customer_name").size()
    stats["fulfillment_mix_by_center"] = us.div(stats["service_center_transaction_count"]).fillna(0)
    thr = stats["bleed_per_unit"].mean() + stats["bleed_per_unit"].std()
    stats["service_center_qc_flag"] = stats["bleed_per_unit"] > thr

    stats.reset_index().to_csv(center_csv_path, index=False)
    return stats.reset_index() # ENSURE 'customer_name' IS A COLUMN WHEN RETURNED
