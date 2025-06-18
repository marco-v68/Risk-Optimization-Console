streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta # Import timedelta for date calculations
import os
import plotly.express as px # For advanced visualizations
import plotly.graph_objects as go # For more control over heatmaps if needed
from typing import Optional # Required for type hinting in data_processing
from sklearn.preprocessing import MinMaxScaler # For heatmap data normalization
import numpy as np # Import numpy for inf handling

# Import your functions from the separate files
# Ensure these files (data_processing.py, safety_stock_logic.py) are in your GitHub repo
from data_processing import (
    load_and_clean,
    generate_sku_kpi_table,
    generate_customer_kpi_table,
    generate_location_kpi_table,
    generate_state_kpi_table,
    generate_monthly_kpi_table,
    generate_category_kpi_table,
    generate_item_class_kpi_table,
    generate_item_type_kpi_table,
    generate_service_center_kpi_table
)
# The generate_safety_stock_table function is defined in safety_stock_logic.py
# There is no circular import here as safety_stock_logic.py does not import from itself.
from safety_stock_logic import generate_safety_stock_table

st.set_page_config(layout="wide", page_title="Comprehensive Inventory & Warranty Analytics", initial_sidebar_state="expanded")

# --- Robust Session State Initialization for all persistent variables ---
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = pd.DataFrame()

if 'generated_dfs' not in st.session_state:
    st.session_state.generated_dfs = {}

# Initialize as_of_datetime in session state if not present
if 'as_of_datetime' not in st.session_state:
    st.session_state.as_of_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

# Define output_paths Globally
output_paths = {
    "safety_stock_output.csv": "Safety Stock Data",
    "sku_kpi_output.csv": "SKU KPIs",
    "customer_kpi_output.csv": "Customer KPIs",
    "location_kpi_output.csv": "Location KPIs",
    "state_kpi_output.csv": "State KPIs",
    "monthly_kpi_output.csv": "Monthly KPIs",
    "category_kpi_output.csv": "Customer Category KPIs",
    "item_class_kpi_output.csv": "Item Class KPIs",
    "item_type_kpi_output.csv": "Item Type KPIs",
    "service_center_kpi_output.csv": "Service Center KPIs"
}

# --- Custom Styling for a Cleaner Look and Full Width ---
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        background-color: #f0f2f6; /* Light gray background */
        color: #333333;
    }

    /* Target all common Streamlit container elements to force full width */
    .main, [data-testid="stAppViewContainer"], .block-container,
    .st-emotion-cache-1pxn4kn, .st-emotion-cache-z5fcl4,
    div[data-testid="stVerticalBlock"] > div:first-child,
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"],
    .st-emotion-cache-zt5awu, /* Common top-level container */
    .st-emotion-cache-1fm77rp, /* Another potential main container */
    .st-emotion-cache-1oe5f0g, /* Yet another container type */
    div.st-emotion-cache-j7qwjs { /* More aggressive targeting */
        max-width: none !important; /* Remove any max-width constraint */
        padding-left: 0 !important;  /* Remove default padding */
        padding-right: 0 !important; /* Remove default padding */
        width: 100% !important;     /* Ensure it takes full width */
    }
    
    /* Adjust padding for the content within the now full-width main area, if desired */
    /* This creates internal spacing without constricting the overall width */
    .block-container {
        padding-left: 3rem !important; /* Add desired left padding */
        padding-right: 3rem !important; /* Add desired right padding */
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        margin-left: auto; /* Re-center if you do set a max-width, otherwise not strictly needed with 100% width */
        margin-right: auto;
    }

    /* Sidebar width - can still be controlled */
    .st-emotion-cache-z5fcl4 { 
        width: 250px; /* Example fixed width for sidebar */
    }

    /* Button styling */
    .st-emotion-cache-h4xjwx {
        background-color: #007aff; /* Apple Blue */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: background-color 0.2s, box-shadow 0.2s;
    }
    .st-emotion-cache-h4xjwx:hover {
        background-color: #005bb5;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Metric container */
    .st-emotion-cache-1v0mbdj {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem;
    }
    
    /* Metric value */
    .st-emotion-cache-1xv01z2 { 
        color: #007aff; /* Apple Blue for values */
        font-size: 2.2em;
        font-weight: 700;
    }
    
    /* Selectbox / Input field */
    .st-emotion-cache-czk5ad {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        padding: 0.5rem;
    }

    /* Expander styling */
    .st-emotion-cache-p2fwss { /* Expander header */
        border-radius: 8px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-weight: 600;
        color: #333333;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-p2fwss:hover {
        background-color: #f9f9f9;
    }
    .st-emotion-cache-p2fwss > div { /* Adjust padding inside expander header */
        padding: 0 !important; 
    }
    .st-emotion-cache-1aw8d9y { /* Expander content area */
        border: 1px solid #e0e0e0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        background-color: #ffffff;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* Tabs styling */
    .st-emotion-cache-l9bibm { /* Tab buttons container */
        background-color: #ffffff;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-1cpx6a9 { /* Individual tab button */
        border-radius: 6px;
        font-weight: 500;
    }
    .st-emotion-cache-1cpx6a9.st-emotion-cache-1cpx6a9-hover { /* Hover state for tab */
        background-color: #e5e5ea;
    }
    .st-emotion-cache-1cpx6a9.st-emotion-cache-1cpx6a9-selected { /* Selected tab */
        background-color: #007aff !important;
        color: white !important;
    }
    .st-emotion-cache-1q1n064 { /* Tab content area */
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a;
        font-weight: 700;
    }
    h1 { font-size: 2.5em; }
    h2 { font-size: 2em; }
    h3 { font-size: 1.5em; }

</style>
""", unsafe_allow_html=True)


st.title("Comprehensive Inventory & Warranty Analytics")

# --- Consolidated App Setup & Data Processing (Initially Collapsed) ---
with st.expander("App Setup & Data Processing", expanded=True): # Keep expanded for ease of use during debugging
    st.header("1. Data Upload & Cleaning")
    st.write("Upload your `warranty_raw.csv` file here to begin the analysis.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Please upload your raw warranty data in CSV format.")

    if uploaded_file is not None:
        # Save the uploaded file to a known path
        input_file_path = "uploaded_warranty_raw.csv"
        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # Load and clean the data
        with st.spinner("Cleaning and processing raw data..."):
            st.session_state.cleaned_df = load_and_clean(input_file_path)
        
        if not st.session_state.cleaned_df.empty:
            st.success("Data cleaned and ready for analysis! Preview below.")
            st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
        else:
            st.error("Data cleaning resulted in an empty DataFrame. Please check your raw data and ensure it's correctly formatted.")
    else:
        st.info("No file uploaded yet. Please upload your `warranty_raw.csv` to proceed.")

    st.header("2. Generate KPI Reports & Safety Stock")

    # Update as_of_datetime in session state whenever the date input changes
    # Use the initial value from session state if it exists, otherwise datetime.now()
    as_of_date_input = st.date_input(
        "Analysis 'As Of' Date",
        value=st.session_state.as_of_datetime.date(), # Use date() for initial value
        help="Set the date for 'Year-to-Date' and 'Last 90 Days' calculations."
    )
    # Only update session_state.as_of_datetime if the date input has actually changed
    if as_of_date_input != st.session_state.as_of_datetime.date():
        st.session_state.as_of_datetime = datetime.combine(as_of_date_input, datetime.min.time())


    if st.button("Run All Analysis & Generate Reports", help="Click to generate all KPI tables and safety stock recommendations."):
        if st.session_state.cleaned_df.empty:
            st.error("Please upload and clean data first before running analysis.")
        else:
            # Debugging check for as_of_datetime in session state
            if 'as_of_datetime' in st.session_state:
                print(f"DEBUG: as_of_datetime from session state IS defined: {st.session_state.as_of_datetime}")
            else:
                st.error("CRITICAL ERROR: 'as_of_datetime' is not in session state. This indicates a script execution issue. Please restart your Colab runtime completely.")
                print("DEBUG: as_of_datetime is NOT in session state at the point of button click.")
                st.stop()


            with st.spinner("Generating all KPI reports and Safety Stock data... This may take a few moments."):
                # Generate Safety Stock Table
                safety_stock_df = generate_safety_stock_table(
                    cleaned_df=st.session_state.cleaned_df,
                    safety_csv_path=list(output_paths.keys())[0],
                    as_of=st.session_state.as_of_datetime # Use from session state
                )
                st.session_state.generated_dfs['safety_stock_output.csv'] = safety_stock_df

                # Generate other KPI Tables
                st.session_state.generated_dfs['sku_kpi_output.csv'] = generate_sku_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[1], as_of=st.session_state.as_of_datetime)
                st.session_state.generated_dfs['customer_kpi_output.csv'] = generate_customer_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[2], as_of=st.session_state.as_of_datetime)
                st.session_state.generated_dfs['location_kpi_output.csv'] = generate_location_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[3], as_of=st.session_state.as_of_datetime)
                st.session_state.generated_dfs['state_kpi_output.csv'] = generate_state_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[4], as_of=st.session_state.as_of_datetime)
                st.session_state.generated_dfs['monthly_kpi_output.csv'] = generate_monthly_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[5], as_of=st.session_state.as_of_datetime)
                st.session_state.generated_dfs['category_kpi_output.csv'] = generate_category_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[6])
                st.session_state.generated_dfs['item_class_kpi_output.csv'] = generate_item_class_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[7])
                st.session_state.generated_dfs['item_type_kpi_output.csv'] = generate_item_type_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[8])
                st.session_state.generated_dfs['service_center_kpi_output.csv'] = generate_service_center_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[9])
                
            st.success("Analysis complete! Proceed to view results in the tabs below.")


# --- Display Results & Reports (Only if data generated) ---
if st.session_state.get('generated_dfs') and not st.session_state.generated_dfs.get('safety_stock_output.csv', pd.DataFrame()).empty:
    st.header("4. Analysis Results & Reports")

    tab_executive_summary, tab_sku_summary, tab_customer_summary, tab_state_summary, tab_location_summary, tab_item_class_summary, tab_item_type_summary, tab_service_center, tab_time_trends, tab_cost_flow_analysis, tab_detailed_kpis = st.tabs(
        ["Executive Summary", "SKU Summary", "Customer Summary", "State Summary", "Location Summary", "Item Class Summary", "Item Type Summary", "Service Center Analysis", "Time Trends", "Cost Flow Analysis", "Detailed KPI Reports"]
    )

    # --- Common KPI Calculation and Display Functions (Moved to global scope for reuse) ---
    def calculate_summary_kpis(df_slice):
        if df_slice.empty:
            return {
                "total_net_cost": 0,
                "total_claims": 0,
                "avg_cost_per_claim": 0,
                "unique_customers": 0,
                "total_units": 0
            }
        
        total_net_cost = df_slice['net_cost_impact'].sum()
        total_claims = df_slice['doc_num'].nunique() # Count unique document numbers
        avg_cost_per_claim = total_net_cost / total_claims if total_claims > 0 else 0
        unique_customers = df_slice['customer_name'].nunique()
        total_units = df_slice['item_qty'].sum()
        
        return {
            "total_net_cost": total_net_cost,
            "total_claims": total_claims,
            "avg_cost_per_claim": avg_cost_per_claim,
            "unique_customers": unique_customers,
            "total_units": total_units
        }

    def get_data_for_period(df, start_date, end_date):
        if start_date is None and end_date is None:
            return df
        return df[(df['txn_date'] >= start_date) & (df['txn_date'] <= end_date)].copy()

    def display_metric_with_trend(column_obj, label, current_value, previous_value, value_format="{:,.2f}", is_cost_metric=False):
        delta = current_value - previous_value if previous_value is not None else None
        delta_str = f"{delta:,.2f}" if delta is not None else None
        
        delta_color_mode = "off"
        if delta is not None:
            if is_cost_metric:
                delta_color_mode = "inverse" if delta > 0 else "normal" # Red if cost increased, Green if decreased
            else:
                delta_color_mode = "normal" if delta > 0 else "inverse" # Green if increased, Red if decreased
        
        column_obj.metric(
            label=label,
            value=value_format.format(current_value),
            delta=delta_str,
            delta_color=delta_color_mode
        )

    # Time Period Definitions relative to as_of_dt (Moved to global scope)
    as_of_dt = st.session_state.as_of_datetime # Ensure this is referenced correctly
    periods = {
        "30 Days": {"current": (as_of_dt - timedelta(days=30), as_of_dt), "prev": (as_of_dt - timedelta(days=60), as_of_dt - timedelta(days=30))},
        "90 Days": {"current": (as_of_dt - timedelta(days=90), as_of_dt), "prev": (as_of_dt - timedelta(days=180), as_of_dt - timedelta(days=90))},
        "YTD": {"current": (as_of_dt.replace(month=1, day=1), as_of_dt), "prev": ((as_of_dt - timedelta(days=365)).replace(month=1, day=1), (as_of_dt - timedelta(days=365)))},
        "2 Yrs": {"current": (as_of_dt - timedelta(days=2*365), as_of_dt), "prev": (as_of_dt - timedelta(days=4*365), as_of_dt - timedelta(days=2*365))},
        "3 Yrs": {"current": (as_of_dt - timedelta(days=3*365), as_of_dt), "prev": (as_of_dt - timedelta(days=6*365), as_of_dt - timedelta(days=3*365))}
    }

    kpi_metrics = { # Moved to global scope
        "Total Net Cost": {"key": "total_net_cost", "format": "${:,.2f}", "is_cost": True},
        "Total Claims": {"key": "total_claims", "format": "{:,.0f}", "is_cost": False},
        "Avg Cost/Claim": {"key": "avg_cost_per_claim", "format": "${:,.2f}", "is_cost": True},
        "Unique Customers": {"key": "unique_customers", "format": "{:,.0f}", "is_cost": False},
        "Total Units": {"key": "total_units", "format": "{:,.0f}", "is_cost": False}
    }

    # Helper function to display leaderboards
    def _display_leaderboard(df, title, sort_col, display_cols, ascending=False, top_n=10):
        st.markdown(f"##### {title}")
        if df.empty or sort_col not in df.columns or not display_cols:
            st.info("Not enough data to display this leaderboard.")
            return

        leaderboard_df = df.copy()
        if sort_col in leaderboard_df.columns:
            # Ensure sort_col is numeric if it contains mixed types for sorting
            # Or convert to string if sorting needs to be lexical on mixed types
            try:
                leaderboard_df[sort_col] = pd.to_numeric(leaderboard_df[sort_col], errors='coerce')
                leaderboard_df = leaderboard_df.dropna(subset=[sort_col]) # Drop rows where sort_col became NaN
            except Exception:
                pass # If not numeric, keep as is and let sort handle strings

            leaderboard_df = leaderboard_df.sort_values(by=sort_col, ascending=ascending).head(top_n)
        
        if not leaderboard_df.empty:
            # Ensure display_cols exist before trying to display them
            display_cols_present = [col for col in display_cols if col in leaderboard_df.columns]
            if display_cols_present:
                st.dataframe(leaderboard_df[display_cols_present].reset_index(drop=True), use_container_width=True)
            else:
                st.info("No valid display columns found for this leaderboard.")
        else:
            st.info("No data to display after sorting/filtering for this leaderboard.")


    with tab_executive_summary:
        st.subheader("Overall Business Performance: Executive Summary")
        st.info("Key Performance Indicators across different time horizons, with trend indicators.")

        cleaned_df_summary = st.session_state.cleaned_df
        
        for period_label, dates in periods.items():
            st.markdown(f"#### {period_label} Performance")
            cols = st.columns(len(kpi_metrics))
            
            current_df = get_data_for_period(cleaned_df_summary, *dates["current"])
            previous_df = get_data_for_period(cleaned_df_summary, *dates["prev"])

            current_kpis = calculate_summary_kpis(current_df)
            previous_kpis = calculate_summary_kpis(previous_df)

            for i, (metric_display, metric_info) in enumerate(kpi_metrics.items()):
                display_metric_with_trend(
                    cols[i],
                    metric_display,
                    current_kpis[metric_info["key"]],
                    previous_kpis[metric_info["key"]],
                    metric_info["format"],
                    metric_info["is_cost"]
                )
            st.markdown("---") # Separator


    with tab_sku_summary:
        st.subheader("SKU Performance: Detailed Summary")
        st.info("Select an SKU to view its key performance indicators and trends across different time horizons, or explore the leaderboards.")

        sku_kpi_df = st.session_state.generated_dfs.get('sku_kpi_output.csv')
        
        if sku_kpi_df is None or sku_kpi_df.empty:
            st.warning("SKU KPI data not available. Please upload and process data in the 'App Setup & Data Processing' tab.")
        else:
            st.markdown("### SKU Leaderboards")
            
            _display_leaderboard(
                sku_kpi_df,
                "Top 10 Underperforming SKUs (by Total Cost)",
                "total_sku_cost",
                ["item_sku", "total_sku_cost", "sku_transaction_count"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                sku_kpi_df,
                "Top 10 SKUs by Loss Per Touchpoint",
                "loss_per_touchpoint",
                ["item_sku", "loss_per_touchpoint", "sku_transaction_count"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                sku_kpi_df,
                "Top 10 SKUs by Customer Pain Index",
                "customer_pain_index",
                ["item_sku", "customer_pain_index", "sku_customer_recurrence"],
                ascending=False,
                top_n=10
            )


            st.markdown("### Individual SKU Performance")
            cleaned_df_sku_summary = st.session_state.cleaned_df # Original cleaned_df for detailed KPI calculation
            all_skus = ['Select an SKU'] + sorted([str(x) for x in cleaned_df_sku_summary['item_sku'].unique().tolist()])
            selected_sku = st.selectbox("Choose an Item SKU for detailed view", all_skus, key="sku_summary_selector")

            if selected_sku != 'Select an SKU':
                st.markdown(f"#### Performance for SKU: `{selected_sku}`")
                sku_df = cleaned_df_sku_summary[cleaned_df_sku_summary['item_sku'] == selected_sku].copy()

                if sku_df.empty:
                    st.info(f"No data found for SKU: `{selected_sku}`.")
                else:
                    for period_label, dates in periods.items():
                        st.markdown(f"##### {period_label} Performance - SKU: `{selected_sku}`")
                        cols = st.columns(len(kpi_metrics))
                        
                        current_df_sku_period = get_data_for_period(sku_df, *dates["current"])
                        previous_df_sku_period = get_data_for_period(sku_df, *dates["prev"])

                        current_kpis_sku = calculate_summary_kpis(current_df_sku_period)
                        previous_kpis_sku = calculate_summary_kpis(previous_df_sku_period)

                        for i, (metric_display, metric_info) in enumerate(kpi_metrics.items()):
                            display_metric_with_trend(
                                cols[i],
                                metric_display,
                                current_kpis_sku[metric_info["key"]],
                                previous_kpis_sku[metric_info["key"]],
                                metric_info["format"],
                                metric_info["is_cost"]
                            )
                        st.markdown("---")


    with tab_customer_summary:
        st.subheader("Customer Performance: Detailed Summary")
        st.info("Select a customer to view their key performance indicators and trends across different time horizons, or explore the leaderboards.")

        customer_kpi_df = st.session_state.generated_dfs.get('customer_kpi_output.csv')

        if customer_kpi_df is None or customer_kpi_df.empty:
            st.warning("Customer KPI data not available. Please upload and process data in the 'App Setup & Data Processing' tab.")
        else:
            st.markdown("### Customer Leaderboards")

            _display_leaderboard(
                customer_kpi_df,
                "Top 10 Customers by Total Cost",
                "total_customer_cost",
                ["customer_name", "total_customer_cost", "customer_transaction_count"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                customer_kpi_df,
                "Top 10 Customers by Bleed Score",
                "customer_bleed_score",
                ["customer_name", "customer_bleed_score", "customer_repeat_rate"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                customer_kpi_df,
                "Top 10 Customers by Repeat Claim Rate",
                "customer_repeat_rate",
                ["customer_name", "customer_repeat_rate", "customer_sku_diversity"],
                ascending=False,
                top_n=10
            )

            st.markdown("### Individual Customer Performance")
            cleaned_df_customer_summary = st.session_state.cleaned_df
            all_customers = ['Select a Customer'] + sorted([str(x) for x in cleaned_df_customer_summary['customer_name'].unique().tolist()])
            selected_customer = st.selectbox("Choose a Customer for detailed view", all_customers, key="customer_summary_selector")

            if selected_customer != 'Select a Customer':
                st.markdown(f"#### Performance for Customer: `{selected_customer}`")
                customer_df = cleaned_df_customer_summary[cleaned_df_customer_summary['customer_name'] == selected_customer].copy()

                if customer_df.empty:
                    st.info(f"No data found for Customer: `{selected_customer}`.")
                else:
                    for period_label, dates in periods.items():
                        st.markdown(f"##### {period_label} Performance - Customer: `{selected_customer}`")
                        cols = st.columns(len(kpi_metrics))
                        
                        current_df_customer_period = get_data_for_period(customer_df, *dates["current"])
                        previous_df_customer_period = get_data_for_period(customer_df, *dates["prev"])

                        current_kpis_customer = calculate_summary_kpis(current_df_customer_period)
                        previous_kpis_customer = calculate_summary_kpis(previous_df_customer_period)

                        for i, (metric_display, metric_info) in enumerate(kpi_metrics.items()):
                            display_metric_with_trend(
                                cols[i],
                                metric_display,
                                current_kpis_customer[metric_info["key"]],
                                previous_kpis_customer[metric_info["key"]],
                                metric_info["format"],
                                metric_info["is_cost"]
                            )
                        st.markdown("---")


    with tab_state_summary:
        st.subheader("State Performance: Detailed Summary")
        st.info("Select a state to view its key performance indicators and trends across different time horizons, or explore the leaderboards.")

        state_kpi_df = st.session_state.generated_dfs.get('state_kpi_output.csv')

        if state_kpi_df is None or state_kpi_df.empty:
            st.warning("State KPI data not available. Please upload and process data in the 'App Setup & Data Processing' tab.")
        else:
            st.markdown("### State Leaderboards")

            _display_leaderboard(
                state_kpi_df,
                "Top 10 States by Total Cost",
                "total_state_cost",
                ["ship_state", "total_state_cost", "state_transaction_count"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                state_kpi_df,
                "Top 10 States by Unit Failure Rate",
                "unit_failure_rate_by_state",
                ["ship_state", "unit_failure_rate_by_state", "state_transaction_count"],
                ascending=False,
                top_n=10
            )
            st.markdown("##### High Risk States (by Bleed Density)")
            high_risk_states = state_kpi_df[state_kpi_df['high_risk_state_flag'] == True].copy()
            if not high_risk_states.empty:
                st.dataframe(high_risk_states[["ship_state", "bleed_density_by_state"]].sort_values(by="bleed_density_by_state", ascending=False).reset_index(drop=True), use_container_width=True)
            else:
                st.info("No high-risk states identified.")
            
            _display_leaderboard(
                state_kpi_df,
                "Top 10 States by Repeat Claim Rate",
                "repeat_claim_rate_by_state",
                ["ship_state", "repeat_claim_rate_by_state", "state_customer_count"],
                ascending=False,
                top_n=10
            )

            st.markdown("### Individual State Performance")
            cleaned_df_state_summary = st.session_state.cleaned_df
            all_states = ['Select a State'] + sorted([str(x) for x in cleaned_df_state_summary['ship_state'].unique().tolist()])
            selected_state = st.selectbox("Choose a State for detailed view", all_states, key="state_summary_selector")

            if selected_state != 'Select a State':
                st.markdown(f"#### Performance for State: `{selected_state}`")
                state_df = cleaned_df_state_summary[cleaned_df_state_summary['ship_state'] == selected_state].copy()

                if state_df.empty:
                    st.info(f"No data found for State: `{selected_state}`.")
                else:
                    for period_label, dates in periods.items():
                        st.markdown(f"##### {period_label} Performance - State: `{selected_state}`")
                        cols = st.columns(len(kpi_metrics))
                        
                        current_df_state_period = get_data_for_period(state_df, *dates["current"])
                        previous_df_state_period = get_data_for_period(state_df, *dates["prev"])

                        current_kpis_state = calculate_summary_kpis(current_df_state_period)
                        previous_kpis_state = calculate_summary_kpis(previous_df_state_period)

                        for i, (metric_display, metric_info) in enumerate(kpi_metrics.items()):
                            display_metric_with_trend(
                                cols[i],
                                metric_display,
                                current_kpis_state[metric_info["key"]],
                                previous_kpis_state[metric_info["key"]],
                                metric_info["format"],
                                metric_info["is_cost"]
                            )
                        st.markdown("---")


    with tab_location_summary:
        st.subheader("Location Performance: Detailed Summary")
        st.info("Select a fulfillment location to view its key performance indicators and trends across different time horizons, or explore the leaderboards.")

        location_kpi_df = st.session_state.generated_dfs.get('location_kpi_output.csv')

        if location_kpi_df is None or location_kpi_df.empty:
            st.warning("Location KPI data not available. Please upload and process data in the 'App Setup & Data Processing' tab.")
        else:
            st.markdown("### Location Leaderboards")

            _display_leaderboard(
                location_kpi_df,
                "Top 10 Locations by Total Cost",
                "total_fulfillment_cost",
                ["fulfillment_loc", "total_fulfillment_cost", "total_units_fulfilled"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                location_kpi_df,
                "Top 10 Locations by Bleed Density",
                "bleed_density_by_loc",
                ["fulfillment_loc", "bleed_density_by_loc", "total_units_fulfilled"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                location_kpi_df,
                "Top 10 Locations by Average Delivery Risk Score",
                "avg_delivery_risk_score",
                ["fulfillment_loc", "avg_delivery_risk_score", "fulfillment_customer_footprint"],
                ascending=False,
                top_n=10
            )

            st.markdown("### Individual Location Performance")
            cleaned_df_location_summary = st.session_state.cleaned_df
            all_locations_summary = ['Select a Location'] + sorted([str(x) for x in cleaned_df_location_summary['fulfillment_loc'].unique().tolist()])
            selected_location = st.selectbox("Choose a Fulfillment Location for detailed view", all_locations_summary, key="location_summary_selector")

            if selected_location != 'Select a Location':
                st.markdown(f"#### Performance for Location: `{selected_location}`")
                location_df = cleaned_df_location_summary[cleaned_df_location_summary['fulfillment_loc'] == selected_location].copy()

                if location_df.empty:
                    st.info(f"No data found for Location: `{selected_location}`.")
                else:
                    for period_label, dates in periods.items():
                        st.markdown(f"##### {period_label} Performance - Location: `{selected_location}`")
                        cols = st.columns(len(kpi_metrics))
                        
                        current_df_location_period = get_data_for_period(location_df, *dates["current"])
                        previous_df_location_period = get_data_for_period(location_df, *dates["prev"])

                        current_kpis_location = calculate_summary_kpis(current_df_location_period)
                        previous_kpis_location = calculate_summary_kpis(previous_df_location_period)

                        for i, (metric_display, metric_info) in enumerate(kpi_metrics.items()):
                            display_metric_with_trend(
                                cols[i],
                                metric_display,
                                current_kpis_location[metric_info["key"]],
                                previous_kpis_location[metric_info["key"]],
                                metric_info["format"],
                                metric_info["is_cost"]
                            )
                        st.markdown("---")


    with tab_item_class_summary:
        st.subheader("Item Class Performance: Detailed Summary")
        st.info("Select an item class to view its key performance indicators and trends across different time horizons, or explore the leaderboards.")

        item_class_kpi_df = st.session_state.generated_dfs.get('item_class_kpi_output.csv')

        if item_class_kpi_df is None or item_class_kpi_df.empty:
            st.warning("Item Class KPI data not available. Please upload and process data in the 'App Setup & Data Processing' tab.")
        else:
            st.markdown("### Item Class Leaderboards")

            _display_leaderboard(
                item_class_kpi_df,
                "Top 10 Item Classes by Total Cost",
                "item_class_total_cost",
                ["item_class", "item_class_total_cost", "item_class_transaction_count"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                item_class_kpi_df,
                "Top 10 Item Classes by Unit Failure Rate",
                "item_class_unit_failure_rate",
                ["item_class", "item_class_unit_failure_rate", "item_class_transaction_count"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                item_class_kpi_df,
                "Top 10 Item Classes by Bleed Density",
                "item_class_bleed_density",
                ["item_class", "item_class_bleed_density", "item_class_transaction_count"],
                ascending=False,
                top_n=10
            )
            st.markdown("##### Top Bleed SKUs per Item Class")
            # FIX: Reset index BEFORE selecting columns to ensure 'item_class' is a column
            if 'item_class_top_bleed_sku' in item_class_kpi_df.columns:
                temp_item_class_df = item_class_kpi_df.reset_index()
                st.dataframe(temp_item_class_df[["item_class", "item_class_top_bleed_sku"]], use_container_width=True)
            else:
                st.info("Item class top bleed SKU data not available or 'item_class_top_bleed_sku' column is missing.")

            st.markdown("### Individual Item Class Performance")
            cleaned_df_item_class_summary = st.session_state.cleaned_df
            all_item_classes_summary = ['Select an Item Class'] + sorted([str(x) for x in cleaned_df_item_class_summary['item_class'].unique().tolist()])
            selected_item_class = st.selectbox("Choose an Item Class for detailed view", all_item_classes_summary, key="item_class_summary_selector")

            if selected_item_class != 'Select an Item Class':
                st.markdown(f"#### Performance for Item Class: `{selected_item_class}`")
                item_class_df = cleaned_df_item_class_summary[cleaned_df_item_class_summary['item_class'] == selected_item_class].copy()

                if item_class_df.empty:
                    st.info(f"No data found for Item Class: `{selected_item_class}`.")
                else:
                    for period_label, dates in periods.items():
                        st.markdown(f"##### {period_label} Performance - Item Class: `{selected_item_class}`")
                        cols = st.columns(len(kpi_metrics))
                        
                        current_df_item_class_period = get_data_for_period(item_class_df, *dates["current"])
                        previous_df_item_class_period = get_data_for_period(item_class_df, *dates["prev"])

                        current_kpis_item_class = calculate_summary_kpis(current_df_item_class_period)
                        previous_kpis_item_class = calculate_summary_kpis(previous_df_item_class_period)

                        for i, (metric_display, metric_info) in enumerate(kpi_metrics.items()):
                            display_metric_with_trend(
                                cols[i],
                                metric_display,
                                current_kpis_item_class[metric_info["key"]],
                                previous_kpis_item_class[metric_info["key"]],
                                metric_info["format"],
                                metric_info["is_cost"]
                            )
                        st.markdown("---")


    with tab_item_type_summary:
        st.subheader("Item Type Performance: Detailed Summary")
        st.info("Select an item type to view its key performance indicators and trends across different time horizons, or explore the leaderboards.")

        item_type_kpi_df = st.session_state.generated_dfs.get('item_type_kpi_output.csv')

        if item_type_kpi_df is None or item_type_kpi_df.empty:
            st.warning("Item Type KPI data not available. Please upload and process data in the 'App Setup & Data Processing' tab.")
        else:
            st.markdown("### Item Type Leaderboards")

            _display_leaderboard(
                item_type_kpi_df,
                "Top 10 Item Types by Total Cost",
                "item_type_total_cost",
                ["item_type", "item_type_total_cost", "item_type_transaction_count"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                item_type_kpi_df,
                "Top 10 Item Types by Unit Failure Rate",
                "item_type_unit_failure_rate",
                ["item_type", "item_type_unit_failure_rate", "item_type_transaction_count"],
                ascending=False,
                top_n=10
            )
            _display_leaderboard(
                item_type_kpi_df,
                "Top 10 Item Types by Bleed Density",
                "item_type_bleed_density",
                ["item_type", "item_type_bleed_density", "item_type_transaction_count"],
                ascending=False,
                top_n=10
            )
            st.markdown("##### Top Bleed SKUs per Item Type")
            # FIX: Reset index BEFORE selecting columns to ensure 'item_type' is a column
            if 'item_type_top_bleed_sku' in item_type_kpi_df.columns:
                temp_item_type_df = item_type_kpi_df.reset_index()
                st.dataframe(temp_item_type_df[["item_type", "item_type_top_bleed_sku"]], use_container_width=True)
            else:
                st.info("Item type top bleed SKU data not available or 'item_type_top_bleed_sku' column is missing.")

            st.markdown("### Individual Item Type Performance")
            cleaned_df_item_type_summary = st.session_state.cleaned_df
            all_item_types_summary = ['Select an Item Type'] + sorted([str(x) for x in cleaned_df_item_type_summary['item_type'].unique().tolist()])
            selected_item_type = st.selectbox("Choose an Item Type for detailed view", all_item_types_summary, key="item_type_summary_selector")

            if selected_item_type != 'Select an Item Type':
                st.markdown(f"#### Performance for Item Type: `{selected_item_type}`")
                item_type_df = cleaned_df_item_type_summary[cleaned_df_item_class_summary['item_type'] == selected_item_type].copy()

                if item_type_df.empty:
                    st.info(f"No data found for Item Type: `{selected_item_type}`.")
                else:
                    for period_label, dates in periods.items():
                        st.markdown(f"##### {period_label} Performance - Item Type: `{selected_item_type}`")
                        cols = st.columns(len(kpi_metrics))
                        
                        current_df_item_type_period = get_data_for_period(item_type_df, *dates["current"])
                        previous_df_item_type_period = get_data_for_period(item_type_df, *dates["prev"])

                        current_kpis_item_type = calculate_summary_kpis(current_df_item_type_period)
                        previous_kpis_item_type = calculate_summary_kpis(previous_df_item_type_period)

                        for i, (metric_display, metric_info) in enumerate(kpi_metrics.items()):
                            display_metric_with_trend(
                                cols[i],
                                metric_display,
                                current_kpis_item_type[metric_info["key"]],
                                previous_kpis_item_type[metric_info["key"]],
                                metric_info["format"],
                                metric_info["is_cost"]
                            )
                        st.markdown("---")


    with tab_service_center:
        st.subheader("Service Center Analysis")
        service_center_df = st.session_state.generated_dfs.get('service_center_kpi_output.csv')
        safety_stock_results_df = st.session_state.generated_dfs.get('safety_stock_output.csv') # Need for safety stock subtabs

        # Nested tabs for different analyses within Service Center
        tab_sc_heatmap, tab_sc_geo, tab_sc_ss_charts, tab_sc_ss_table = st.tabs(
            ["Service Center KPI Heatmap", "Geographic Heatmaps", "Safety Stock Charts", "Detailed Safety Stock Table"]
        )

        with tab_sc_heatmap:
            if service_center_df is not None and not service_center_df.empty:
                st.write("#### Service Center Performance Heatmap")
                st.info("This heatmap shows the relative intensity of various KPIs for each Service Center (Customer Name). Brighter colors indicate higher values after normalization.")

                # Debugging: Show columns of the service_center_df
                with st.expander("Debug Service Center Data Columns", expanded=False):
                    st.write("Columns in service_center_df:", service_center_df.columns.tolist())
                    st.write("Head of service_center_df:", service_center_df.head())

                # The generate_service_center_kpi_table function returns customer_name as index
                # Ensure 'customer_name' is a column for filtering/indexing by resetting index if it's the index
                if 'customer_name' not in service_center_df.columns and service_center_df.index.name == 'customer_name':
                    service_center_df = service_center_df.reset_index()


                # Select numerical columns for the heatmap
                potential_heatmap_kpis = [
                    'total_service_center_cost',
                    'total_labor_cost',
                    'total_mileage_cost',
                    'avg_cost_per_visit',
                    'avg_units_serviced',
                    'bleed_per_unit',
                    'avg_margin_loss_per_visit',
                    'service_center_sku_diversity',
                    'service_center_customer_repeats'
                ]
                
                # Filter heatmap_kpis to only include columns actually present in service_center_df
                available_heatmap_kpis = [col for col in potential_heatmap_kpis if col in service_center_df.columns]

                # Crucial check: Ensure 'customer_name' exists and there's at least one KPI for the heatmap
                if 'customer_name' not in service_center_df.columns:
                    st.warning("Heatmap: 'customer_name' column not found in service center data. Cannot generate heatmap.")
                elif not available_heatmap_kpis:
                    st.warning("Heatmap: No relevant KPI columns found in service center data for the heatmap. Please check your data or selected KPIs.")
                else:
                    # Filter for only relevant KPI columns and drop rows with all NaN for selected KPIs
                    # Also ensure customer_name is not NaN
                    heatmap_df = service_center_df[['customer_name'] + available_heatmap_kpis].copy()
                    heatmap_df = heatmap_df.dropna(subset=available_heatmap_kpis, how='all')
                    heatmap_df = heatmap_df.dropna(subset=['customer_name']) # Drop rows where customer_name is NaN

                    if not heatmap_df.empty:
                        # Set customer_name as index for easier matrix formation
                        heatmap_df = heatmap_df.set_index('customer_name')
                        
                        # Handle infinities and large numbers, replace with NaN then impute or drop
                        # Also convert columns to numeric, coercing errors
                        for col in heatmap_df.columns:
                            if pd.api.types.is_numeric_dtype(heatmap_df[col]):
                                heatmap_df[col] = pd.to_numeric(heatmap_df[col], errors='coerce')
                                heatmap_df[col] = heatmap_df[col].replace([np.inf, -np.inf], np.nan)
                        
                        # Fill remaining NaNs with 0 for heatmap visualization (or mean/median if preferred)
                        # This is important after type conversion and inf handling
                        heatmap_df = heatmap_df.fillna(0)

                        # Check if all numeric columns are now zero (can happen after fillna)
                        if (heatmap_df[available_heatmap_kpis].sum().sum() == 0) and not heatmap_df.empty:
                            st.warning("Heatmap: All selected KPI values are zero after processing. Heatmap may not be meaningful.")
                        
                        # Only proceed with scaling if there's actual non-zero data to scale
                        # Also check for division by zero in scaler (if a column is all zeros, it might cause issues)
                        # We specifically filter to ensure the columns exist in heatmap_df before scaling.
                        cols_to_scale = [col for col in available_heatmap_kpis if col in heatmap_df.columns and heatmap_df[col].nunique() > 1]
                        
                        if not heatmap_df.empty and cols_to_scale: # Check if there are columns to scale
                            scaler = MinMaxScaler()
                            # Scale only the columns that have variance
                            heatmap_df[cols_to_scale] = scaler.fit_transform(heatmap_df[cols_to_scale])
                            
                            # For columns that were all zeros and thus not scaled, ensure they are still zero or handled
                            for col in available_heatmap_kpis:
                                if col not in cols_to_scale:
                                    if col in heatmap_df.columns:
                                        heatmap_df[col] = 0 # Ensure they remain zero if they were already zero

                            fig_heatmap = px.imshow(
                                heatmap_df[available_heatmap_kpis], # Plot only the available and (potentially) scaled KPIs
                                x=available_heatmap_kpis, # X-axis uses the available KPI names
                                y=heatmap_df.index,
                                color_continuous_scale=px.colors.sequential.Viridis, # A good sequential color scale
                                title="Normalized Service Center KPI Heatmap",
                                labels={
                                    "x": "KPI",
                                    "y": "Service Center (Customer Name)",
                                    "color": "Normalized Value"
                                }
                            )
                            
                            fig_heatmap.update_xaxes(side="top") # Move KPI labels to top
                            fig_heatmap.update_layout(height=max(600, len(heatmap_df.index) * 20), width=900) # Dynamic height based on number of customers
                            
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        elif not heatmap_df.empty:
                            st.warning("Heatmap: All selected KPI columns have no variance (all values are the same or zero). Heatmap may not be meaningful as normalization cannot be applied effectively.")
                        else:
                            st.warning("No valid data available for the Service Center Performance Heatmap after cleaning and filtering.")
                    else:
                        st.warning("No valid data available for the Service Center Performance Heatmap after cleaning and filtering.")
            else:
                st.info("Service Center KPI data not available or is empty. Run analysis first.")

        with tab_sc_geo:
            st.write("#### Geographic Service Center Impact")
            st.info("Visualizing key metrics by US State and Canadian Province for different time periods.")

            if not st.session_state.cleaned_df.empty:
                # Define metric options for geographic maps
                geo_metric_options = {
                    "Total Net Cost Impact ($)": "net_cost_impact",
                    "Total Units Transacted": "item_qty",
                    "Average Refund Per Unit ($)": "refund_per_unit",
                    "Average Margin Loss Intensity": "margin_loss_intensity",
                    "Total Labor Cost ($)": "total_labor_cost",  # New metric
                    "Total Mileage Cost ($)": "total_mileage_cost" # New metric
                }
                selected_geo_metric_display = st.selectbox(
                    "Select Metric for Geographic Heatmap",
                    list(geo_metric_options.keys()),
                    key="geo_metric_selector_tab_sc_geo" # Unique key
                )
                selected_geo_metric_col = geo_metric_options[selected_geo_metric_display]

                # Define time period options for geographic maps
                time_period_options = ["Total", "Year-to-Date", "Last 90 Days", "Last 30 Days"]
                selected_time_period = st.selectbox(
                    "Select Time Period for Geographic Heatmap",
                    time_period_options,
                    key="geo_time_period_selector_tab_sc_geo" # Unique key
                )

                # Helper function to prepare geographic data based on selections
                def prepare_filtered_geo_data(df, metric_col, time_period_str, as_of_dt):
                    filtered_df = df.copy()

                    # Apply time period filter
                    if time_period_str == "Year-to-Date":
                        filtered_df = filtered_df[filtered_df['txn_date'] >= as_of_dt.replace(month=1, day=1)]
                    elif time_period_str == "Last 90 Days":
                        filtered_df = filtered_df[filtered_df['txn_date'] >= as_of_dt - timedelta(days=90)]
                    elif time_period_str == "Last 30 Days":
                        filtered_df = filtered_df[filtered_df['txn_date'] >= as_of_dt - timedelta(days=30)]
                    # "Total" requires no filtering

                    if filtered_df.empty:
                        return pd.DataFrame()

                    # Map fulfillment_loc to a more explicit country name
                    filtered_df['country'] = filtered_df['fulfillment_loc'].apply(lambda x: 'United States' if x.startswith('H') else 'Canada')

                    # Standardize Canadian province names for Plotly
                    # This mapping covers common abbreviations to full names for Plotly's internal maps
                    canada_province_mapping = {
                        'AB': 'Alberta', 'BC': 'British Columbia', 'MB': 'Manitoba', 'NB': 'New Brunswick',
                        'NL': 'Newfoundland and Labrador', 'NS': 'Nova Scotia', 'ON': 'Ontario',
                        'PE': 'Prince Edward Island', 'QC': 'Quebec', 'SK': 'Saskatchewan',
                        'NT': 'Northwest Territories', 'NU': 'Nunavut', 'YT': 'Yukon'
                    }
                    filtered_df.loc[filtered_df['country'] == 'Canada', 'ship_state'] = \
                        filtered_df.loc[filtered_df['country'] == 'Canada', 'ship_state'].replace(canada_province_mapping)

                    # Aggregate by ship_state and country for the selected metric
                    if metric_col in ['net_cost_impact', 'item_qty', 'refund_per_unit', 'margin_loss_intensity']:
                        # For sums or averages
                        if metric_col in ['net_cost_impact', 'item_qty']:
                            geo_data = filtered_df.groupby(['ship_state', 'country'])[metric_col].sum().reset_index()
                        else: # refund_per_unit, margin_loss_intensity
                            geo_data = filtered_df.groupby(['ship_state', 'country'])[metric_col].mean().replace([np.inf, -np.inf], np.nan).fillna(0).reset_index()
                    elif metric_col == "total_labor_cost":
                        labor_df = filtered_df[filtered_df["item_sku"] == "SCLAB"]
                        if not labor_df.empty:
                            geo_data = labor_df.groupby(['ship_state', 'country'])['net_cost_impact'].sum().reset_index()
                        else:
                            geo_data = pd.DataFrame(columns=['ship_state', 'country', 'net_cost_impact']) # Return empty if no data
                    elif metric_col == "total_mileage_cost":
                        mileage_df = filtered_df[filtered_df["item_sku"] == "SCMIL"]
                        if not mileage_df.empty:
                            geo_data = mileage_df.groupby(['ship_state', 'country'])['net_cost_impact'].sum().reset_index()
                        else:
                            geo_data = pd.DataFrame(columns=['ship_state', 'country', 'net_cost_impact']) # Return empty if no data
                    else:
                        st.warning(f"Unsupported metric for geographic heatmap: {metric_col}")
                        return pd.DataFrame()

                    geo_data.columns = ['state_province', 'country', 'metric_value']
                    return geo_data

                geo_data = prepare_filtered_geo_data(
                    st.session_state.cleaned_df,
                    selected_geo_metric_col,
                    selected_time_period,
                    st.session_state.as_of_datetime
                )

                if not geo_data.empty:
                    # --- US Map ---
                    us_data = geo_data[geo_data['country'] == 'United States'].copy()
                    if not us_data.empty and 'state_province' in us_data.columns and 'metric_value' in us_data.columns:
                        st.write(f"#### US States - {selected_geo_metric_display} ({selected_time_period})")
                        fig_us_map = px.choropleth(
                            us_data,
                            locations="state_province",
                            locationmode="USA-states", # This tells Plotly to use US state names/abbreviations
                            color="metric_value",
                            color_continuous_scale="Viridis",
                            scope="usa",
                            title=f"{selected_geo_metric_display} by US State",
                            labels={"metric_value": selected_geo_metric_display}
                        )
                        fig_us_map.update_layout(height=600)
                        st.plotly_chart(fig_us_map, use_container_width=True)
                    else:
                        st.info(f"No US state data available for {selected_geo_metric_display} in the {selected_time_period} period.")

                    # --- Canada Map ---
                    canada_data = geo_data[geo_data['country'] == 'Canada'].copy()
                    if not canada_data.empty and 'state_province' in canada_data.columns and 'metric_value' in canada_data.columns:
                        st.write(f"#### Canadian Provinces - {selected_geo_metric_display} ({selected_time_period})")
                        st.info("Debugging: Here's a preview of the Canadian data being used for the map. Please ensure 'state_province' values match 'properties.name' in the GeoJSON.")
                        st.dataframe(canada_data.head()) # FIX: Added for debugging Canadian map issues

                        fig_canada_map = px.choropleth(
                            canada_data,
                            locations="state_province",
                            locationmode="geojson-id", # Use geojson-id for custom boundaries
                            geojson="https://raw.githubusercontent.com/datasets/canada-provinces/master/data/canada_provinces.geojson", # Use a robust GeoJSON for Canadian provinces
                            featureidkey="properties.name", # Key in geojson features to match 'locations'
                            color="metric_value",
                            color_continuous_scale="Viridis",
                            scope="north america", # Use 'north america' scope, which covers Canada
                            title=f"{selected_geo_metric_display} by Canadian Province",
                            labels={"metric_value": selected_geo_metric_display}
                        )
                        fig_canada_map.update_geos(
                            fitbounds="locations", # Zoom to the locations provided by the data
                            visible=True # Ensure map is visible
                        )
                        fig_canada_map.update_layout(height=600)
                        st.plotly_chart(fig_canada_map, use_container_width=True)
                    else:
                        st.info(f"No Canadian province data available for {selected_geo_metric_display} in the {selected_time_period} period. Please check if 'ship_state' values match 'properties.name' in the GeoJSON.")
                else:
                    st.info("No geographic data available for the selected metric and time period after cleaning.")
            else:
                st.info("Cleaned data not available for geographic heatmap. Please upload and process data first.")

        with tab_sc_ss_charts: # New sub-tab for Safety Stock Charts
            st.subheader("Safety Stock Charts")
            if safety_stock_results_df is not None and not safety_stock_results_df.empty:
                st.write("#### Total Safety Stock Value by Item Class")
                ss_value_by_class = safety_stock_results_df.groupby('item_class')['total_safety_stock_value'].sum().sort_values(ascending=False)
                st.bar_chart(ss_value_by_class, use_container_width=True)

                st.write("#### Count of Items by Target Service Level (Z-score)")
                item_count_by_z = safety_stock_results_df['item_specific_z'].value_counts().sort_index()
                st.bar_chart(item_count_by_z, use_container_width=True)

                # Scatter Plot: Risk Score vs. Total Safety Stock Value
                if not safety_stock_results_df.empty and \
                'S_score' in safety_stock_results_df.columns and \
                'total_safety_stock_value' in safety_stock_results_df.columns and \
                'item_class' in safety_stock_results_df.columns and \
                'item_sku' in safety_stock_results_df.columns and \
                'fulfillment_loc' in safety_stock_results_df.columns and \
                'safety_stock_qty' in safety_stock_results_df.columns and \
                'Reorder_Point' in safety_stock_results_df.columns:
                    
                    st.write("#### Risk Score vs. Total Safety Stock Value by Item Class")
                    st.info("Hover over points for details. Adjust size based on Safety Stock Quantity.")
                    
                    plot_df_scatter = safety_stock_results_df.dropna(subset=['S_score', 'total_safety_stock_value', 'safety_stock_qty']).copy()
                    
                    if not plot_df_scatter.empty:
                        fig_scatter = px.scatter(
                            plot_df_scatter,
                            x="S_score",
                            y="total_safety_stock_value",
                            color="item_class", # Color by item class
                            hover_name="item_sku", # Show SKU on hover
                            hover_data=['fulfillment_loc', 'safety_stock_qty', 'Reorder_Point'], # Additional data on hover
                            size='safety_stock_qty', # Size bubbles by safety stock quantity
                            log_y=True, # Log scale for better visibility of cost differences
                            title="Safety Stock Risk (S_score) vs. Value (log scale)",
                            labels={
                                "S_score": "Composite Safety Score (Higher = More Risk)",
                                "total_safety_stock_value": "Total Safety Stock Value ($)",
                                "item_class": "Item Class"
                            }
                        )
                        fig_scatter.update_layout(height=500) # Set a fixed height for consistency
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.warning("No data available for the Risk Score vs. Total Safety Stock Value scatter plot after cleaning.")
                else:
                    st.info("Cannot generate Risk Score vs. Value scatter plot. Missing required columns or empty data after analysis.")
            else:
                st.info("Safety Stock data not available or is empty. Please run the analysis first.")

        with tab_sc_ss_table: # New sub-tab for Detailed Safety Stock Table
            st.subheader("Detailed Safety Stock Data")
            if safety_stock_results_df is not None and not safety_stock_results_df.empty:
                st.info("Filter the detailed safety stock data below using the search and dropdown options. The table is sorted by 'RRR: Service Part' item class first, then by 'S_score' (Safety Score) highest to lowest.")

                # Sorting logic for the safety stock table
                # 1. Prioritize 'RRR: Service Part'
                # 2. Then sort by 'S_score' (Safety Score) highest to lowest
                sorted_safety_stock_df = safety_stock_results_df.copy()
                
                # Create a boolean mask for 'RRR: Service Part'
                is_service_part = sorted_safety_stock_df['item_class'] == 'RRR: Service Part'
                
                # Sort by 'item_class' (service part first), then 'S_score' (descending)
                # Use a temporary key for sorting item_class to put 'RRR: Service Part' at the top
                sorted_safety_stock_df['sort_key_item_class'] = np.where(is_service_part, 0, 1)
                sorted_safety_stock_df = sorted_safety_stock_df.sort_values(
                    by=['sort_key_item_class', 'S_score'],
                    ascending=[True, False]
                ).drop(columns=['sort_key_item_class']) # Drop the temporary sort key

                # Filters for the table
                filter_cols_ss = ['item_sku', 'fulfillment_loc', 'item_class']
                current_filtered_df_ss = sorted_safety_stock_df.copy() # Use the newly sorted DF

                for col in filter_cols_ss:
                    if col == 'item_sku': # Text input for SKU search
                        search_term_ss = st.text_input(f"Search by {col} (partial match)", key=f"safety_stock_search_{col}_sc_tab") # Unique key
                        if search_term_ss:
                            current_filtered_df_ss = current_filtered_df_ss[
                                current_filtered_df_ss[col].astype(str).str.contains(search_term_ss, case=False, na=False)
                            ]
                    else: # Selectbox for other columns
                        # FIX: Convert all unique values to string for sorting to avoid TypeError
                        unique_values_ss = sorted_safety_stock_df[col].unique().tolist()
                        all_unique_values_ss = ['All'] + sorted([str(x) for x in unique_values_ss])

                        selected_value_ss = st.selectbox(f"Filter by {col}", all_unique_values_ss, key=f"safety_stock_filter_{col}_sc_tab") # Unique key
                        if selected_value_ss != 'All':
                            # Attempt to convert back to original type if unique values are numeric
                            if pd.api.types.is_numeric_dtype(sorted_safety_stock_df[col]):
                                try:
                                    selected_value_ss = pd.to_numeric(selected_value_ss)
                                except ValueError:
                                    pass # Keep as string if conversion fails
                            current_filtered_df_ss = current_filtered_df_ss[current_filtered_df_ss[col].astype(str) == selected_value_ss] # FIX: Convert column to str for comparison
                
                st.write(f"Displaying {len(current_filtered_df_ss)} of {len(sorted_safety_stock_df)} entries.")
                st.dataframe(current_filtered_df_ss, use_container_width=True)

                # Download button for Safety Stock data
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_data_ss = convert_df_to_csv(safety_stock_results_df)
                st.download_button(
                    label="Download Full Safety Stock Data as CSV",
                    data=csv_data_ss,
                    file_name="safety_stock_output.csv",
                    mime="text/csv",
                    key="download_safety_stock_full_sc_tab" # Unique key
                )
            else:
                st.info("Safety Stock data not available or is empty. Please run the analysis first.")


    with tab_time_trends:
        st.subheader("Monthly Key Performance Indicators (KPIs)")
        st.info("Observe trends in key metrics over time. Use the dropdown to select different KPIs.")

        monthly_kpi_df = st.session_state.generated_dfs.get('monthly_kpi_output.csv')

        if monthly_kpi_df is not None and not monthly_kpi_df.empty:
            st.info(f"DEBUG: Columns in monthly_kpi_df: {monthly_kpi_df.columns.tolist()}") # Debugging print
            # Check if 'month' column exists before processing
            if 'month' in monthly_kpi_df.columns:
                monthly_kpi_df['month'] = pd.to_datetime(monthly_kpi_df['month'])
                monthly_kpi_df = monthly_kpi_df.sort_values('month')

                time_trend_metric_options = {
                    "Monthly Total Cost ($)": "monthly_total_cost",
                    "Monthly Transaction Count": "monthly_transaction_count",
                    "Monthly GL Transaction Count": "monthly_gl_transaction_count",
                    "Monthly Credit Memo Count": "monthly_credit_memo_count",
                    "Monthly Invoice Count": "monthly_invoice_count",
                    "Monthly Total Units Transacted": "monthly_total_units",
                    "Monthly Avg Cost Per Unit ($)": "monthly_avg_cost_per_unit",
                    "Monthly Margin Bleed Rate": "monthly_margin_bleed_rate",
                    "Monthly Repeat Customers": "monthly_repeat_customers",
                    "Monthly High Risk State Events": "monthly_high_risk_state_events",
                    "Monthly Fulfillment Risk Score": "monthly_fulfillment_risk_score"
                }

                selected_time_trend_metric_display = st.selectbox(
                    "Select KPI for Time Trend",
                    list(time_trend_metric_options.keys()),
                    key="time_trend_metric_selector"
                )
                selected_time_trend_metric_col = time_trend_metric_options[selected_time_trend_metric_display]

                if selected_time_trend_metric_col in monthly_kpi_df.columns:
                    fig_time_trend = px.line(
                        monthly_kpi_df,
                        x="month",
                        y=selected_time_trend_metric_col,
                        title=f"{selected_time_trend_metric_display} Over Time",
                        labels={
                            "month": "Month",
                            selected_time_trend_metric_col: selected_time_trend_metric_display
                        }
                    )
                    fig_time_trend.update_xaxes(
                        dtick="M1", # Tick every month
                        tickformat="%b\n%Y", # Format as Jan\n2023
                        showgrid=True
                    )
                    fig_time_trend.update_yaxes(showgrid=True)
                    fig_time_trend.update_layout(hovermode="x unified") # Show all traces on hover
                    st.plotly_chart(fig_time_trend, use_container_width=True)
                else:
                    st.warning(f"Selected metric '{selected_time_trend_metric_display}' not found in monthly KPI data. Please ensure 'data_processing.py' produces this column.")
            else:
                st.warning("The 'month' column is missing from the monthly KPI data. Please ensure 'data_processing.py' correctly generates this column in 'monthly_kpi_output.csv'.")
        else:
            st.info("Monthly KPI data not available or is empty. Please run the analysis first.")


    with tab_cost_flow_analysis: # NEW TAB for Sankey Diagram
        st.subheader("Cost Flow Analysis (Sankey Diagram)")
        st.info("Visualize the flow of 'Net Cost Impact' across different dimensions to understand cost distribution.")

        cleaned_df_sankey = st.session_state.cleaned_df
        if cleaned_df_sankey.empty:
            st.warning("No data loaded for cost flow analysis. Please upload and process data in the 'App Setup & Data Processing' tab.")
        else:
            sankey_view_options = {
                "Fulfillment Location -> Item Class -> Ship State": ["fulfillment_loc", "item_class", "ship_state"],
                "Customer Name -> Item Class -> Ship State": ["customer_name", "item_class", "ship_state"],
                "Customer Name -> Fulfillment Location -> Item Class": ["customer_name", "fulfillment_loc", "item_class"],
                "Item Class -> Item Type -> Ship State": ["item_class", "item_type", "ship_state"]
            }
            selected_sankey_view = st.selectbox(
                "Select Sankey Diagram View",
                list(sankey_view_options.keys()),
                key="sankey_view_selector"
            )
            flow_path_columns = sankey_view_options[selected_sankey_view]

            # Aggregate data for Sankey based on selected flow path
            # Ensure all columns in flow_path_columns exist in cleaned_df_sankey before grouping
            # And also ensure 'net_cost_impact' exists
            required_cols = flow_path_columns + ['net_cost_impact']
            if not all(col in cleaned_df_sankey.columns for col in required_cols):
                st.warning(f"Required columns for selected Sankey view are missing: {', '.join(required_cols)}. Please check your data.")
            else:
                sankey_data = cleaned_df_sankey.groupby(flow_path_columns)['net_cost_impact'].sum().reset_index()
                sankey_data = sankey_data[sankey_data['net_cost_impact'] > 0] # Only show positive cost flows

                if sankey_data.empty:
                    st.info("No positive net cost impact data found for the selected Sankey diagram after aggregation.")
                else:
                    # Prepare data for Sankey Diagram
                    # All unique nodes (sources, intermediates, targets)
                    all_nodes = pd.Series(dtype=object)
                    for col in flow_path_columns:
                        all_nodes = pd.concat([all_nodes, sankey_data[col]])
                    all_nodes = all_nodes.unique().tolist()
                    
                    # Create a mapping from node name to integer index
                    label_to_index = {label: i for i, label in enumerate(all_nodes)}

                    links = []
                    # Create links for the diagram based on the selected flow path
                    for i in range(len(flow_path_columns) - 1):
                        source_col = flow_path_columns[i]
                        target_col = flow_path_columns[i+1]
                        
                        temp_links = sankey_data.groupby([source_col, target_col])['net_cost_impact'].sum().reset_index()
                        temp_links['source'] = temp_links[source_col].map(label_to_index)
                        temp_links['target'] = temp_links[target_col].map(label_to_index)
                        temp_links['value'] = temp_links['net_cost_impact']
                        links.append(temp_links)

                    # Combine all links
                    all_links = pd.concat(links)

                    # Handle potential NaN values from mapping if some nodes don't exist
                    # This ensures only valid links are used
                    all_links = all_links.dropna(subset=['source', 'target', 'value']).copy()
                    all_links[['source', 'target']] = all_links[['source', 'target']].astype(int)

                    if all_links.empty:
                        st.warning("No valid links could be created for the Sankey diagram after processing.")
                    else:
                        # Create the Sankey Diagram
                        fig_sankey = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=all_nodes,
                                color="blue" # Default color for nodes
                            ),
                            link=dict(
                                source=all_links['source'],
                                target=all_links['target'],
                                value=all_links['value'],
                                label=all_links.apply(lambda r: f"{all_nodes[r['source']]} to {all_nodes[r['target']]}: ${r['value']:,.2f}", axis=1)
                            )
                        )])

                        fig_sankey.update_layout(title_text=f"Net Cost Impact Flow: {selected_sankey_view}", font_size=10, height=800)
                        st.plotly_chart(fig_sankey, use_container_width=True)


    with tab_detailed_kpis:
        st.subheader("Detailed KPI Reports")
        st.info("All generated KPI tables are displayed below. No clicking around needed. Use the filters within each table to refine your view. Use the dropdown to quickly jump to a specific report.")

        # Dropdown to select report to view
        report_options_detailed = list(output_paths.values())
        selected_report_display_name_detailed = st.selectbox(
            "Select a Report to View", 
            report_options_detailed, 
            key="detailed_report_selector",
            help="Choose which detailed KPI report you'd like to examine."
        )

        selected_report_filename_detailed = next(key for key, value in output_paths.items() if value == selected_report_display_name_detailed)
        
        if selected_report_filename_detailed in st.session_state.generated_dfs:
            current_df_to_display = st.session_state.generated_dfs[selected_report_filename_detailed]

            if not current_df_to_display.empty:
                st.write(f"#### {selected_report_display_name_detailed} Table")
                
                st.info("Use the filters below to refine your view of the table data.")
                
                # Filters for the table (Bloomberg style interaction)
                filter_cols = ['item_sku', 'fulfillment_loc', 'item_class', 'customer_name', 'ship_state', 'month', 'customer_category', 'item_type']
                # FIX: Corrected the NameError: filter_df_to_display was not defined.
                # It should check against the `filter_cols` list and the `current_df_to_display`'s actual columns.
                available_filter_cols = [col for col in filter_cols if col in current_df_to_display.columns]

                current_filtered_df = current_df_to_display.copy()

                for col in available_filter_cols:
                    if col == 'item_sku' or col == 'customer_name': # Text input for search
                        search_term = st.text_input(f"Search by {col} (partial match)", key=f"{selected_report_filename_detailed}_{col}_search")
                        if search_term:
                            current_filtered_df = current_filtered_df[
                                current_filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
                            ]
                    else: # Selectbox for categories
                        # FIX: Convert all unique values to string for sorting to avoid TypeError
                        unique_values = current_df_to_display[col].unique().tolist()
                        all_unique_values = ['All'] + sorted([str(x) for x in unique_values])

                        selected_value = st.selectbox(f"Filter by {col}", all_unique_values, key=f"{selected_report_filename_detailed}_{col}_filter")
                        if selected_value != 'All':
                            # Attempt to convert back to original type if unique values are numeric
                            if pd.api.types.is_numeric_dtype(current_df_to_display[col]):
                                try:
                                    selected_value = pd.to_numeric(selected_value)
                                except ValueError:
                                    pass # Keep as string if conversion fails
                            current_filtered_df = current_filtered_df[current_filtered_df[col].astype(str) == selected_value] # FIX: Convert column to str for comparison
                
                st.write(f"Displaying {len(current_filtered_df)} of {len(current_df_to_display)} entries.")
                st.dataframe(current_filtered_df, use_container_width=True)

                # --- Download Button for Current Report ---
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_data = convert_df_to_csv(current_filtered_df) # Download filtered data
                st.download_button(
                    label=f"Download {selected_report_display_name_detailed} as CSV",
                    data=csv_data,
                    file_name=selected_report_filename_detailed,
                    mime="text/csv",
                    key=f"download_{selected_report_filename_detailed}"
                )
                st.markdown("---") # Separator between tables
            else:
                st.info(f"The selected report '{selected_report_display_name_detailed}' is empty. Please ensure data is available for this report type.")
        else:
            st.info(f"Report '{selected_report_display_name_detailed}' not yet generated. Please run the analysis first.")
