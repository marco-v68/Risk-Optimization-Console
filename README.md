# Risk Optimization Console

This Streamlit application provides a comprehensive suite of analytics for inventory and warranty data, including:

* **Data Cleaning & Processing:** Handles raw transaction data.
* **KPI Generation:** Calculates key performance indicators across various dimensions (SKU, Customer, Location, State, Monthly, Item Class, Item Type, Service Center).
* **Safety Stock & Reorder Point Calculation:** Recommends optimal safety stock levels and reorder points based on historical failures and risk factors.
* **Interactive Dashboards:** Visualizes KPIs through tables, charts, heatmaps, and Sankey diagrams for insightful analysis.

## How to Use

1.  **Upload Data:** Upload your `warranty_raw.csv` file in the "App Setup & Data Processing" section.
2.  **Run Analysis:** Click "Run All Analysis & Generate Reports" after setting your desired "As Of" date.
3.  **Explore Tabs:** Navigate through the different tabs to view Executive Summaries, detailed KPI reports, and advanced visualizations.

## Requirements

This application requires the following Python libraries:
* `streamlit`
* `pandas`
* `numpy`
* `plotly`
* `scikit-learn`

(See `requirements.txt` for exact versions)
