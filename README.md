# Phonepe-Pulse-Analytics-Streamlit-Dashboard
An interactive Streamlit dashboard for exploring nationwide PhonePe Pulse data with automated JSON ETL pipelines, advanced geo-spatial EDA, and predictive Polynomial Regression forecasting models.

# 📊 PhonePe Pulse Data Visualization & Forecasting Platform

An enterprise-grade data platform that automates the extraction, processing, analysis, and forecasting of nationwide transaction and user footprint metrics from the PhonePe Pulse ecosystem. This project features a full ETL pipeline, rigorous exploratory data analysis, multi-model machine learning comparison, and a live interactive Streamlit dashboard.

---

## 🚀 Key Features

* **Automated Extraction (ETL):** Recursively traverses deeply nested, state-and-year-based JSON directories, cleaning and structuring data fragments into uniform local CSV assets.
* **Interactive Visualizations:** Implements 7 advanced geo-spatial, temporal, and co-dependency charts (using Plotly, Seaborn, and Matplotlib) rendered natively inside the platform.
* **Multi-Model Predictive Simulator:** Trains and compares Linear Regression, Polynomial Regression (Degree 2), and Random Forest Regressor models to forecast transaction valuation.
* **Decoupled Architecture:** Serializes the trained model weights (`.pkl`) locally via `joblib`, enabling the user-facing Streamlit dashboard to serve real-time predictions without the lag of runtime training.

---

## 📂 Project Architecture & Repository Structure

```text
PhonePe-Pulse-Analytics/
│
├── .gitignore               # Excludes python bytecodes, local check-pointing, and environment dumps
├── README.md                # Comprehensive portfolio landing documentation
├── requirements.txt         # Production-ready package dependency lockfile
├── apps.py                  # Streamlit dashboard application source code
├── phonepe_poly_model.pkl   # Serialized Polynomial Regression Pipeline artifact
├── phonepe_scaler.pkl       # Serialized StandardScaler parameter configurations
│
├── Data/                    # Standardized structured relational data matrices (CSV format)
│   ├── agg_transaction.csv
│   ├── agg_user.csv
│   ├── agg_insurance.csv
│   ├── map_transaction.csv
│   └── map_user.csv
│
└── Notebooks/               # Development environments for modeling and analysis
    └── analysis.ipynb       # Main notebook housing ETL logic, hypothesis testing, and ML optimization
