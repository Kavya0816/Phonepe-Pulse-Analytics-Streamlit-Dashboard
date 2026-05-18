# 📊 PhonePe Pulse Data Visualization & Forecasting Platform

An enterprise-grade data platform that automates the extraction, processing, analysis, and forecasting of nationwide transaction and user footprint metrics from the PhonePe Pulse ecosystem. This project features a full ETL pipeline, rigorous exploratory data analysis, multi-model machine learning comparison, and a live interactive Streamlit dashboard.

---

## 🚀 Key Features

* **Automated Extraction (ETL):** Recursively traverses deeply nested, state-and-year-based JSON directories, cleaning and structuring data fragments into uniform local CSV assets.
* **Interactive Visualizations:** Implements 7 advanced geo-spatial, temporal, and co-dependency charts (using Plotly, Seaborn, and Matplotlib) rendered natively inside the platform.
* **Multi-Model Predictive Simulator:** Trains and compares Linear Regression, Polynomial Regression (Degree 2), and Random Forest Regressor models to forecast transaction valuation.
* **Decoupled Architecture:** Serializes the trained model weights (`.pkl`) locally via `joblib`, enabling the user-facing Streamlit dashboard to serve real-time predictions without the lag of runtime training.

---

