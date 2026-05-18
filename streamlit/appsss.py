
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="PhonePe Dashboard",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("PhonePe Transaction Insights Dashboard")

st.markdown("---")

# =====================================================
# LOAD CSV FILES
# =====================================================

agg_transaction_df = pd.read_csv("agg_transaction.csv")

agg_user_df = pd.read_csv("agg_user.csv")

agg_insurance_df = pd.read_csv("agg_insurance.csv")

map_transaction_df = pd.read_csv("map_transaction.csv")

map_user_df = pd.read_csv("map_user.csv")

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("Filters")

years = sorted(agg_transaction_df['Year'].unique())

selected_year = st.sidebar.selectbox(
    "Select Year",
    years
)

states = sorted(agg_transaction_df['State'].unique())

selected_state = st.sidebar.selectbox(
    "Select State",
    states
)

# =====================================================
# FILTER DATA
# =====================================================

filtered_transaction = agg_transaction_df[
    (agg_transaction_df['Year'] == selected_year) &
    (agg_transaction_df['State'] == selected_state)
]

filtered_user = map_user_df[
    (map_user_df['Year'] == selected_year) &
    (map_user_df['State'] == selected_state)
]

filtered_insurance = agg_insurance_df[
    (agg_insurance_df['Year'] == selected_year) &
    (agg_insurance_df['State'] == selected_state)
]

# =====================================================
# KPI SECTION
# =====================================================

st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

total_amount = int(filtered_transaction['Transaction_Amount'].sum())

total_count = int(filtered_transaction['Transaction_Count'].sum())

total_users = int(filtered_user['Registered_Users'].sum())

with col1:
    st.metric(
        "Total Transaction Amount",
        f"₹ {total_amount:,}"
    )

with col2:
    st.metric(
        "Total Transaction Count",
        f"{total_count:,}"
    )

with col3:
    st.metric(
        "Registered Users",
        f"{total_users:,}"
    )

st.markdown("---")

# =====================================================
# CHART 1 — TRANSACTION TYPE ANALYSIS
# =====================================================

st.subheader("Transaction Type Distribution")

transaction_type = filtered_transaction.groupby(
    'Transaction_Type'
)['Transaction_Amount'].sum().reset_index()

fig1 = px.pie(
    transaction_type,
    names='Transaction_Type',
    values='Transaction_Amount',
    title='Transaction Type Distribution'
)

st.plotly_chart(fig1, use_container_width=True)

# =====================================================
# CHART 2 — QUARTERLY TREND
# =====================================================

st.subheader("Quarterly Transaction Trend")

quarterly_data = filtered_transaction.groupby(
    'Quarter'
)['Transaction_Amount'].sum().reset_index()

fig2 = px.line(
    quarterly_data,
    x='Quarter',
    y='Transaction_Amount',
    markers=True,
    title='Quarterly Transaction Amount'
)

st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# CHART 3 — TOP DISTRICTS
# =====================================================

st.subheader("Top Districts by Transaction Amount")

filtered_map_transaction = map_transaction_df[
    (map_transaction_df['Year'] == selected_year) &
    (map_transaction_df['State'] == selected_state)
]

district_data = filtered_map_transaction.groupby(
    'District'
)['Transaction_Amount'].sum().reset_index()

district_data = district_data.sort_values(
    by='Transaction_Amount',
    ascending=False
)

fig3 = px.bar(
    district_data.head(10),
    x='District',
    y='Transaction_Amount',
    color='District',
    title='Top Districts'
)

st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# CHART 4 — APP OPENS
# =====================================================

st.subheader("App Opens Analysis")

app_open_data = filtered_user.groupby(
    'District'
)['App_Opens'].sum().reset_index()

fig4 = px.bar(
    app_open_data.head(10),
    x='District',
    y='App_Opens',
    color='District',
    title='District-wise App Opens'
)

st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# CHART 5 — INSURANCE ANALYSIS
# =====================================================

st.subheader("Insurance Transaction Analysis")

insurance_data = filtered_insurance.groupby(
    'Quarter'
)['Transaction_Amount'].sum().reset_index()

fig5 = px.bar(
    insurance_data,
    x='Quarter',
    y='Transaction_Amount',
    color='Quarter',
    title='Insurance Transaction Amount'
)

st.plotly_chart(fig5, use_container_width=True)

# =====================================================
# DATAFRAME DISPLAY
# =====================================================

st.subheader("Filtered Transaction Data")

st.dataframe(filtered_transaction)

# =====================================================
# MACHINE LEARNING SECTION
# =====================================================

st.markdown("---")

st.header("Machine Learning Models")

# =====================================================
# PREPARE DATA
# =====================================================

model_df = agg_transaction_df.dropna()

X = model_df[[
    'Year',
    'Quarter',
    'Transaction_Count'
]]

y = model_df['Transaction_Amount']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# LINEAR REGRESSION
# =====================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)

# =====================================================
# POLYNOMIAL REGRESSION
# =====================================================

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

poly_pred = poly_model.predict(X_test_poly)

poly_r2 = r2_score(y_test, poly_pred)

# =====================================================
# RANDOM FOREST
# =====================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)

# =====================================================
# MODEL COMPARISON TABLE
# =====================================================

st.subheader("Model Comparison")

comparison_df = pd.DataFrame({

    'Model': [
        'Linear Regression',
        'Polynomial Regression',
        'Random Forest'
    ],

    'R2 Score': [
        linear_r2,
        poly_r2,
        rf_r2
    ]

})

st.dataframe(comparison_df)

# =====================================================
# MODEL COMPARISON CHART
# =====================================================

fig_ml = px.bar(
    comparison_df,
    x='Model',
    y='R2 Score',
    color='Model',
    title='Machine Learning Model Comparison'
)

st.plotly_chart(fig_ml, use_container_width=True)

# =====================================================
# USER INPUTS
# =====================================================

st.subheader("Predict Transaction Amount")

input_year = st.number_input(
    "Enter Year",
    min_value=2018,
    max_value=2030,
    value=2025
)

input_quarter = st.selectbox(
    "Select Quarter",
    [1, 2, 3, 4]
)

input_transaction_count = st.number_input(
    "Enter Transaction Count",
    min_value=0,
    value=100000
)

# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button("Predict Using All Models"):

    # LINEAR REGRESSION

    linear_prediction = linear_model.predict([[
        input_year,
        input_quarter,
        input_transaction_count
    ]])

    linear_prediction = int(abs(linear_prediction[0]))

    # POLYNOMIAL REGRESSION

    poly_input = poly.transform([[
        input_year,
        input_quarter,
        input_transaction_count
    ]])

    poly_prediction = poly_model.predict(poly_input)

    poly_prediction = int(abs(poly_prediction[0]))

    # RANDOM FOREST

    rf_prediction = rf_model.predict([[
        input_year,
        input_quarter,
        input_transaction_count
    ]])

    rf_prediction = int(abs(rf_prediction[0]))

    # =================================================
    # DISPLAY RESULTS
    # =================================================

    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Linear Regression",
            f"₹ {linear_prediction:,}"
        )

    with col2:
        st.metric(
            "Polynomial Regression",
            f"₹ {poly_prediction:,}"
        )

    with col3:
        st.metric(
            "Random Forest",
            f"₹ {rf_prediction:,}"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.write(
    "PhonePe Analytics Dashboard using Python, Streamlit and Machine Learning"
)
