"""
app.py – Car Mileage Predictor (Final Version with Model Info)

Run:
    streamlit run app.py
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Car Mileage Predictor",
    page_icon="",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

rf = joblib.load("rf_model.pkl")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown("<h1 style='text-align:center;'>CAR MILEAGE PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Machine Learning Mileage Prediction System</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2 = st.tabs(["Predict", "Model Information"])

# =========================================================
# TAB 1 – PREDICT
# =========================================================

with tab1:

    col_cat, col_num = st.columns(2)

    # -------- CATEGORICAL --------
    with col_cat:
        st.subheader("Categorical Features")

        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        body_type = st.selectbox("Body Type", ["SUV", "Sedan", "Hatchback", "MUV"])
        drivetrain = st.selectbox("Drivetrain", ["FWD", "RWD", "4WD"])

    # -------- NUMERIC --------
    with col_num:
        st.subheader("Numeric Features")

        displacement = st.slider("Displacement (cc)", 500, 5000, 1200)
        cylinders = st.slider("Cylinders", 2, 12, 4)
        power = st.slider("Power (PS)", 30, 600, 85)
        torque = st.slider("Torque (Nm)", 40, 800, 120)
        kerb_weight = st.slider("Kerb Weight (kg)", 500, 3000, 1100)
        gears = st.slider("Gears", 4, 9, 5)

    st.markdown("---")

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        predict_btn = st.button("Predict Mileage", use_container_width=True)

    # -------- Manual Encoding --------
    fuel_map = {"CNG": 0, "Diesel": 1, "Petrol": 2}
    drive_map = {"FWD": 0, "RWD": 1, "4WD": 2}
    body_map = {"Hatchback": 0, "MUV": 1, "SUV": 2, "Sedan": 3}

    if predict_btn:

        input_data = [[
            displacement,
            kerb_weight,
            power,
            torque,
            gears,
            cylinders,
            fuel_map.get(fuel_type, 0),
            drive_map.get(drivetrain, 0),
            body_map.get(body_type, 0)
        ]]

        prediction = rf.predict(input_data)[0]

        st.markdown("##Prediction Result")

        r1, r2, r3 = st.columns([1,2,1])
        with r2:
            st.success(f"### {prediction:.2f} km per litre")

# =========================================================
# TAB 2 – MODEL INFORMATION
# =========================================================

with tab2:

    st.subheader("Model Details")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Algorithm", "Random Forest Regressor")
        st.metric("Number of Trees", "300")

    with col2:
        st.metric("Model Accuracy (R²)", "0.95")   # Replace with your real value
        st.metric("Problem Type", "Regression")

    st.markdown("---")

    st.info("""
    This model predicts car mileage (km/litre) 
    using engine specifications and vehicle features.
    """)