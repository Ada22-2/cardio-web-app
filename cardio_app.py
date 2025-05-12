import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

# Load model, scaler, and feature order
model = joblib.load("cardio_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("feature_order.json", "r") as f:
    feature_order = json.load(f)

st.title("🫀 Cardiovascular Disease Risk Predictor")
st.write("Enter patient details to assess risk of heart disease.")

# User inputs
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# On click
if st.button("Predict"):
    # Convert sex to numeric
    sex_val = 1 if sex == "Male" else 0

    # Construct input DataFrame with correct column order
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex_val,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    input_df = input_df[feature_order]  # enforce order

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)

    # Optional: show scaled input
    st.write("📊 Scaled Input:")
    st.dataframe(pd.DataFrame(input_scaled, columns=feature_order))

    # Custom threshold
    threshold = 0.65

    if proba[0][1] > threshold:
        st.error(f"⚠️ At Risk of Heart Disease\nConfidence: {proba[0][1]*100:.2f}%")
    else:
        st.success(f"✅ Very Low Risk of Heart Disease\nConfidence: {(1 - proba[0][1])*100:.2f}%")
