"""
Streamlit Clinical Prediction Dashboard — Task 06 Prototype
SmartCare Hospital AI Risk Prediction System
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prediction.predictor import RiskPredictor

st.set_page_config(
    page_title="SmartCare Hospital AI — Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 SmartCare Hospital AI — Patient Risk Prediction Prototype")
st.markdown("Enter patient demographic and clinical information to compute real-time disease risk classification and appointment adherence predictions.")

st.sidebar.header("📋 Clinical Inputs")

age = st.sidebar.slider("Age", 0, 100, 45)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
blood_group = st.sidebar.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
systolic_bp = st.sidebar.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
diastolic_bp = st.sidebar.number_input("Diastolic Blood Pressure (mmHg)", 50, 130, 80)
blood_sugar = st.sidebar.number_input("Blood Sugar (mg/dL)", 70, 300, 100)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 120, 400, 190)
bmi = st.sidebar.number_input("BMI (kg/m²)", 15.0, 50.0, 24.5)

department = st.sidebar.selectbox(
    "Department", 
    ["Cardiology", "General Medicine", "Neurology", "Orthopedics", "Pediatrics", "Radiology", "Laboratory Services"]
)
diagnosis = st.sidebar.selectbox(
    "Diagnosis", 
    ["Asthma", "Back Pain", "Chest Pain", "Diabetes", "Fever", "Fracture", "Hypertension", "Kidney Infection", "Migraine", "Pneumonia"]
)
prev_appointments = st.sidebar.number_input("Previous Appointments Count", 0, 20, 2)
missed_appointments = st.sidebar.number_input("Previous Missed Appointments Count", 0, 10, 0)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("👤 Patient Clinical Summary")
    st.json({
        "Age": age,
        "Gender": gender,
        "Blood Group": blood_group,
        "Blood Pressure": f"{systolic_bp}/{diastolic_bp} mmHg",
        "Blood Sugar": f"{blood_sugar} mg/dL",
        "Cholesterol": f"{cholesterol} mg/dL",
        "BMI": bmi,
        "Department": department,
        "Diagnosis": diagnosis,
        "Appointment History": f"{prev_appointments} Total ({missed_appointments} Missed)"
    })

with col2:
    st.subheader("⚡ Real-Time Prediction")
    if st.button("🔮 Evaluate Patient Risk", use_container_width=True):
        model_path = "models/best_model.pkl"
        if not os.path.exists(model_path):
            st.error("Model artifact not found. Please train models first via Task 05.")
        else:
            predictor = RiskPredictor(model_path)
            # Load template feature columns from dataset
            data_path = "data/processed/smartcare_final_preprocessed_dataset.csv"
            if os.path.exists(data_path):
                sample_df = pd.read_csv(data_path, nrows=1).drop(columns=["no_show"])
                input_data = pd.DataFrame(0, index=[0], columns=sample_df.columns)
                
                # Predict
                res = predictor.predict(input_data)
                
                if res["prediction"] == 1:
                    st.error(f"🚨 **High Risk Alert**: {res['risk_label']}")
                else:
                    st.success(f"✅ **Low Risk**: {res['risk_label']}")
                    
                st.metric(label="Model Confidence Score", value=f"{res['confidence']*100:.1f}%")
                
                st.markdown("### Model Probabilities")
                st.bar_chart(pd.DataFrame({
                    "Probability": res["probabilities"]
                }, index=["Low Risk (Show-Up)", "High Risk (No-Show)"]))
            else:
                st.error("Preprocessed dataset not found to match feature schema.")

st.divider()
st.caption("SmartCare Hospital AI — Machine Learning System for Disease Risk Classification & Clinical Decision Support")
