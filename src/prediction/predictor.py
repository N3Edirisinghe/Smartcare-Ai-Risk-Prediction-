"""
Prediction Inference Module for Task 06.
SmartCare Hospital AI — Option A: Appointment No-Show Prediction
"""
import os
import joblib
import pandas as pd
import numpy as np

class RiskPredictor:
    def __init__(self, model_path="models/best_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
        self.model = joblib.load(model_path)
        
    def predict(self, input_df):
        """
        Accepts preprocessed input dataframe and returns prediction risk level and probability.
        """
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0] if hasattr(self.model, "predict_proba") else [0.5, 0.5]
        
        risk_label = "High Risk (No-Show Likely)" if prediction == 1 else "Low Risk (Show-Up Likely)"
        confidence = probabilities[prediction] if hasattr(self.model, "predict_proba") else 1.0
        
        return {
            "prediction": int(prediction),
            "risk_label": risk_label,
            "confidence": float(confidence),
            "probabilities": probabilities
        }
