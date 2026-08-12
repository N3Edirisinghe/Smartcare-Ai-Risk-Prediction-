# SmartCare AI – XAI and Dashboard

## Overview

This part of the SmartCare AI project focuses on Explainable AI (XAI) and an interactive dashboard for the trained machine learning model.

SHAP (SHapley Additive exPlanations) is used to understand which features have the greatest influence on the model's predictions. A Streamlit dashboard is also used to provide an interactive way to view predictions and their explanations.

## Files

The main files used for this part are:

- `best_model.pkl` – trained machine learning model.
- `smartcare_final_preprocessed_dataset.csv` – preprocessed dataset used for the XAI analysis.
- `app.py` – Streamlit dashboard application.
- `SmartCare_XAI_Dashboard.ipynb` – Google Colab notebook containing the XAI analysis and dashboard setup.

## XAI

SHAP was used to explain the predictions made by the trained model.

The analysis includes:

- SHAP feature importance
- Mean absolute SHAP values
- Feature ranking
- Global model explanation
- Individual prediction explanations

The most important features identified by the SHAP analysis were:

1. Waiting days
2. Missed previous appointments
3. Previous appointments
4. Appointment day of week
5. Age
6. Blood sugar
7. Previous missed rate
8. Appointment month
9. Systolic blood pressure
10. Cholesterol

The highest mean absolute SHAP value was observed for `numerical__waiting_days`, followed by `numerical__missed_previous_appointments` and `numerical__previous_appointments`.

## Model Features

The trained model uses 44 preprocessed features, including:

- Age
- Waiting days
- Previous appointments
- Missed previous appointments
- Previous missed rate
- Appointment month
- Appointment day of week
- Blood pressure
- Blood sugar
- Cholesterol
- BMI
- Gender
- Blood group
- Department
- Diagnosis
- Blood pressure category

Categorical variables were converted into numerical features during preprocessing.

## Dashboard

The Streamlit dashboard provides an interactive interface for the trained model.

The dashboard allows the user to:

- Enter input values
- Generate a model prediction
- View the prediction result
- View prediction probability
- Understand the important features affecting the prediction
- View SHAP-based explanations

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- SHAP
- Matplotlib
- Streamlit
- Google Colab

## Installation

Install the required libraries using:

```bash
pip install shap joblib streamlit pandas numpy matplotlib scikit-learn
