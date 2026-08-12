
import os
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartCare AI",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = (
    "/content/drive/MyDrive/"
    "smartcare -ai/xai/best_model.pkl"
)

DATA_PATH = (
    "/content/drive/MyDrive/"
    "smartcare -ai/xai/"
    "smartcare_final_preprocessed_dataset.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


try:

    model = load_model()
    df = load_dataset()

except Exception as e:

    st.error("Could not load the SmartCare files.")
    st.exception(e)
    st.stop()


# ============================================================
# MODEL FEATURES
# ============================================================

X = df.drop(columns=["no_show"])


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="text-align:center">

    <h1>🏥 SmartCare AI</h1>

    <h3>
    Appointment No-Show Prediction
    </h3>

    <p>
    Machine Learning + Explainable AI
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏥 SmartCare AI")

st.sidebar.write(
    "Enter patient and appointment information "
    "to generate a prediction."
)

st.sidebar.divider()

st.sidebar.write(
    "**Model:** Logistic Regression"
)

st.sidebar.write(
    "**Features:** 44"
)

st.sidebar.write(
    "**XAI:** SHAP"
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("👤 Patient Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=40,
        step=1
    )


with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )


with col3:

    blood_group = st.selectbox(
        "Blood Group",
        [
            "A+",
            "A-",
            "AB+",
            "AB-",
            "B+",
            "B-",
            "O+",
            "O-"
        ]
    )


# ============================================================
# APPOINTMENT INFORMATION
# ============================================================

st.header("📅 Appointment Information")

col1, col2, col3 = st.columns(3)


with col1:

    waiting_days = st.number_input(
        "Waiting Days",
        min_value=0,
        max_value=365,
        value=7
    )


with col2:

    previous_appointments = st.number_input(
        "Previous Appointments",
        min_value=0,
        max_value=100,
        value=2
    )


with col3:

    missed_previous_appointments = st.number_input(
        "Previously Missed Appointments",
        min_value=0,
        max_value=100,
        value=0
    )


col1, col2, col3 = st.columns(3)


with col1:

    appointment_month = st.selectbox(
        "Appointment Month",
        list(range(1, 13)),
        format_func=lambda x: [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ][x - 1]
    )


with col2:

    appointment_dayofweek = st.selectbox(
        "Appointment Day",
        list(range(7)),
        format_func=lambda x: [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ][x]
    )


with col3:

    has_previous_appointment = st.selectbox(
        "Has Previous Appointment?",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


# ============================================================
# PREVIOUS APPOINTMENT RATE
# ============================================================

if previous_appointments > 0:

    previous_missed_rate = (
        missed_previous_appointments /
        previous_appointments
    )

else:

    previous_missed_rate = 0.0


st.info(
    f"Calculated previous missed rate: "
    f"{previous_missed_rate:.2f}"
)


# ============================================================
# CLINICAL INFORMATION
# ============================================================

st.header("🩺 Clinical Information")

col1, col2, col3 = st.columns(3)


with col1:

    systolic_bp = st.number_input(
        "Systolic Blood Pressure",
        min_value=50.0,
        max_value=250.0,
        value=120.0
    )


with col2:

    diastolic_bp = st.number_input(
        "Diastolic Blood Pressure",
        min_value=30.0,
        max_value=150.0,
        value=80.0
    )


with col3:

    blood_sugar = st.number_input(
        "Blood Sugar (mg/dL)",
        min_value=0.0,
        max_value=500.0,
        value=100.0
    )


col1, col2, col3 = st.columns(3)


with col1:

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=50.0,
        max_value=500.0,
        value=180.0
    )


with col2:

    bmi = st.number_input(
        "BMI",
        min_value=5.0,
        max_value=80.0,
        value=24.0
    )


with col3:

    bp_category = st.selectbox(
        "BP Category",
        [
            "Elevated",
            "High",
            "Normal",
            "Very High"
        ]
    )


# ============================================================
# MEDICAL INFORMATION
# ============================================================

st.header("🏥 Medical Information")

col1, col2 = st.columns(2)


with col1:

    department = st.selectbox(
        "Department",
        [
            "Cardiology",
            "General Medicine",
            "Laboratory Services",
            "Neurology",
            "Orthopedics",
            "Pediatrics",
            "Radiology"
        ]
    )


with col2:

    diagnosis = st.selectbox(
        "Diagnosis",
        [
            "Asthma",
            "Back Pain",
            "Chest Pain",
            "Diabetes",
            "Fever",
            "Fracture",
            "Hypertension",
            "Kidney Infection",
            "Migraine",
            "Pneumonia"
        ]
    )


st.divider()


# ============================================================
# BUILD 44-FEATURE INPUT
# ============================================================

def create_input():

    # Start with ALL model features set to zero

    row = pd.DataFrame(
        0.0,
        index=[0],
        columns=X.columns
    )


    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    values = {

        "numerical__age":
            age,

        "numerical__waiting_days":
            waiting_days,

        "numerical__previous_appointments":
            previous_appointments,

        "numerical__missed_previous_appointments":
            missed_previous_appointments,

        "numerical__previous_missed_rate":
            previous_missed_rate,

        "numerical__has_previous_appointment":
            has_previous_appointment,

        "numerical__appointment_month":
            appointment_month,

        "numerical__appointment_dayofweek":
            appointment_dayofweek,

        "numerical__systolic_bp":
            systolic_bp,

        "numerical__diastolic_bp":
            diastolic_bp,

        "numerical__blood_sugar_mg_dl":
            blood_sugar,

        "numerical__cholesterol_mg_dl":
            cholesterol,

        "numerical__bmi":
            bmi
    }


    for feature, value in values.items():

        if feature in row.columns:

            row.loc[0, feature] = value


    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    gender_feature = (
        "categorical__gender_" + gender
    )

    if gender_feature in row.columns:

        row.loc[
            0,
            gender_feature
        ] = 1


    # --------------------------------------------------------
    # BLOOD GROUP
    # --------------------------------------------------------

    blood_feature = (
        "categorical__blood_group_" +
        blood_group
    )

    if blood_feature in row.columns:

        row.loc[
            0,
            blood_feature
        ] = 1


    # --------------------------------------------------------
    # DEPARTMENT
    # --------------------------------------------------------

    department_feature = (
        "categorical__department_" +
        department
    )

    if department_feature in row.columns:

        row.loc[
            0,
            department_feature
        ] = 1


    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    diagnosis_feature = (
        "categorical__diagnosis_" +
        diagnosis
    )

    if diagnosis_feature in row.columns:

        row.loc[
            0,
            diagnosis_feature
        ] = 1


    # --------------------------------------------------------
    # BP CATEGORY
    # --------------------------------------------------------

    bp_feature = (
        "categorical__bp_category_" +
        bp_category
    )

    if bp_feature in row.columns:

        row.loc[
            0,
            bp_feature
        ] = 1


    return row


# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.button(
     "Predict Appointment No-Show",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    try:

        input_data = create_input()


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = (
                model.predict_proba(
                    input_data
                )[0]
            )

            classes = list(
                model.classes_
            )

            if 1 in classes:

                no_show_probability = (
                    probabilities[
                        classes.index(1)
                    ]
                )

            else:

                no_show_probability = (
                    probabilities[-1]
                )

        else:

            no_show_probability = None


        # ====================================================
        # RESULT
        # ====================================================

        st.header("Prediction Result")


        if int(prediction) == 1:

            st.error(
                " HIGH NO-SHOW RISK"
            )

        else:

            st.success(
                " LIKELY TO ATTEND"
            )


        if no_show_probability is not None:

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "No-Show Probability",
                    f"{no_show_probability * 100:.2f}%"
                )

            with c2:

                attendance_probability = (
                    1 -
                    no_show_probability
                )

                st.metric(
                    "Attendance Probability",
                    f"{attendance_probability * 100:.2f}%"
                )


        # ====================================================
        # SHAP
        # ====================================================

        st.header(
            "Explainable AI — Why?"
        )

        with st.spinner(
            "Generating SHAP explanation..."
        ):

            explainer = shap.LinearExplainer(
                model,
                X
            )

            explanation = explainer(
                input_data
            )


        # ----------------------------------------------------
        # WATERFALL
        # ----------------------------------------------------

        fig = plt.figure(
            figsize=(12, 8)
        )

        shap.plots.waterfall(
            explanation[0],
            max_display=15,
            show=False
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )


        # ====================================================
        # TOP FEATURES
        # ====================================================

        shap_table = pd.DataFrame({

            "Feature":
                input_data.columns,

            "Input Value":
                input_data.iloc[0].values,

            "SHAP Value":
                explanation.values[0]

        })

        shap_table["Impact"] = (
            shap_table["SHAP Value"]
            .apply(
                lambda x:
                "Increases No-Show Prediction"
                if x > 0
                else "Decreases No-Show Prediction"
            )
        )

        shap_table["Absolute SHAP"] = (
            np.abs(
                shap_table["SHAP Value"]
            )
        )

        shap_table = (
            shap_table
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
            .head(10)
        )


        st.subheader(
            "Top Factors Influencing Prediction"
        )

        st.dataframe(
            shap_table[
                [
                    "Feature",
                    "Input Value",
                    "SHAP Value",
                    "Impact"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        with st.expander(
            "View processed model input"
        ):

            st.dataframe(
                input_data.T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.warning(
            """
            **Important:** This is an academic prototype.
            The prediction is a machine-learning estimate and
            should not be treated as a medical diagnosis or as
            a replacement for professional healthcare judgement.
            """
        )


    except Exception as e:

        st.error(
            " Prediction failed."
        )

        st.exception(e)
