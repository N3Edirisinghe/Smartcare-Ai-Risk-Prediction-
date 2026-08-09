import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.pipeline import Pipeline

from .cleaning import clean_data


# ============================================================
# Feature configuration
# ============================================================

NUMERICAL_FEATURES = [
    "age",
    "waiting_days",
    "previous_appointments",
    "missed_previous_appointments",
    "previous_missed_rate",
    "has_previous_appointment",
    "appointment_month",
    "appointment_dayofweek",
    "systolic_bp",
    "diastolic_bp",
    "blood_sugar_mg_dl",
    "cholesterol_mg_dl",
    "bmi"
]


CATEGORICAL_FEATURES = [
    "gender",
    "blood_group",
    "department",
    "diagnosis",
    "bp_category"
]


FEATURES = (
    NUMERICAL_FEATURES +
    CATEGORICAL_FEATURES
)


TARGET = "no_show"


# ============================================================
# Feature Engineering
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for SmartCare
    no-show prediction.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Appointment month
    # --------------------------------------------------------

    if "appointment_date" in df.columns:

        df["appointment_month"] = (
            df["appointment_date"].dt.month
        )

        # Monday = 0, Sunday = 6
        df["appointment_dayofweek"] = (
            df["appointment_date"].dt.dayofweek
        )

    # --------------------------------------------------------
    # Previous missed appointment rate
    # --------------------------------------------------------

    df["previous_missed_rate"] = np.where(
        df["previous_appointments"] > 0,

        df["missed_previous_appointments"] /
        df["previous_appointments"],

        0
    )

    # --------------------------------------------------------
    # Previous appointment indicator
    # --------------------------------------------------------

    df["has_previous_appointment"] = (
        df["previous_appointments"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Blood pressure category
    # --------------------------------------------------------

    def categorize_bp(row):

        systolic = row["systolic_bp"]
        diastolic = row["diastolic_bp"]

        if systolic < 120 and diastolic < 80:
            return "Normal"

        elif systolic < 130 and diastolic < 80:
            return "Elevated"

        elif systolic < 140 or diastolic < 90:
            return "High"

        else:
            return "Very High"

    df["bp_category"] = df.apply(
        categorize_bp,
        axis=1
    )

    return df


# ============================================================
# Create preprocessing transformer
# ============================================================

def create_preprocessor():
    """
    Create the complete preprocessing transformer.

    Numerical:
        StandardScaler

    Categorical:
        OneHotEncoder
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),

            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# ============================================================
# Complete preprocessing
# ============================================================

def prepare_dataset(df: pd.DataFrame):
    """
    Clean the dataset and perform feature engineering.
    """

    # Cleaning
    df = clean_data(df)

    # Feature engineering
    df = engineer_features(df)

    # Input features
    X = df[FEATURES].copy()

    # Target
    y = df[TARGET].copy()

    return X, y, df


# ============================================================
# Save processed dataset
# ============================================================

def save_processed_dataset(
    X_processed,
    y,
    feature_names,
    output_path
):
    """
    Save processed features and target as CSV.
    """

    processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names
    )

    processed_df[TARGET] = (
        y.reset_index(drop=True)
    )

    processed_df.to_csv(
        output_path,
        index=False
    )

    return processed_df