import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the SmartCare dataset.

    Operations:
    - Remove duplicate records
    - Handle missing room_type values
    - Convert appointment_date to datetime
    """

    df = df.copy()

    # Remove complete duplicate rows
    df = df.drop_duplicates()

    # room_type: None represents no room assignment
    if "room_type" in df.columns:
        df["room_type"] = df["room_type"].fillna("None")

    # Convert appointment date
    if "appointment_date" in df.columns:
        df["appointment_date"] = pd.to_datetime(
            df["appointment_date"],
            errors="coerce"
        )

    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of missing values.
    """

    missing = df.isnull().sum()

    result = pd.DataFrame({
        "Missing Values": missing,
        "Percentage": (
            missing / len(df) * 100
        ).round(2)
    })

    return result[
        result["Missing Values"] > 0
    ].sort_values(
        "Missing Values",
        ascending=False
    )


def detect_outliers(
    df: pd.DataFrame,
    numerical_columns=None
) -> pd.DataFrame:
    """
    Detect numerical outliers using the IQR method.
    """

    if numerical_columns is None:
        numerical_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

    results = []

    for column in numerical_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = (
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ).sum()

        results.append({
            "Feature": column,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outlier Count": outlier_count
        })

    return pd.DataFrame(results)