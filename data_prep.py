"""
Data cleaning functions for the IBM/Kaggle Telco Customer Churn dataset,
written to be unit tested (see test_data_prep.py).

Source: Telco Customer Churn (IBM Sample Data Sets), via Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""
import pandas as pd

BINARY_YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
]

SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

CATEGORICAL_COLS = ["InternetService", "Contract", "PaymentMethod"]


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges is read as a string in the raw file because 11 rows
    contain blank whitespace instead of a number (all are customers with
    0 months tenure — they haven't been billed yet). Convert to numeric,
    filling those with 0 rather than dropping the rows."""
    df = df.copy()
    if "TotalCharges" not in df.columns:
        raise KeyError("Expected a 'TotalCharges' column")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def encode_binary_yes_no(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    """Map Yes/No columns to 1/0. Returns a copy."""
    df = df.copy()
    columns = columns or BINARY_YES_NO_COLS
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe")
        df[col] = df[col].map({"Yes": 1, "No": 0})
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains values other than Yes/No")
    return df


def simplify_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Several service columns use 'No internet service' / 'No phone service'
    as a third category, which is redundant with InternetService/PhoneService
    already being 0. Collapse those to a plain 'No' before encoding."""
    df = df.copy()
    for col in SERVICE_COLS:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe")
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def encode_categoricals(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    """One-hot encode multi-category columns (InternetService, Contract,
    PaymentMethod, gender). Returns a copy."""
    df = df.copy()
    columns = columns or CATEGORICAL_COLS
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe")
    dummies = pd.get_dummies(df[columns], prefix=columns)
    df = pd.concat([df.drop(columns=columns), dummies], axis=1)
    return df


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error if expected columns are missing, instead of
    failing silently or cryptically further down the pipeline."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clean_customers(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline used by both training and inference, so the
    two paths can never silently drift apart (a common real-world MLOps bug).

    is_training=True keeps the Churn column and customerID drop; at
    inference time the API sends rows without a Churn column, so it's
    handled separately in main.py.
    """
    required = ["tenure", "MonthlyCharges", "TotalCharges", "Contract",
                "InternetService", "PaymentMethod", "gender"] + SERVICE_COLS + \
               ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    validate_schema(df, required)

    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df = fix_total_charges(df)

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    if is_training and "Churn" in df.columns:
        binary_cols = binary_cols + ["Churn"]
    df = encode_binary_yes_no(df, binary_cols)

    df = simplify_service_columns(df)
    df = encode_categoricals(df, CATEGORICAL_COLS + ["gender"])

    return df
