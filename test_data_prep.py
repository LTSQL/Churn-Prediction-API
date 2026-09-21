import pandas as pd
import pytest
from data_prep import (
    fix_total_charges, encode_binary_yes_no, simplify_service_columns,
    encode_categoricals, validate_schema, clean_customers,
)


def _sample_row(**overrides):
    row = {
        "customerID": "0001-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": "250.5",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def test_fix_total_charges_converts_blank_string_to_zero():
    df = pd.DataFrame([_sample_row(TotalCharges=" ")])
    out = fix_total_charges(df)
    assert out["TotalCharges"].iloc[0] == 0.0


def test_fix_total_charges_converts_valid_string_to_float():
    df = pd.DataFrame([_sample_row(TotalCharges="1889.5")])
    out = fix_total_charges(df)
    assert out["TotalCharges"].iloc[0] == 1889.5


def test_fix_total_charges_raises_without_column():
    df = pd.DataFrame({"other": [1]})
    with pytest.raises(KeyError):
        fix_total_charges(df)


def test_encode_binary_yes_no_maps_correctly():
    df = pd.DataFrame({"Partner": ["Yes", "No"]})
    out = encode_binary_yes_no(df, ["Partner"])
    assert out["Partner"].tolist() == [1, 0]


def test_encode_binary_yes_no_raises_on_bad_value():
    df = pd.DataFrame({"Partner": ["Yes", "Maybe"]})
    with pytest.raises(ValueError):
        encode_binary_yes_no(df, ["Partner"])


def test_simplify_service_columns_collapses_no_internet_service():
    df = pd.DataFrame([_sample_row()])
    out = simplify_service_columns(df)
    assert out["OnlineSecurity"].iloc[0] == 0
    assert out["OnlineBackup"].iloc[0] == 1


def test_encode_categoricals_creates_dummy_columns():
    df = pd.DataFrame({"Contract": ["Month-to-month", "Two year"]})
    out = encode_categoricals(df, ["Contract"])
    assert "Contract" not in out.columns
    assert "Contract_Month-to-month" in out.columns
    assert "Contract_Two year" in out.columns


def test_validate_schema_raises_when_columns_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df, ["a", "b"])


def test_clean_customers_end_to_end_drops_customer_id_and_encodes():
    df = pd.DataFrame([_sample_row(), _sample_row(customerID="0002-TEST", Churn="Yes")])
    out = clean_customers(df)
    assert "customerID" not in out.columns
    assert out["TotalCharges"].dtype.kind == "f"
    assert set(out["Churn"].unique()) <= {0, 1}
    assert "Contract_Month-to-month" in out.columns


def test_clean_customers_handles_blank_total_charges_row():
    df = pd.DataFrame([_sample_row(TotalCharges=" ", tenure=0)])
    out = clean_customers(df)
    assert out["TotalCharges"].iloc[0] == 0.0
