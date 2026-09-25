"""
FastAPI service for the churn model (trained on the real Telco Customer
Churn dataset).
Run with: uvicorn main:app --reload
Docs at:  http://127.0.0.1:8000/docs
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from data_prep import clean_customers

app = FastAPI(title="Telco Customer Churn Prediction API", version="1.0.0")

# Model file sits right next to this script — no folder structure needed.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.joblib")
_bundle = None

YesNo = Literal["Yes", "No"]
ServiceOption = Literal["Yes", "No", "No internet service", "No phone service"]


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    tenure: int = Field(..., ge=0, le=100, example=5)
    PhoneService: YesNo = "Yes"
    MultipleLines: ServiceOption = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: ServiceOption = "No"
    OnlineBackup: ServiceOption = "No"
    DeviceProtection: ServiceOption = "No"
    TechSupport: ServiceOption = "No"
    StreamingTV: ServiceOption = "No"
    StreamingMovies: ServiceOption = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = "Electronic check"
    MonthlyCharges: float = Field(..., ge=0, example=85.0)
    TotalCharges: float = Field(..., ge=0, example=425.0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool


def load_model():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model file not found — run train_model.py first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    bundle = load_model()
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    raw = pd.DataFrame([customer.dict()])
    raw["TotalCharges"] = raw["TotalCharges"].astype(str)  # match raw CSV dtype expectations
    try:
        cleaned = clean_customers(raw, is_training=False)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Align columns to the exact set/order seen at training time (adds any
    # missing one-hot columns as 0, e.g. a category not present in this request)
    for col in feature_cols:
        if col not in cleaned.columns:
            cleaned[col] = 0
    cleaned = cleaned[feature_cols]

    prob = model.predict_proba(cleaned)[0, 1]
    return PredictionResponse(churn_probability=round(float(prob), 4), churn_prediction=prob >= 0.5)
