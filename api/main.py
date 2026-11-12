from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

app = FastAPI(title="API de predicción de diabetes", version="1.0")


class Paciente(BaseModel):
    Pregnancies: int = Field(ge=0, le=20)
    Glucose: Optional[float] = Field(default=None, gt=0, le=300)
    BloodPressure: Optional[float] = Field(default=None, gt=0, le=200)
    SkinThickness: Optional[float] = Field(default=None, gt=0, le=100)
    Insulin: Optional[float] = Field(default=None, gt=0, le=900)
    BMI: Optional[float] = Field(default=None, gt=0, le=80)
    DiabetesPedigreeFunction: float = Field(ge=0, le=3)
    Age: int = Field(ge=18, le=120)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(paciente: Paciente):
    df = pd.DataFrame([paciente.model_dump()])[FEATURES].astype("float64")
    proba = float(model.predict_proba(df)[0, 1])
    return {
        "prediccion": int(proba >= 0.5),
        "probabilidad_diabetes": round(proba, 4),
    }