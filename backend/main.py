from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from backend.database import Base, PredictionLog, SessionLocal, engine
from backend.ml.explainer import PredictionExplainer, create_natural_language_explanation

MODEL_DIR = Path(__file__).resolve().parent / "models"
artifacts: dict[str, dict[str, Any]] = {}
explainers: dict[str, PredictionExplainer] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    for condition in ("ckd", "diabetes"):
        model_path = MODEL_DIR / f"{condition}_ensemble.joblib"
        if model_path.exists():
            artifact = joblib.load(model_path)
            artifacts[condition] = artifact
            explainers[condition] = PredictionExplainer(artifact)
    yield


app = FastAPI(title="HealthAI Risk Assessment API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClinicalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CKDInput(ClinicalInput):
    age: float = Field(ge=0, le=120)
    blood_pressure: float = Field(ge=0, le=300)
    specific_gravity: float = Field(ge=1.0, le=1.1)
    albumin: float = Field(ge=0, le=5)
    sugar: float = Field(ge=0, le=5)
    red_blood_cells: Literal["normal", "abnormal"]
    pus_cell: Literal["normal", "abnormal"]
    pus_cell_clumps: Literal["present", "notpresent"]
    bacteria: Literal["present", "notpresent"]
    blood_glucose_random: float = Field(ge=0, le=1000)
    blood_urea: float = Field(ge=0, le=500)
    serum_creatinine: float = Field(ge=0, le=100)
    sodium: float = Field(ge=0, le=250)
    potassium: float = Field(ge=0, le=30)
    hemoglobin: float = Field(ge=0, le=30)
    packed_cell_volume: float = Field(ge=0, le=100)
    hypertension: Literal["yes", "no"]
    diabetes_mellitus: Literal["yes", "no"]
    red_blood_cell_count: float = Field(ge=0, le=20)


class DiabetesInput(ClinicalInput):
    pregnancies: float = Field(ge=0, le=30)
    glucose: float = Field(ge=0, le=500)
    blood_pressure: float = Field(ge=0, le=300)
    skin_thickness: float = Field(ge=0, le=150)
    insulin: float = Field(ge=0, le=3000)
    bmi: float = Field(ge=0, le=100)
    diabetes_pedigree_function: float = Field(ge=0, le=10)
    age: float = Field(ge=0, le=120)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def risk_level(score: float) -> str:
    if score < 30:
        return "LOW"
    if score < 60:
        return "MODERATE"
    if score < 80:
        return "HIGH"
    return "CRITICAL"


def _predict(condition: str, request: BaseModel, db: Session) -> dict[str, Any]:
    if condition not in artifacts:
        raise HTTPException(
            status_code=503,
            detail=f"The {condition} model is not available. Run `python -m backend.ml.train_models` from the project root first.",
        )
    artifact = artifacts[condition]
    values = request.model_dump()
    if condition == "ckd":
        for key in ("red_blood_cells", "pus_cell", "pus_cell_clumps", "bacteria", "hypertension", "diabetes_mellitus"):
            values[key] = values[key].strip().lower().replace(" ", "")
    ordered = {feature: values[feature] for feature in artifact["numeric_features"] + artifact["categorical_features"]}
    row_frame = pd.DataFrame([ordered])
    transformed = artifact["preprocessor"].transform(row_frame)
    probabilities = {
        name: float(model.predict_proba(transformed)[0, 1])
        for name, model in artifact["models"].items()
    }
    raw_score = sum(artifact["weights"][name] * probabilities[name] for name in probabilities)
    score = round(float(np.clip(raw_score * 100, 0, 100)), 1)
    level = risk_level(score)
    shap_features, lime_weights = explainers[condition].explain(transformed, ordered)
    explanation = create_natural_language_explanation(level, shap_features, artifact["demo_data"])
    log = PredictionLog(condition=condition, risk_score=score, risk_level=level)
    db.add(log)
    db.commit()
    db.refresh(log)
    return {
        "condition": condition,
        "risk_score": score,
        "risk_level": level,
        "model_probabilities": {name: round(value * 100, 2) for name, value in probabilities.items()},
        "shap_features": shap_features,
        "lime_weights": lime_weights,
        "explanation": explanation,
        "demo_data": artifact["demo_data"],
        "history_id": log.id,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models": {condition: condition in artifacts for condition in ("ckd", "diabetes")},
        "xai": "SHAP + LIME",
    }


@app.post("/api/predict/ckd")
def predict_ckd(request: CKDInput, db: Session = Depends(get_db)) -> dict[str, Any]:
    return _predict("ckd", request, db)


@app.post("/api/predict/diabetes")
def predict_diabetes(request: DiabetesInput, db: Session = Depends(get_db)) -> dict[str, Any]:
    return _predict("diabetes", request, db)


@app.get("/api/history")
def history(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    entries = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(200).all()
    return [
        {
            "id": entry.id,
            "condition": entry.condition,
            "risk_score": entry.risk_score,
            "risk_level": entry.risk_level,
            "created_at": entry.created_at.isoformat(),
        }
        for entry in entries
    ]
