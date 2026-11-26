"""Train the demo CKD and diabetes ensembles.

Place optional source CSVs at backend/data/ckd.csv and
backend/data/diabetes.csv. Without them, reproducible synthetic data is used.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
MODEL_DIR = BACKEND_DIR / "models"
RANDOM_STATE = 42

CKD_NUMERIC = [
    "age", "blood_pressure", "specific_gravity", "albumin", "sugar",
    "blood_glucose_random", "blood_urea", "serum_creatinine", "sodium",
    "potassium", "hemoglobin", "packed_cell_volume", "red_blood_cell_count",
]
CKD_CATEGORICAL = [
    "red_blood_cells", "pus_cell", "pus_cell_clumps", "bacteria",
    "hypertension", "diabetes_mellitus",
]
CKD_FEATURES = CKD_NUMERIC + CKD_CATEGORICAL
DIABETES_NUMERIC = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
    "bmi", "diabetes_pedigree_function", "age",
]
MODEL_NAMES = ("xgboost", "random_forest", "lightgbm")
WEIGHTS = {"xgboost": 0.40, "random_forest": 0.35, "lightgbm": 0.25}


def _synthetic_ckd(rows: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.integers(18, 90, rows)
    bp = np.clip(rng.normal(78, 16, rows), 45, 180)
    creatinine = np.clip(rng.lognormal(0.1, 0.75, rows), 0.3, 12)
    urea = np.clip(rng.normal(38, 22, rows), 5, 200)
    hemoglobin = np.clip(rng.normal(13, 2.2, rows), 4, 20)
    albumin = rng.choice([0, 1, 2, 3, 4, 5], rows, p=[.64, .15, .09, .06, .04, .02])
    logit = -4.1 + .035 * (age - 45) + .025 * (bp - 80) + 1.15 * creatinine + .018 * (urea - 35) - .24 * (hemoglobin - 12) + .55 * albumin
    probability = 1 / (1 + np.exp(-logit))
    disease = rng.binomial(1, probability)
    frame = pd.DataFrame({
        "age": age, "blood_pressure": bp,
        "specific_gravity": rng.choice([1.005, 1.010, 1.015, 1.020, 1.025], rows),
        "albumin": albumin, "sugar": rng.choice([0, 1, 2, 3, 4, 5], rows),
        "blood_glucose_random": np.clip(rng.normal(130, 45, rows), 55, 400),
        "blood_urea": urea, "serum_creatinine": creatinine,
        "sodium": np.clip(rng.normal(138, 5, rows), 110, 160),
        "potassium": np.clip(rng.normal(4.5, .8, rows), 2, 9),
        "hemoglobin": hemoglobin,
        "packed_cell_volume": np.clip(rng.normal(40, 8, rows), 10, 60),
        "red_blood_cell_count": np.clip(rng.normal(4.5, .9, rows), 1.5, 8),
        "red_blood_cells": np.where(disease & (rng.random(rows) < .7), "abnormal", "normal"),
        "pus_cell": np.where(disease & (rng.random(rows) < .6), "abnormal", "normal"),
        "pus_cell_clumps": np.where(disease & (rng.random(rows) < .5), "present", "notpresent"),
        "bacteria": np.where(disease & (rng.random(rows) < .3), "present", "notpresent"),
        "hypertension": np.where((bp > 95) | (disease & (rng.random(rows) < .45)), "yes", "no"),
        "diabetes_mellitus": np.where(disease & (rng.random(rows) < .35), "yes", "no"),
        "target": disease,
    })
    for column in CKD_NUMERIC + CKD_CATEGORICAL:
        frame.loc[rng.random(rows) < .025, column] = np.nan
    return frame


def _synthetic_diabetes(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 1)
    pregnancies = rng.integers(0, 14, rows)
    glucose = np.clip(rng.normal(120, 32, rows), 45, 220)
    bmi = np.clip(rng.normal(32, 7, rows), 15, 60)
    age = rng.integers(18, 82, rows)
    logit = -3.3 + .035 * (glucose - 100) + .055 * (bmi - 25) + .025 * (age - 35) + .08 * pregnancies
    probability = 1 / (1 + np.exp(-logit))
    frame = pd.DataFrame({
        "pregnancies": pregnancies, "glucose": glucose,
        "blood_pressure": np.clip(rng.normal(72, 12, rows), 35, 130),
        "skin_thickness": np.clip(rng.normal(22, 10, rows), 0, 70),
        "insulin": np.clip(rng.lognormal(4.5, .8, rows), 0, 900),
        "bmi": bmi, "diabetes_pedigree_function": rng.gamma(2, .2, rows),
        "age": age, "target": rng.binomial(1, probability),
    })
    for column in DIABETES_NUMERIC:
        frame.loc[rng.random(rows) < .04, column] = np.nan
    return frame


def _load_dataset(path: Path, features: list[str], fallback: Any) -> pd.DataFrame:
    if not path.exists():
        print(f"{path.name} not found; training with reproducible synthetic demo data.")
        return fallback()
    frame = pd.read_csv(path, na_values=["?", " ?"])
    aliases = {
        "bp": "blood_pressure", "sg": "specific_gravity", "al": "albumin",
        "su": "sugar", "rbc": "red_blood_cells", "pc": "pus_cell",
        "pcc": "pus_cell_clumps", "ba": "bacteria", "bgr": "blood_glucose_random",
        "bu": "blood_urea", "sc": "serum_creatinine", "sod": "sodium",
        "pot": "potassium", "hemo": "hemoglobin", "pcv": "packed_cell_volume",
        "rbcc": "red_blood_cell_count", "htn": "hypertension", "dm": "diabetes_mellitus",
        "bloodpressure": "blood_pressure", "skinthickness": "skin_thickness",
        "diabetespedigreefunction": "diabetes_pedigree_function",
        "redbloodcells": "red_blood_cells", "puscell": "pus_cell",
        "puscellclumps": "pus_cell_clumps", "bloodglucoserandom": "blood_glucose_random",
        "bloodurea": "blood_urea", "serumcreatinine": "serum_creatinine",
        "packedcellvolume": "packed_cell_volume", "redbloodcellcount": "red_blood_cell_count",
        "specificgravity": "specific_gravity", "diabetesmellitus": "diabetes_mellitus",
    }
    frame.columns = [
        aliases.get(re.sub(r"[^a-z0-9]", "", str(column).lower()), re.sub(r"[^a-z0-9]", "", str(column).lower()))
        for column in frame.columns
    ]
    target_column = next((name for name in ("target", "class", "classification", "outcome") if name in frame.columns), None)
    if target_column is None:
        raise ValueError(f"{path} must contain a target/classification/Outcome column.")
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame = frame[features + [target_column]].copy()
    frame.rename(columns={target_column: "target"}, inplace=True)
    frame["target"] = frame["target"].map(_normalize_target)
    if frame["target"].isna().any():
        raise ValueError(f"{path} contains unrecognized target values; use 0/1, positive/negative, or ckd/notckd.")
    return frame


def _normalize_target(value: Any) -> int | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "yes", "positive", "ckd", "ckd.", "diabetic"}:
        return 1
    if normalized in {"0", "no", "negative", "notckd", "not ckd", "notckd.", "non-diabetic"}:
        return 0
    return None


def _clean_categories(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in CKD_CATEGORICAL:
        frame[column] = frame[column].map(
            lambda value: str(value).strip().lower().replace(" ", "") if pd.notna(value) else np.nan
        )
        frame[column] = frame[column].replace({
            "normal": "normal", "abnormal": "abnormal",
            "present": "present", "notpresent": "notpresent",
            "yes": "yes", "no": "no", "": np.nan,
        })
    return frame


def _build_preprocessor(numeric: list[str], categorical: list[str], use_knn: bool) -> ColumnTransformer:
    numeric_imputer = KNNImputer(n_neighbors=5) if use_knn else SimpleImputer(strategy="median")
    numeric_pipeline = Pipeline([("imputer", numeric_imputer), ("scaler", StandardScaler())])
    transformers: list[tuple[str, Any, list[str]]] = [("numeric", numeric_pipeline, numeric)]
    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _build_models() -> dict[str, Any]:
    return {
        "xgboost": XGBClassifier(
            n_estimators=180, max_depth=4, learning_rate=.05, subsample=.9,
            colsample_bytree=.9, eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=4,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=240, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=4,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=180, max_depth=-1, learning_rate=.05, verbosity=-1,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=4,
        ),
    }


def _fit_condition(name: str, frame: pd.DataFrame, numeric: list[str], categorical: list[str], use_knn: bool) -> None:
    X = frame[numeric + categorical].copy()
    if categorical:
        X = _clean_categories(X)
    y = frame["target"].astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(f"{name} training data must contain at least two examples of each class.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE
    )
    preprocessor = _build_preprocessor(numeric, categorical, use_knn)
    train_matrix = preprocessor.fit_transform(X_train)
    test_matrix = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    models = _build_models()
    metrics: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        model.fit(train_matrix, y_train)
        probabilities = model.predict_proba(test_matrix)[:, 1]
        metrics[model_name] = {
            "accuracy": float(accuracy_score(y_test, probabilities >= .5)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        print(f"{name} {model_name}: accuracy={metrics[model_name]['accuracy']:.3f}, AUC={metrics[model_name]['roc_auc']:.3f}")
    artifact = {
        "condition": name,
        "feature_names": feature_names,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "preprocessor": preprocessor,
        "models": models,
        "training_matrix": train_matrix,
        "weights": WEIGHTS,
        "metrics": metrics,
        "demo_data": not (DATA_DIR / f"{name}.csv").exists(),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_DIR / f"{name}_ensemble.joblib")
    print(f"Saved {name} ensemble to {MODEL_DIR / f'{name}_ensemble.joblib'}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ckd = _load_dataset(DATA_DIR / "ckd.csv", CKD_FEATURES, _synthetic_ckd)
    diabetes = _load_dataset(DATA_DIR / "diabetes.csv", DIABETES_NUMERIC, _synthetic_diabetes)
    for column in ("glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"):
        if (DATA_DIR / "diabetes.csv").exists():
            diabetes.loc[diabetes[column] == 0, column] = np.nan
    _fit_condition("ckd", ckd, CKD_NUMERIC, CKD_CATEGORICAL, use_knn=True)
    _fit_condition("diabetes", diabetes, DIABETES_NUMERIC, [], use_knn=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise
