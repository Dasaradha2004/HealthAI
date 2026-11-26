"""SHAP, LIME, and human-readable explanations for ensemble predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

DISCLAIMER = (
    "This demonstration is not a diagnosis or medical advice. Its synthetic-data "
    "models are not clinically validated. Please consult a qualified healthcare professional."
)


def _positive_class_shap(explainer: shap.TreeExplainer, row: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(row)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[0, :, -1]
    elif values.ndim == 2:
        values = values[0]
    return values.astype(float)


class PredictionExplainer:
    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        self.feature_names = artifact["feature_names"]
        self.models = artifact["models"]
        self.weights = artifact["weights"]
        self.lime = LimeTabularExplainer(
            np.asarray(artifact["training_matrix"], dtype=float),
            feature_names=self.feature_names,
            class_names=["lower risk", "higher risk"],
            mode="classification",
            discretize_continuous=False,
            random_state=23,
        )
        self.shap_explainers = {
            name: shap.TreeExplainer(model) for name, model in self.models.items()
        }

    def explain(
        self, transformed_row: np.ndarray, original_values: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        row = np.asarray(transformed_row, dtype=float).reshape(1, -1)
        contributions = np.zeros(len(self.feature_names), dtype=float)
        for model_name, explainer in self.shap_explainers.items():
            contributions += self.weights[model_name] * _positive_class_shap(explainer, row)
        shap_rows = []
        for index in np.argsort(np.abs(contributions))[::-1]:
            transformed_name = self.feature_names[index]
            feature_name = transformed_name
            value: Any = round(float(row[0, index]), 4)
            if original_values:
                categorical = next(
                    (name for name in self.artifact["categorical_features"] if transformed_name.startswith(f"{name}_")),
                    None,
                )
                numeric = next(
                    (name for name in self.artifact["numeric_features"] if transformed_name == name),
                    None,
                )
                if categorical:
                    category = transformed_name[len(categorical) + 1:]
                    feature_name = f"{categorical} ({category})"
                    value = original_values[categorical]
                elif numeric:
                    feature_name = numeric
                    value = original_values[numeric]
            contribution = float(contributions[index])
            shap_rows.append({
                "feature": feature_name,
                "value": value,
                "contribution": round(contribution, 5),
                "direction": "increases" if contribution >= 0 else "decreases",
            })
        lime_rows = []
        lime_explanation = self.lime.explain_instance(
            row[0],
            lambda matrix: self._ensemble_probabilities(matrix),
            num_features=min(10, len(self.feature_names)),
            labels=(1,),
        )
        for feature_index, weight in lime_explanation.as_map().get(1, []):
            lime_name = self.feature_names[feature_index]
            categorical = next(
                (name for name in self.artifact["categorical_features"] if lime_name.startswith(f"{name}_")),
                None,
            )
            if categorical:
                lime_name = f"{categorical} ({lime_name[len(categorical) + 1:]})"
            lime_rows.append({
                "feature": lime_name,
                "contribution": round(float(weight), 5),
                "direction": "increases" if weight >= 0 else "decreases",
            })
        return shap_rows, lime_rows

    def _ensemble_probabilities(self, matrix: np.ndarray) -> np.ndarray:
        scores = np.zeros(matrix.shape[0], dtype=float)
        for model_name, model in self.models.items():
            scores += self.weights[model_name] * model.predict_proba(matrix)[:, 1]
        return np.column_stack((1 - scores, scores))


def create_natural_language_explanation(
    risk_level: str,
    shap_features: list[dict[str, Any]],
    demo_data: bool,
) -> str:
    positive = [item["feature"].replace("_", " ") for item in shap_features if item["contribution"] > 0][:3]
    negative = [item["feature"].replace("_", " ") for item in shap_features if item["contribution"] < 0][:2]
    parts = [f"The model assigned a {risk_level.lower()} risk category."]
    if positive:
        parts.append("Factors pushing the estimate higher include " + ", ".join(positive) + ".")
    if negative:
        parts.append("Factors pushing the estimate lower include " + ", ".join(negative) + ".")
    if demo_data:
        parts.append("This model was trained on synthetic demonstration data, so its estimate is not medically validated.")
    parts.append(DISCLAIMER)
    return " ".join(parts)
