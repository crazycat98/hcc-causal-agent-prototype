"""Prediction utilities for the deterministic ML boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import FEATURE_RANGES, SYNTHETIC_DATA_NOTICE


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    selected_features: list[str]
    feature_selection: dict[str, Any]
    metrics: dict[str, Any]
    safety_notice: str


def load_model(artifact_path: Path) -> ModelBundle:
    raw = joblib.load(artifact_path)
    return ModelBundle(
        model=raw["model"],
        selected_features=list(raw["selected_features"]),
        feature_selection=dict(raw["feature_selection"]),
        metrics=dict(raw["metrics"]),
        safety_notice=str(raw.get("safety_notice", SYNTHETIC_DATA_NOTICE)),
    )


def validate_features(payload: dict[str, Any], selected_features: list[str]) -> dict[str, float]:
    missing = [name for name in selected_features if name not in payload]
    if missing:
        raise ValueError(f"Missing required synthetic feature(s): {', '.join(missing)}")

    unknown = sorted(set(payload) - set(selected_features))
    if unknown:
        raise ValueError(f"Unknown feature(s) for M1 contract: {', '.join(unknown)}")

    validated: dict[str, float] = {}
    for name in selected_features:
        value = payload[name]
        if isinstance(value, bool):
            numeric = float(int(value))
        elif isinstance(value, (int, float)):
            numeric = float(value)
        else:
            raise ValueError(f"Feature {name} must be numeric.")

        minimum, maximum = FEATURE_RANGES[name]
        if numeric < minimum or numeric > maximum:
            raise ValueError(f"Feature {name}={numeric} is outside allowed range [{minimum}, {maximum}].")

        if name == "portal_vein_invasion" and numeric not in (0.0, 1.0):
            raise ValueError("portal_vein_invasion must be 0 or 1.")

        validated[name] = numeric

    return validated


def predict_one(bundle: ModelBundle, features: dict[str, Any]) -> dict[str, Any]:
    validated = validate_features(features, bundle.selected_features)
    x = np.array([[validated[name] for name in bundle.selected_features]], dtype=float)
    probability_high_grade = float(bundle.model.predict_proba(x)[0][1])
    label = (
        "synthetic_high_pathology_grade"
        if probability_high_grade >= 0.5
        else "synthetic_low_or_intermediate_grade"
    )

    return {
        "safety_notice": SYNTHETIC_DATA_NOTICE,
        "prediction": {
            "label": label,
            "probability_high_grade": probability_high_grade,
            "probability_low_or_intermediate": 1.0 - probability_high_grade,
            "uncertain_probability_band": 0.4 <= probability_high_grade <= 0.6,
        },
        "features_used": list(bundle.selected_features),
        "input_echo": validated,
        "model": {
            "type": "RandomForestClassifier",
            "feature_selection_method": bundle.feature_selection.get("method"),
            "cv_auc_mean": bundle.metrics.get("cv_auc_mean"),
            "cv_auc_std": bundle.metrics.get("cv_auc_std"),
        },
        "disclaimer": "该输出仅用于科研学习与工程演示，不作为任何临床诊断、治疗或分级依据。",
    }

