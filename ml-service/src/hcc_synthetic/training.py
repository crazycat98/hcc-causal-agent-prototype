"""Training pipeline for the synthetic HCC grading model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .causal import simulate_fci_selection
from .data import as_matrix, generate_synthetic_dataset
from .features import SYNTHETIC_DATA_NOTICE


@dataclass(frozen=True)
class TrainingConfig:
    n_samples: int = 1400
    random_state: int = 42
    n_splits: int = 5


def build_model(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=320,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(config: TrainingConfig) -> dict[str, Any]:
    features, y = generate_synthetic_dataset(
        n_samples=config.n_samples,
        random_state=config.random_state,
    )
    selection = simulate_fci_selection(list(features.keys()))
    selected_features = selection["selected_features"]
    if not isinstance(selected_features, list):
        raise TypeError("Feature selector returned an invalid selected feature list.")

    x = as_matrix(features, selected_features)
    model = build_model(config.random_state)
    cv = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    auc_scores = cross_val_score(model, x, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    model.fit(x, y)
    train_prob = model.predict_proba(x)[:, 1]

    return {
        "model": model,
        "selected_features": selected_features,
        "feature_selection": selection,
        "metrics": {
            "cv_auc_mean": float(np.mean(auc_scores)),
            "cv_auc_std": float(np.std(auc_scores)),
            "cv_auc_scores": [float(score) for score in auc_scores],
            "train_auc": float(roc_auc_score(y, train_prob)),
            "positive_rate": float(np.mean(y)),
            "n_samples": config.n_samples,
            "n_splits": config.n_splits,
        },
        "safety_notice": SYNTHETIC_DATA_NOTICE,
        "config": asdict(config),
    }


def save_artifact(bundle: dict[str, Any], artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)

    metrics_path = artifact_path.with_suffix(".metrics.json")
    serializable = {
        "selected_features": bundle["selected_features"],
        "feature_selection": bundle["feature_selection"],
        "metrics": bundle["metrics"],
        "safety_notice": bundle["safety_notice"],
        "config": bundle["config"],
    }
    metrics_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

