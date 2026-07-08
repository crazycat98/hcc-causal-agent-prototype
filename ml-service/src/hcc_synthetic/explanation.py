"""SHAP explanation utilities for the synthetic HCC grading demo.

Safety: 演示用合成数据，非真实患者数据；非临床诊断依据。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shap

from .features import CAUSAL_FEATURES, SYNTHETIC_DATA_NOTICE
from .predictor import ModelBundle, predict_one, validate_features


def _class_one_shap_values(raw_shap_values: Any) -> np.ndarray:
    """Return class-1 SHAP values for a single sample across SHAP versions."""

    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) < 2:
            raise ValueError("Expected binary-class SHAP values.")
        return np.asarray(raw_shap_values[1])[0]

    values = np.asarray(raw_shap_values)
    if values.ndim == 3:
        if values.shape[0] == 1 and values.shape[2] >= 2:
            return values[0, :, 1]
        if values.shape[0] >= 2 and values.shape[1] == 1:
            return values[1, 0, :]
    if values.ndim == 2:
        return values[0]

    raise ValueError(f"Unsupported SHAP values shape: {values.shape}")


def _class_one_expected_value(expected_value: Any) -> float:
    expected = np.asarray(expected_value)
    if expected.ndim == 0:
        return float(expected)
    if expected.shape[0] >= 2:
        return float(expected[1])
    return float(expected[0])


def _trust_for_feature(feature_name: str) -> tuple[str, str]:
    if feature_name in CAUSAL_FEATURES:
        return (
            "high_trust_causal_candidate",
            "该特征同时位于 SHAP Top-N 与模拟 FCI 因果候选集合中，可作为较高可信的模型解释线索。",
        )

    return (
        "statistical_association_only",
        "该特征不在模拟 FCI 因果候选集合中，仅表示模型统计相关贡献，建议谨慎解读。",
    )


def explain_one(bundle: ModelBundle, features: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
    if top_n < 1 or top_n > len(bundle.selected_features):
        raise ValueError(f"top_n must be between 1 and {len(bundle.selected_features)}.")

    validated = validate_features(features, bundle.selected_features)
    x_raw = np.array([[validated[name] for name in bundle.selected_features]], dtype=float)

    scaler = bundle.model.named_steps["scaler"]
    rf_model = bundle.model.named_steps["rf"]
    x_scaled = scaler.transform(x_raw)

    explainer = shap.TreeExplainer(rf_model)
    raw_shap_values = explainer.shap_values(x_scaled)
    class_one_values = _class_one_shap_values(raw_shap_values)
    base_value = _class_one_expected_value(explainer.expected_value)

    ranked_indices = np.argsort(np.abs(class_one_values))[::-1][:top_n]
    causal_candidates = set(CAUSAL_FEATURES)
    top_features: list[dict[str, Any]] = []

    for index in ranked_indices:
        feature_name = bundle.selected_features[int(index)]
        shap_value = float(class_one_values[int(index)])
        trust_level, consistency_note = _trust_for_feature(feature_name)
        top_features.append(
            {
                "feature": feature_name,
                "value": validated[feature_name],
                "shap_value": shap_value,
                "abs_shap_value": abs(shap_value),
                "direction": "pushes_toward_high_grade"
                if shap_value >= 0
                else "pushes_toward_low_or_intermediate",
                "trust_level": trust_level,
                "consistency_note": consistency_note,
            }
        )

    high_trust_features = [
        item["feature"] for item in top_features if item["feature"] in causal_candidates
    ]
    statistical_only_features = [
        item["feature"] for item in top_features if item["feature"] not in causal_candidates
    ]

    if high_trust_features:
        consistency_summary = "SHAP Top-N 与模拟 FCI 因果候选集合存在交集；交集特征标记为高可信解释线索。"
    else:
        consistency_summary = "SHAP Top-N 与模拟 FCI 因果候选集合无交集；本次解释仅可作为统计相关线索。"

    return {
        "safety_notice": SYNTHETIC_DATA_NOTICE,
        "prediction": predict_one(bundle, features)["prediction"],
        "shap": {
            "method": "shap.TreeExplainer",
            "target_class": "synthetic_high_pathology_grade",
            "top_n": top_n,
            "base_value": base_value,
            "top_features": top_features,
            "causal_candidate_features": list(CAUSAL_FEATURES),
            "high_trust_features": high_trust_features,
            "statistical_only_features": statistical_only_features,
            "consistency_summary": consistency_summary,
            "caveat": "SHAP 解释反映模型内部贡献，不等同于真实医学因果效应；本原型仅用于工程演示。",
        },
        "disclaimer": "该输出仅用于科研学习与工程演示，不作为任何临床诊断、治疗或分级依据。",
    }

