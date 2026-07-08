"""Synthetic HCC grading demo package.

Safety: 演示用合成数据，非真实患者数据；非临床诊断依据。
"""

from .features import CAUSAL_FEATURES, FEATURE_RANGES, SYNTHETIC_DATA_NOTICE
from .predictor import ModelBundle, load_model, predict_one

__all__ = [
    "CAUSAL_FEATURES",
    "FEATURE_RANGES",
    "SYNTHETIC_DATA_NOTICE",
    "ModelBundle",
    "load_model",
    "predict_one",
]

