"""Feature contract for the synthetic HCC grading demo.

The variables below are generated for demonstration only:
演示用合成数据，非真实患者数据；非临床诊断依据。
"""

from __future__ import annotations

from dataclasses import dataclass


SYNTHETIC_DATA_NOTICE = "演示用合成数据，非真实患者数据；非临床诊断依据。"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    minimum: float
    maximum: float
    description: str


CAUSAL_FEATURES: list[str] = [
    "tumor_size_cm",
    "afp_ng_ml",
    "alt_u_l",
    "ast_u_l",
    "bilirubin_umol_l",
    "albumin_g_l",
    "platelet_10e9_l",
    "portal_vein_invasion",
    "radiomics_entropy",
    "radiomics_glcm_contrast",
]


FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec("tumor_size_cm", 0.5, 20.0, "Synthetic tumor diameter in centimeters."),
    FeatureSpec("afp_ng_ml", 0.5, 50000.0, "Synthetic alpha-fetoprotein value."),
    FeatureSpec("alt_u_l", 5.0, 500.0, "Synthetic ALT value."),
    FeatureSpec("ast_u_l", 5.0, 500.0, "Synthetic AST value."),
    FeatureSpec("bilirubin_umol_l", 2.0, 150.0, "Synthetic total bilirubin value."),
    FeatureSpec("albumin_g_l", 15.0, 55.0, "Synthetic albumin value."),
    FeatureSpec("platelet_10e9_l", 20.0, 500.0, "Synthetic platelet count."),
    FeatureSpec("portal_vein_invasion", 0.0, 1.0, "Synthetic binary portal vein invasion flag."),
    FeatureSpec("radiomics_entropy", 2.0, 8.0, "Synthetic radiomics entropy feature."),
    FeatureSpec("radiomics_glcm_contrast", 10.0, 250.0, "Synthetic GLCM contrast feature."),
]


FEATURE_RANGES: dict[str, tuple[float, float]] = {
    spec.name: (spec.minimum, spec.maximum) for spec in FEATURE_SPECS
}


NON_CAUSAL_FEATURES: list[str] = [
    "age_years",
    "sex_male",
    "hbv_marker",
    "synthetic_noise_1",
    "synthetic_noise_2",
    "scanner_protocol_id",
]

