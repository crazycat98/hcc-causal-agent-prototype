from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_synthetic.data import sample_payload
from hcc_synthetic.predictor import load_model, predict_one


def main() -> None:
    artifact_path = ROOT / "artifacts" / "hcc_rf_pipeline.joblib"
    if not artifact_path.exists():
        raise SystemExit("Model artifact is missing. Run: python ml-service/scripts/train_model.py")

    bundle = load_model(artifact_path)
    features = sample_payload()
    result = predict_one(bundle, features)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

