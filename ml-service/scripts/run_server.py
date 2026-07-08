from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_synthetic.server import run_server


def main() -> None:
    artifact_path = ROOT / "artifacts" / "hcc_rf_pipeline.joblib"
    if not artifact_path.exists():
        raise SystemExit("Model artifact is missing. Run: python ml-service/scripts/train_model.py")
    run_server(artifact_path)


if __name__ == "__main__":
    main()

