from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_synthetic.training import TrainingConfig, save_artifact, train_model


def main() -> None:
    artifact_path = ROOT / "artifacts" / "hcc_rf_pipeline.joblib"
    bundle = train_model(TrainingConfig())
    save_artifact(bundle, artifact_path)
    metrics = bundle["metrics"]
    print("M1 training complete")
    print(f"artifact={artifact_path}")
    print(f"cv_auc_mean={metrics['cv_auc_mean']:.3f}")
    print(f"cv_auc_std={metrics['cv_auc_std']:.3f}")
    print(f"train_auc={metrics['train_auc']:.3f}")
    print(f"positive_rate={metrics['positive_rate']:.3f}")
    print(bundle["safety_notice"])


if __name__ == "__main__":
    main()

