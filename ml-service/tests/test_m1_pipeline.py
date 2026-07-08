from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_synthetic.data import sample_payload
from hcc_synthetic.explanation import explain_one
from hcc_synthetic.predictor import load_model, predict_one
from hcc_synthetic.training import TrainingConfig, save_artifact, train_model


class M1M3PipelineTest(unittest.TestCase):
    def test_train_save_load_predict_and_explain(self) -> None:
        bundle = train_model(TrainingConfig(n_samples=300, random_state=123, n_splits=3))
        self.assertGreater(bundle["metrics"]["cv_auc_mean"], 0.65)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "model.joblib"
            save_artifact(bundle, artifact)
            loaded = load_model(artifact)
            features = sample_payload(random_state=11)
            prediction = predict_one(loaded, features)
            explanation = explain_one(loaded, features, top_n=5)

        self.assertIn("prediction", prediction)
        self.assertIn("演示用合成数据", prediction["safety_notice"])
        self.assertEqual(len(prediction["features_used"]), 10)
        self.assertEqual(explanation["shap"]["method"], "shap.TreeExplainer")
        self.assertEqual(len(explanation["shap"]["top_features"]), 5)
        self.assertTrue(explanation["shap"]["high_trust_features"])


if __name__ == "__main__":
    unittest.main()

