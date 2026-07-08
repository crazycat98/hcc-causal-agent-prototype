# ML Service

M1-M3 implements the deterministic prediction and SHAP explanation boundary for the prototype.

Safety notice: 演示用合成数据，非真实患者数据；非临床诊断依据。

## What It Does

- Generates synthetic tabular samples.
- Simulates FCI feature selection using a fixed causal candidate set.
- Trains a `RandomForestClassifier`.
- Reports five-fold cross-validation AUC.
- Saves a local model artifact.
- Exposes `/predict` and `/explain` through a small stdlib HTTP service.

## Why stdlib HTTP for M1-M3?

FastAPI is listed as a suitable service option in the overall plan. M1 started with Python's standard library HTTP server to keep the first milestone runnable. The endpoint contract is intentionally simple and can be moved to FastAPI in a later step.

## Commands

```powershell
python ml-service/scripts/train_model.py
python ml-service/scripts/predict_sample.py
python ml-service/scripts/run_server.py
```

`/explain` uses `shap.TreeExplainer` on the Random Forest step of the trained pipeline. The feature ranking is then checked against the fixed simulated FCI causal candidate set.

