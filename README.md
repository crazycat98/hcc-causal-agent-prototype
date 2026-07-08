# HCC Causal Agent Prototype

This project is a research-learning and portfolio prototype. It simulates a closed-loop workflow for HCC pathology-grading prediction:

```text
synthetic patient features
        |
Agent tool loop
        |
deterministic ML prediction
        |
SHAP explanation + causal-candidate consistency check
        |
public-source hybrid retrieval with citations
        |
session memory + cross-session case memory
        |
evaluation loop
        |
interactive Next.js demo
        |
structured non-clinical report
```

## Mandatory Safety Statement

- 演示用合成数据，非真实患者数据。
- 本系统不作为任何临床诊断依据，不应替代医生判断、病理检查、影像检查或任何正式医疗流程。
- 当前医学与模型输出仅用于工程架构演示。检索语料限制为公开、可追溯、经过自行整理改写的短文本，并保留来源。
- LLM 层不得绕过 Tool 输出自行编造医学判断或诊断结论；风险预测、SHAP 数值、一致性标签、医学引用和病例记忆必须来自确定性 Tool。

## Current Progress

### M1: Synthetic Data + Simulated FCI + Random Forest + Prediction Service

- Generates synthetic data only.
- Simulates FCI-selected causal candidate features with a fixed documented feature set.
- Trains a `RandomForestClassifier`.
- Reports five-fold cross-validation AUC.
- Exposes CLI and local HTTP prediction interfaces.

Latest local result:

```text
cv_auc_mean=0.828
cv_auc_std=0.017
train_auc=0.992
positive_rate=0.540
```

### M2: Vercel AI SDK Agent Tool Loop

- Uses Vercel AI SDK 7 `generateText`, `tool`, and `stopWhen: isStepCount(n)`.
- Defines strict Zod schemas for the 10 synthetic causal candidate features.
- Uses `MockLanguageModelV4` for an offline deterministic demo when no real LLM API key is available.

### M3: SHAP Explanation + Causal-SHAP Consistency

- Adds Python `/explain` endpoint using `shap.TreeExplainer`.
- Returns Top-N single-sample SHAP feature contributions.
- Checks each SHAP Top-N feature against the fixed simulated FCI causal candidate set.

### M4: Hybrid Retrieval + Citation Constraints

- Adds `/retrieval` with a 20-entry public-source knowledge base.
- Implements BM25 + local hash embedding + heuristic rerank.
- Adds `retrieveMedicalEvidence` as a Vercel AI SDK Tool.
- Final report cites returned paragraphs with `[KB-ID]` and source URLs.

### M5: Layered Memory + Confidence-Driven Clarification

- Adds `/memory` with local JSON storage.
- Session working memory stores feature collection state keyed by `session_id`.
- Cross-session case memory stores synthetic analysis records keyed by `patient_id`.
- Same-session follow-up can provide only missing features; known fields are reused.
- Same-patient revisit generates a trend comparison against the previous synthetic record.

### M6: Evaluation Loop

- Adds `/eval` with three 20-case datasets.
- Medical QA set evaluates retrieval expected evidence and source traceability.
- Synthetic patient set evaluates tool-call accuracy, report field completeness, missing-feature clarification, uncertainty annotation, and revisit trend output.
- Safety set evaluates disclaimer retention, forbidden clinical claims, citation validity, and tool-boundary behavior under adversarial prompts.
- Writes JSON and Markdown reports to `eval/reports`.

Latest local M6 result:

```text
overall_pass_rate=100.0%
retrieval_traceable_source_rate=100.0%
patient_tool_sequence_accuracy=100.0%
safety_tool_boundary_rate=100.0%
```

This is an engineering evaluation of the deterministic prototype, not clinical validation.

### M7: Interactive Web Demo

- Adds `/web`, a Next.js + Tailwind single-page demo.
- Provides synthetic feature entry, missing-feature clarification, structured report display, SHAP Top feature cards, citation cards, Tool trace, and revisit trend display.
- Adds a Next API route that calls the existing Agent runner; the browser UI does not compute prediction probabilities, SHAP values, citations, or memory comparisons.
- Keeps the safety notice visible in the first viewport: 演示用合成数据，非真实患者数据；非临床诊断依据。

## Run

Install dependencies:

```powershell
npm.cmd install
python -m pip install -r ml-service/requirements.txt
```

Train M1 model:

```powershell
npm.cmd run m1:train
```

Start M1-M3 ML service:

```powershell
npm.cmd run m1:serve
```

Run M5 Agent demo in another terminal:

```powershell
npm.cmd run m5:demo
```

Run M6 evaluation:

```powershell
npm.cmd run eval
```

Start M7 web demo:

```powershell
# Terminal 1: ML service required by the prediction and SHAP tools
npm.cmd run m1:serve

# Terminal 2: Next.js frontend
npm.cmd run web:dev
```

Then open:

```text
http://127.0.0.1:3000
```

Run tests:

```powershell
npm.cmd run typecheck
npm.cmd test
python -m unittest ml-service/tests/test_m1_pipeline.py
npm.cmd run web:typecheck
npm.cmd run web:build
```

## Architecture

```text
M1-M3 ML
synthetic data generator
        |
simulated FCI causal feature selector
        |
Random Forest trainer + CV metrics
        |
model artifact
        |
local HTTP service: /predict + /explain

M4 retrieval
public-source short summaries
        |
BM25 + local hash embedding + heuristic rerank
        |
cited evidence snippets

M5 memory
session_id -> feature collection state
patient_id -> prior synthetic case records

M6 evaluation
medical QA set + patient case set + safety set
        |
latest-report.json + latest-report.md

M7 web
Next.js + Tailwind UI
        |
/api/analyze
        |
existing Agent runner

Agent
user synthetic feature input
        |
checkFeatureCompleteness
        |
getPatientHistory
        |
predictHccGrade
        |
explainPredictionWithShap
        |
retrieveMedicalEvidence
        |
saveCaseMemory
        |
structured safety-bounded report
```

## Example Interaction

```text
User: patient_id=synthetic-demo-001, submit all 10 synthetic features.
Agent Tools: checkFeatureCompleteness -> getPatientHistory -> predictHccGrade -> explainPredictionWithShap -> retrieveMedicalEvidence -> saveCaseMemory.
UI: displays probability and label from the ML Tool, SHAP Top features from the SHAP Tool, cited KB paragraphs with [KB-ID], and a report ending with the non-clinical disclaimer.

User: submit only tumor_size_cm and afp_ng_ml in the same session.
Agent Tools: checkFeatureCompleteness only.
UI: displays missing synthetic features and does not run prediction, SHAP, retrieval, or memory save.
```
