# Key Design Decisions

## M1

### Synthetic-only data

All training and demo inputs are synthetic. The generator creates plausible-looking numeric feature distributions for an HCC pathology-grading demo, but these are not real patient records and must not be interpreted clinically.

Required notice: 演示用合成数据，非真实患者数据；非临床诊断依据。

### Simulated FCI feature selection

The original plan allows using `causal-learn` FCI or a rules-based simulation when FCI is not practical in the prototype. M1 uses a fixed, documented causal candidate feature list as a simulation of FCI-selected variables.

Reason:

- This keeps M1 deterministic and easy to run locally.
- It avoids claiming causal discovery validity before we have a carefully designed causal graph, background knowledge constraints, and validation protocol.
- Later milestones can replace `simulate_fci_selection` with an actual FCI implementation without changing the prediction service contract.

### Deterministic model boundary

The prediction label and probability are produced only by the Random Forest artifact. LLM/Agent layers must treat this service as the source of truth for risk prediction.

## M2

### AI SDK 7 Loop Control

The user-facing plan mentions `maxSteps`, which appears in many AI SDK examples and earlier APIs. The installed current AI SDK is 7.x, where loop control is expressed with `stopWhen: isStepCount(n)`. M2 uses that current API while preserving the intended multi-step behavior.

### Offline Deterministic Model

M2 uses `MockLanguageModelV4` from `ai/test` by default. This keeps the milestone runnable without an LLM API key while still exercising the real AI SDK `generateText` tool loop, tool schemas, tool execution callbacks, and step accumulation.

Future production/demo modes can swap the mock model for a real provider model without changing the tool contracts.

## M3

### Real SHAP Dependency

M3 installs and uses `shap.TreeExplainer` rather than a hand-rolled approximation. On this Windows/Python 3.14 environment, installing `shap` from the default PyPI source timed out, so the successful install used binary wheels from the Tsinghua PyPI mirror:

```powershell
python -m pip install shap --only-binary=:all: -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Explaining The Random Forest Step

The trained model artifact is a scikit-learn `Pipeline` with `StandardScaler` and `RandomForestClassifier`. The `/explain` endpoint transforms the raw feature row with the saved scaler and then runs `shap.TreeExplainer` on the Random Forest step.

### Causal-SHAP Consistency

The consistency check compares SHAP Top-N feature names with the fixed simulated FCI causal candidate set:

- In the intersection: `high_trust_causal_candidate`.
- Outside the candidate set: `statistical_association_only`.

## M4

### Public Traceable KB

The M4 knowledge base uses short, self-authored summaries linked to public sources such as NCI Dictionary, MedlinePlus, IBSI, and PyRadiomics docs. It does not scrape or store full copyrighted medical papers.

### Local Hybrid Retrieval

M4 implements the requested retrieval stages with local, replaceable components:

- BM25 keyword scoring implemented in TypeScript.
- Lightweight hashed token-vector cosine similarity as a local embedding substitute.
- Heuristic rerank based on matched terms, SHAP feature coverage, and source traceability.

This keeps the prototype runnable without Elasticsearch, embedding APIs, or cross-encoder downloads.

### Citation Constraint

The report formatter only renders paragraphs returned by `retrieveMedicalEvidence`, each with `[KB-ID]` and source URL. If retrieval confidence is low, it renders `现有资料不足，无法给出解释。`

## M5

### JSON Memory Instead Of SQLite

The plan allows SQLite or local JSON. M5 uses local JSON files to keep the prototype simple and inspectable:

- `memory/data/session-memory.json`
- `memory/data/case-memory.json`

Runtime JSON files are git-ignored. Tests use temporary memory directories.

### Session Working Memory

`checkFeatureCompleteness` merges the current partial feature payload into session memory before checking completeness. This lets a later turn provide only missing features while reusing previously supplied synthetic values.

### Cross-Session Case Memory

`getPatientHistory` reads previous synthetic records for a `patient_id`. `saveCaseMemory` appends the completed analysis and returns a comparison against the previous record.

The comparison includes high-grade probability delta, prediction label change, and top changed numeric features.

### Confidence-Driven Clarification

M5 keeps the hard stop for missing key features: downstream prediction, SHAP, retrieval, and case save tools are not called until all 10 required features are available.

The final report also surfaces prediction probability uncertainty when probability is in the 0.4-0.6 band and retrieval confidence warnings.

## M6

### Deterministic Evaluation Harness

M6 evaluates the local deterministic prototype without requiring a real LLM provider. The script starts an in-process mock ML HTTP service for `/predict` and `/explain`, then calls the real Agent runner, retrieval module, and memory module.

### Three Evaluation Sets

- `medical-qa.json`: 20 retrieval relevance and citation traceability cases.
- `patient-cases.json`: 20 synthetic patient workflow cases.
- `safety-cases.json`: 20 adversarial prompts.

### Metrics

The evaluation report includes:

- retrieval expected-evidence accuracy
- retrieval sufficiency accuracy
- traceable source rate
- tool sequence accuracy
- report field completeness
- missing-feature clarification accuracy
- uncertainty annotation accuracy
- revisit trend accuracy
- disclaimer retention rate
- forbidden-claim block rate
- citation validity rate
- tool-boundary rate

### Current Result

The current local run reports 100% across the designed engineering metrics. This should be described as prototype pipeline validation, not clinical validation.

## M7

### Next.js UI As A Thin Display Layer

M7 adds a Next.js + Tailwind frontend under `web/`. The page is intentionally a thin interface over the existing Agent runner:

- It collects synthetic feature values and optional natural-language instructions.
- It calls `/api/analyze`, which calls `runHccAgent`.
- It displays returned Tool outputs and the final report.
- It does not compute prediction probabilities, SHAP values, citations, or memory trends in browser code.

This keeps the original safety boundary intact: risk prediction comes from the deterministic ML Tool, explanation comes from the SHAP Tool, medical background comes from retrieval Tool results, and revisit comparison comes from memory Tool output.

### UI Safety Placement

The mandatory statement `演示用合成数据，非真实患者数据；非临床诊断依据。` appears in:

- shared code constants in `agent/src/safety.ts`
- the web UI first viewport
- the API response payload
- README and design decision documentation

### Local Demo Constraint

The web demo depends on the local Python ML service for `/predict` and `/explain`. If `npm.cmd run m1:serve` is not running, `/api/analyze` returns a friendly error instead of fabricating a fallback prediction.

### Next Build Mode

The existing TypeScript Agent code uses NodeNext-style `.js` import specifiers that are resolved to `.ts` source files by `tsc` and `tsx`. Next 16's default Turbopack build did not resolve those external workspace imports in this repository layout, so M7 uses `next dev/build --webpack` with `resolve.extensionAlias` for `.js -> .ts/.tsx/.js`.

This preserves direct imports from the web API route to the existing Agent runner instead of adding a duplicate HTTP adapter or a subprocess bridge.

## M8

### Retrieval Ablation Evaluation

M8 adds module-level ablation for the local hybrid retrieval chain. The evaluator re-runs the 20-case medical QA set under these modes:

- full hybrid retrieval
- remove BM25
- remove local hash embedding
- remove heuristic rerank
- remove query expansion
- remove diverse Top-K selection
- BM25 only
- embedding only

The ablation report includes Top-K expected-evidence accuracy, Top-1 expected-evidence accuracy, MRR, sufficiency accuracy, traceable-source rate, and percentage-point deltas versus `full_hybrid`.

### Current Finding

On the current small medical QA set, Top-K evidence accuracy remains 100% across variants, but ranking-sensitive metrics expose a tuning issue:

- `full_hybrid`: Top-1 evidence accuracy 85.0%, MRR 0.904.
- `remove_query_expansion`: Top-1 evidence accuracy 95.0% (+10.0 pp), MRR 0.967 (+6.3 pp).

This indicates that broad query expansion can dilute ranking quality in the 20-entry prototype knowledge base. The next optimization should make expansion term-aware and weight-controlled rather than appending all feature synonyms uniformly.
