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

## M9

### DeepReason Migration Strategy

M9 starts the migration from the current Vercel AI SDK tool loop to a DeepReason-style multi-agent workflow. The migration policy is:

- keep the HCC domain capabilities that already work;
- replace the orchestration core;
- add Evidence and Gate as first-class workflow artifacts;
- migrate Memory writes through proposal and approval;
- keep the Next.js product UI;
- reuse the current evaluation datasets as regression benchmarks.

The project should not be rebuilt from scratch. The current Python ML service, Random Forest artifact, SHAP explanation service, retrieval knowledge base, memory compatibility layer, evaluation datasets, and safety statements remain the baseline.

### Legacy Runner Kept During Migration

`agent/src/runner.ts` remains temporarily as the legacy baseline runner. It should not be deleted until the DeepReason runner passes parity tests for:

- missing feature clarification;
- deterministic prediction tool call;
- deterministic SHAP explanation tool call;
- retrieval and traceable citation behavior;
- safety disclaimer retention;
- case history comparison;
- high-risk safety prompts.

### Workflow Target

The target HCC workflow is:

```text
intake -> feature_check -> history_retrieve -> prediction -> shap_explain
-> evidence_retrieve -> claim_check -> source_verify -> confidence_gate
-> report_generate -> verify -> memory_proposal -> respond
```

Unsupported claims should loop through `build_follow_up_query -> evidence_retrieve -> claim_check` with a bounded retry count. After the retry limit, the workflow must output evidence gaps instead of generating unsupported medical conclusions.

### Baseline Snapshot

The migration baseline is recorded in `docs/DEEPREASON_HCC_MIGRATION_BASELINE.md`.

Current baseline:

- TypeScript tests: 8 passed.
- TypeScript typecheck: passed.
- Evaluation report: 100.0% overall pass rate across the current 20/20/20 test sets.
- Python ML tests: 1 passed.
- ML artifact: 1400 synthetic samples, 10 selected features, 5-fold CV AUC mean 0.828.

## M10

### DeepReason Handler Adapter Layer

M10 adds the first migration code layer: `agent/src/deepreason/`.

This layer wraps existing HCC capabilities as deterministic DeepReason-style handlers. It does not yet replace the Vercel AI SDK runner. The goal is to separate domain tool execution from the old `generateText()` loop before introducing WorkflowSpec.

New handler mapping:

- `checkFeatureCompleteness` -> `feature_check` / `feature_collector`
- `getPatientHistory` -> `history_retrieve` / `memory_manager`
- `predictHccGrade` -> `prediction` / `prediction_operator`
- `explainPredictionWithShap` -> `shap_explain` / `explanation_operator`
- `retrieveMedicalEvidence` -> `evidence_retrieve` / `medical_retriever`
- `saveCaseMemory` -> `case_memory_write` / `memory_manager`

Each handler returns:

- the original parsed output;
- DeepReason node and agent metadata;
- execution trace timestamps and duration;
- evidence drafts for later EvidenceLedger migration.

### Evidence Drafts Before Full EvidenceLedger

Phase 2 intentionally returns `evidenceDrafts`, not final EvidenceLedger records. This keeps the adapter layer simple while preparing Phase 5:

- prediction output -> `model_prediction` evidence draft;
- SHAP output -> `model_explanation` evidence draft;
- retrieval hits -> `knowledge_base` evidence drafts;
- session feature state -> `session_state` evidence draft;
- case history -> `case_memory` evidence draft when history exists.

### Legacy Memory Write Kept Temporarily

`saveCaseMemory` remains a legacy-compatible direct write in Phase 2. It is named `case_memory_write` in the adapter metadata to make the future risk boundary explicit.

In Phase 6 this path must change to:

```text
memory_proposal -> memory_gate -> approved_by -> write
```

Until then, tests should treat it as a compatibility adapter, not as the final DeepReason memory design.

### Phase 2 Verification

The new adapter tests pass alongside the existing M1-M8 baseline:

- TypeScript typecheck: passed.
- TypeScript tests: 11 passed.
- Evaluation report: 100.0% overall pass rate across current test sets.
- Python ML tests: 1 passed.

## M11

### Configuration-Driven HCC WorkflowSpec

M11 introduces `configs/workflows/hcc_analysis.workflow.json` as the target orchestration contract for the DeepReason migration.

The specification separates workflow topology from handler implementation. It describes:

- node identity, assigned agent name, handler kind, and handler name;
- human-readable input and output contracts;
- checkpoints and node-level Gate policies;
- conditional edges and handoff payload contracts;
- bounded retry counters, limits, and exhausted destinations;
- reviewer requirements;
- protected and terminal nodes.

This phase does not introduce a new runtime loop. The legacy Vercel AI SDK runner remains the executable baseline while the DeepReason workflow is assembled and parity-tested.

### Workflow Safety Is Configuration Validation

The HCC workflow validator treats safety-critical topology as a configuration invariant, not as prompt guidance.

It enforces:

- incomplete features branch to clarification and forbid downstream Tools;
- prediction can only follow `history_retrieve`;
- prediction hands off to deterministic SHAP explanation;
- unsupported Claims use a bounded evidence retry;
- the evidence retry uses `evidence_retry_count` and `max_retry: 2`;
- exhausted evidence retry reaches `confidence_gate` for limited output;
- report generation can only follow `confidence_gate` or one bounded verification revision;
- memory proposal can only follow successful report verification;
- required HCC nodes remain protected and reachable.

This prevents a configuration edit from silently creating a path that bypasses deterministic prediction, evidence review, report Gate, or memory verification.

### Planned Handlers Are Not Executable Handlers

Workflow validation exposes two separate states:

```text
ok
implementationReady
```

`ok` reports whether the workflow structure and safety rules are valid. `implementationReady` additionally requires every plugin handler to exist.

At M11 the specification is structurally valid, but these handlers are deliberately marked `planned`:

- `checkClaims`;
- `buildFollowUpQuery`;
- `verifyEvidenceSources`;
- `evaluateMedicalConfidenceGate`;
- `generateHccReport`;
- `createMemoryProposal`.

Therefore the truthful M11 state is:

```text
ok: true
implementationReady: false
```

The project must not route production or demo API traffic through this WorkflowSpec until the planned handlers, AgentSpec, and runtime are implemented.

### Bounded Recovery Policies

Evidence recovery and report revision use separate counters:

```text
evidence_retry_count: max 2
report_revision_count: max 1
```

Evidence retry exhaustion does not authorize a stronger answer. It routes to `confidence_gate`, which must restrict the report to supported Claims and expose evidence gaps.

Report revision exhaustion routes to `respond` with a limited result and safety notice, preventing an infinite report-generation loop.

### Phase 3 Verification

M11 adds six WorkflowSpec tests, bringing the TypeScript test total to 17.

Verification results:

- TypeScript typecheck: passed.
- TypeScript tests: 17 passed.
- Existing evaluation report: 100.0% overall pass rate, with all 12 metrics at 20/20.
- Python ML tests: 1 passed.

Mutation tests confirm that validation rejects:

- a direct `prediction -> report_generate` bypass;
- an evidence retry with no `max_retry`;
- deletion of a required protected workflow node.

## M12

### HCC AgentsSpec And Role Boundaries

M12 introduces `configs/agents/hcc_analysis.agents.json` and `agent/src/deepreason/agentsSpec.ts`.

The HCC `AgentsSpec` follows the DeepReason reference structure:

- `agents`;
- `protected_agents`;
- `model_role`;
- `tools`;
- `permissions`;
- `memory_access`;
- `workflow_nodes`;
- `handoff_contract`;
- `ui`.

It defines 10 protected agents:

- `hcc_coordinator`;
- `feature_collector`;
- `memory_manager`;
- `prediction_operator`;
- `explanation_operator`;
- `medical_retriever`;
- `claim_checker`;
- `source_verifier`;
- `medical_safety_reviewer`;
- `report_writer`.

This phase still does not replace the executable legacy runner. It makes the future DeepReason runtime auditable by turning each role boundary into a schema and validation rule.

### Tool Authority Is Agent-Scoped

High-risk capabilities are intentionally isolated:

- `prediction_operator` can only call `predictHccGrade`;
- `explanation_operator` can only call `explainPredictionWithShap`;
- `report_writer` can only call `generateHccReport` and cannot call prediction, SHAP, or retrieval tools;
- `medical_retriever` can retrieve public rewritten knowledge snippets but cannot generate medical judgments;
- `source_verifier` can verify source integrity but cannot generate claims;
- `memory_manager` can read case history and create memory proposals, but cannot use the legacy `saveCaseMemory` direct-write adapter.

The validator rejects any `AgentsSpec` that assigns `saveCaseMemory` to a DeepReason agent. That adapter remains only for legacy parity until Phase 6 replaces it with:

```text
memory_proposal -> memory_gate -> approved_by -> write
```

### Memory Access Is Scoped

Allowed memory scopes are explicit. Examples:

- `short_term:read`;
- `session_state:write`;
- `case_memory:read`;
- `case_memory:proposal`;
- `evidence_ledger:append`;
- `gate_decisions:append`.

`case_memory:write` is deliberately invalid in the HCC `AgentsSpec`. Long-term case persistence must go through Memory Proposal and approval.

### AgentsSpec Cross-Validates WorkflowSpec

M12 adds cross-validation between `AgentsSpec` and `WorkflowSpec`:

- every workflow node must be bound to exactly one agent;
- the bound agent must match the `agent` declared on the workflow node;
- every plugin workflow node's handler must appear in the owning agent's `tools`;
- every workflow-referenced agent must exist in the Agent spec;
- every required HCC agent must be protected.

This prevents drift where the workflow says `prediction` belongs to `prediction_operator`, but the agent file accidentally gives that node or tool to `report_writer`.

### Implementation Readiness

As with M11, M12 separates structural validity from runtime readiness.

Current truthful state:

```text
ok: true
implementationReady: false
```

The spec is valid, but these planned tools still need implementation:

- `checkClaims`;
- `buildFollowUpQuery`;
- `verifyEvidenceSources`;
- `evaluateMedicalConfidenceGate`;
- `generateHccReport`;
- `createMemoryProposal`.

### Phase 4 Verification

M12 adds eight `AgentsSpec` tests, bringing the TypeScript test total to 25.

Verification results:

- TypeScript typecheck: passed.
- TypeScript tests: 25 passed.
- Existing evaluation report: 100.0% overall pass rate, with all 12 metrics at 20/20.
- Python ML tests: 1 passed.

Mutation tests confirm that validation rejects:

- `report_writer` gaining access to `predictHccGrade`;
- `memory_manager` gaining access to the legacy `saveCaseMemory` direct-write tool;
- deletion of the protected `source_verifier` agent;
- binding `prediction` to `report_writer`;
- adding unsupported `case_memory:write`;
- removing the synthetic-data-only boundary.

## M13

### Evidence Governance Handlers

M13 implements three previously planned DeepReason governance handlers:

- `checkClaims`;
- `buildFollowUpQuery`;
- `verifyEvidenceSources`.

These handlers are deterministic TypeScript functions. They do not call an LLM and do not create new medical conclusions from free text.

The migration state changes from:

```text
6 implemented handlers + 6 planned handlers
```

to:

```text
9 implemented handlers + 3 planned handlers
```

The remaining planned handlers are:

- `evaluateMedicalConfidenceGate`;
- `generateHccReport`;
- `createMemoryProposal`.

### EvidenceItem Before Report Generation

Phase 2 emitted lightweight `evidenceDrafts`. M13 converts those drafts into structured Evidence items before report generation.

An Evidence item contains:

- stable `evidenceId`;
- source type and claim type;
- source URI and locator;
- confidence;
- content hash;
- synthetic-data notice.

This means the downstream report and Gate no longer need to reason over raw tool payloads only. They can consume a normalized evidence ledger shape.

### Claim-Evidence Binding

M13 adds explicit Claim-Evidence records with:

- `claimId`;
- claim text;
- claim type;
- bound Evidence IDs;
- support status;
- confidence;
- required follow-up terms;
- notes.

Supported status is computed from deterministic rules:

- required source type must match;
- required terms must appear in Evidence content, source URI, locator, or label when terms are provided;
- confidence must meet the threshold;
- minimum evidence count must be satisfied.

If a Claim is unsupported or partially supported, the handler records a follow-up requirement instead of letting the report writer produce an unsupported conclusion.

### Bounded Evidence Recovery

`buildFollowUpQuery` reads only unsupported Claims and produces a focused retrieval query.

It respects:

```text
evidenceRetryCount
maxRetry
```

At retry exhaustion it returns `shouldRetry: false`, so the workflow can route to `confidence_gate` with evidence gaps rather than looping or inventing a conclusion.

### Source Verification Is Non-Generative

`verifyEvidenceSources` verifies Evidence integrity only. It checks:

- HTTPS URI for knowledge-base sources;
- locator presence;
- duplicate Evidence IDs;
- duplicate content hashes;
- missing Evidence references from Claims;
- source completeness;
- Claim coverage.

It does not judge clinical correctness and does not write medical explanations.

### Workflow And Agent Spec Updates

The following Workflow nodes are now marked implemented:

- `claim_check`;
- `build_follow_up_query`;
- `source_verify`.

`WorkflowSpec` and `AgentsSpec` planned lists now contain only:

- `createMemoryProposal`;
- `evaluateMedicalConfidenceGate`;
- `generateHccReport`.

Truthful state remains:

```text
ok: true
implementationReady: false
```

because Gate, Report generation, Memory Proposal, and runtime replacement are not done yet.

### Phase 5 Verification

M13 adds five Evidence tests, bringing the TypeScript test total to 30.

Verification results:

- TypeScript typecheck: passed.
- TypeScript tests: 30 passed.
- Existing evaluation report: 100.0% overall pass rate, with all 12 metrics at 20/20.
- Python ML tests: 1 passed.

New tests confirm:

- Evidence drafts become Evidence items with content hashes;
- supported and unsupported Claims are separated;
- unsupported Claims produce bounded follow-up queries;
- retry exhaustion stops follow-up retrieval;
- untraceable KB evidence and missing Evidence references are rejected;
- Evidence handlers expose DeepReason node, agent, and trace metadata.

## M14

### Deterministic Medical Confidence Gate

M14 implements `evaluateMedicalConfidenceGate`.

The Gate is deterministic. It does not ask an LLM whether a medical conclusion is acceptable. It checks structured artifacts:

- model prediction Evidence exists;
- SHAP explanation Evidence exists;
- Claim-Evidence support status;
- source verification result;
- evidence retry exhaustion;
- required disclaimer presence.

Gate statuses:

- `allow`: required tool outputs, source checks, Claims, and disclaimer are present;
- `limited`: some Claims or sources are incomplete, but at least one permitted Claim can still be reported with evidence gaps;
- `interrupt`: no supported Claims are available;
- `deny`: required prediction evidence, SHAP evidence, or disclaimer is missing.

For `deny` and `interrupt`, the Gate clears `permittedClaimIds`, so downstream report generation cannot include even technically supported Claims without authorization.

### Evidence-Only Report Generation

M14 implements `generateHccReport`.

The report writer consumes:

- `gateDecision`;
- `evidenceItems`;
- `claimEvidenceMap`;
- `sourceVerification`.

It does not call prediction, SHAP, retrieval, or an LLM. It only renders Claims listed in `gateDecision.permittedClaimIds`.

Unsupported, denied, or unpermitted Claims appear only as:

- evidence gaps;
- omitted Claim IDs;
- limited-output notes.

They are not written as report conclusions.

### Citation Boundary

The report cites Evidence IDs, not raw hidden tool state. Citation lines include:

```text
evidenceId
source.uri
source.locator
```

Model statements cite model Evidence. SHAP statements cite SHAP Evidence. Medical background statements cite verified KB Evidence.

This preserves the Claim -> Evidence -> source traceability path introduced in M13.

### Workflow And Agent Spec Updates

The following Workflow nodes are now marked implemented:

- `confidence_gate`;
- `report_generate`.

`WorkflowSpec` and `AgentsSpec` planned lists now contain only:

- `createMemoryProposal`.

Truthful state remains:

```text
ok: true
implementationReady: false
```

because controlled long-term memory proposal and runtime replacement are not implemented yet.

### Phase 6 Verification

M14 adds five Gate/Report tests, bringing the TypeScript test total to 35.

Verification results:

- TypeScript typecheck: passed.
- TypeScript tests: 35 passed.
- Existing evaluation report: 100.0% overall pass rate, with all 12 metrics at 20/20.
- Python ML tests: 1 passed.

New tests confirm:

- fully supported tool and source evidence receives `allow`;
- unsupported Claims plus retry exhaustion receive `limited`;
- missing prediction evidence or disclaimer receives `deny`;
- report generation includes only Gate-permitted Claims and Evidence IDs;
- unsupported Claims do not appear as conclusions in the report body;
- Gate and Report handlers expose DeepReason node, agent, and trace metadata.

## M15

### Controlled Memory Proposal

M15 implements `createMemoryProposal`.

The handler preserves the existing JSON memory backend but changes the DeepReason write semantics:

```text
proposal -> approval check -> controlled apply
```

Without `approvedBy`, it returns `pending_approval` and does not write to `case-memory.json`.

With `approvedBy`, it applies the proposal only if all required checks pass.

### Memory Write Preconditions

The approved apply path requires:

- `patientId`;
- Gate status `allow` or `limited`;
- valid source verification;
- at least one Evidence item;
- preserved synthetic-data notice on all Evidence items;
- model prediction Evidence;
- at least one Gate-permitted Claim;
- at least one verified Evidence ID;
- explicit `approvedBy`.

If any check fails, the handler returns `rejected` and does not call `saveCaseMemory`.

### Legacy Store Reuse

When a proposal is approved and valid, M15 deliberately reuses `saveCaseMemory()`.

This keeps:

- the existing `case-memory.json` schema;
- revisit trend comparison;
- high-grade probability delta;
- changed-feature comparison;
- previous-record lookup.

The change is not a storage rewrite. It is a permission and governance wrapper around the write path.

### Legacy Runner Boundary

The old Vercel AI SDK runner still uses the legacy `saveCaseMemory` tool for parity tests.

In the DeepReason `AgentsSpec`, `saveCaseMemory` remains forbidden. The final DeepReason path must use:

```text
createMemoryProposal -> approvedBy -> saveCaseMemory
```

### Implementation Readiness

After M15:

```text
WorkflowSpec ok: true
WorkflowSpec implementationReady: true
AgentsSpec ok: true
AgentsSpec implementationReady: true
```

All plugin handlers referenced by the HCC specs now have TypeScript implementations.

This does not mean the migration is finished. Remaining work:

- implement the DeepReason workflow runtime adapter;
- connect `web/app/api/analyze/route.ts` to that adapter;
- add DeepReason-specific runtime metrics to evaluation;
- keep legacy runner only as a parity baseline.

### Phase 7 Verification

M15 adds five Memory Proposal tests, bringing the TypeScript test total to 40.

Verification results:

- TypeScript typecheck: passed.
- TypeScript tests: 40 passed.
- Existing evaluation report: 100.0% overall pass rate, with all 12 metrics at 20/20.
- Python ML tests: 1 passed.

New tests confirm:

- pending proposals do not write memory;
- approved proposals write exactly once through the existing store;
- missing `patientId` blocks writes;
- Gate `deny` blocks writes;
- Memory Proposal handler exposes DeepReason node, agent, and trace metadata.

## M16

### DeepReason Runtime Replaces The Primary Web API Orchestrator

M16 adds `agent/src/deepreason/runtime.ts` and switches `web/app/api/analyze/route.ts` from the legacy Vercel AI SDK runner to the DeepReason-style workflow runtime.

The legacy `agent/src/runner.ts` remains in the repository as a parity baseline for old M1-M7 tests and evaluation. It is no longer the primary `/api/analyze` execution path.

The new runtime executes the implemented HCC workflow in deterministic order:

```text
intake
-> feature_check
-> history_retrieve
-> prediction
-> shap_explain
-> evidence_retrieve
-> claim_check
-> build_follow_up_query / evidence_retrieve retry when needed
-> source_verify
-> confidence_gate
-> report_generate
-> verify
-> memory_proposal
-> respond
```

### Compatibility Contract

The API preserves the existing frontend fields:

```text
analysis.completeness
analysis.history
analysis.prediction
analysis.explanation
analysis.evidence
analysis.memory
text
toolCalls
trace
```

It also adds DeepReason migration fields:

```text
workflow_trace
agent_trace
claim_evidence_map
evidence_items
source_verification
gate_decisions
retry_count
memory_proposal
verification_result
deepreason.workflowTrace
deepreason.agentTrace
deepreason.claimEvidenceMap
deepreason.gateDecision
deepreason.memoryProposal
```

The frontend now displays a lightweight DeepReason panel with Gate status, Claim-Evidence count, evidence retry count, Memory Proposal status, Gate reasons, and Workflow Trace nodes.

### Missing Feature Boundary

If any of the 10 synthetic causal candidate features are missing, the runtime stops after `feature_check` and returns a clarification request.

In that path it does not call:

- `predictHccGrade`;
- `explainPredictionWithShap`;
- `retrieveMedicalEvidence`;
- `createMemoryProposal`.

This keeps the old M5 clarification behavior while making the branch explicit in `workflow_trace`.

### Evidence Retry And Gate Behavior

Unsupported Claims trigger bounded evidence recovery through `buildFollowUpQuery` and `retrieveMedicalEvidence`.

The runtime uses `maxEvidenceRetry: 2` by default. If Claims remain unsupported after the retry limit, the Gate receives `evidenceRetryExhausted: true` and returns a limited result instead of allowing unsupported medical explanations.

The implementation performs `source_verify` before `confidence_gate` even on exhausted evidence paths. This is stricter than the minimum graph branch because the current Gate function requires a source-verification artifact.

### Memory Proposal Instead Of Direct Write

The DeepReason API path no longer calls the legacy direct `saveCaseMemory` tool. It calls `createMemoryProposal`.

Default behavior:

- no `approvedBy` -> `pending_approval`;
- no write to `case-memory.json`;
- `analysis.memory.saved` remains `false` for UI compatibility;
- revisit comparison is computed as a display snapshot when history exists.

Approved writes still reuse the existing JSON memory writer, but only through `createMemoryProposal` after Gate, Evidence, source verification, synthetic-data notice, and `patient_id` checks pass.

### Phase 8 Verification

M16 adds three runtime tests and updates the API-route test, bringing the TypeScript test total to 43.

Verification results:

```text
npm run typecheck: passed
npm run web:typecheck: passed
npm test: 43 passed
npm run eval: overall pass rate 100.0%
python -m pytest ml-service\tests -q: 1 passed
```

New tests confirm:

- complete requests execute the DeepReason workflow and keep legacy `analysis.*` outputs;
- missing synthetic features stop after `feature_check` and skip downstream tools;
- unsupported Claims trigger bounded retry and receive a `limited` Gate decision;
- `/api/analyze` returns DeepReason trace, Gate, Claim-Evidence, Evidence items, and a pending Memory Proposal;
- the DeepReason API path does not call the legacy direct `saveCaseMemory` tool.

## M17

### DeepReason-Specific Evaluation Metrics

M17 extends `eval/run-eval.ts` so the evaluation report now covers both the legacy deterministic prototype metrics and the migrated DeepReason runtime.

The legacy M6/M8 metrics remain unchanged:

- medical QA retrieval relevance and citation traceability;
- patient workflow tool sequence, report completeness, clarification, uncertainty, and revisit trend;
- high-risk safety prompts;
- retrieval ablation metrics.

The new DeepReason section evaluates the migrated runtime on the same 20 synthetic patient workflow cases, plus one forced unsupported-Claim retry stress case.

### New DeepReason Metrics

The new report includes:

```text
deepreason_tool_boundary_rate
deepreason_claim_evidence_coverage_rate
deepreason_source_verification_pass_rate
deepreason_gate_decision_accuracy
deepreason_memory_proposal_gate_rate
deepreason_report_verification_rate
deepreason_legacy_parity_task_outcome_rate
deepreason_safety_disclaimer_retention_rate
deepreason_max_retry_evidence_gap_rate
```

These are framework and safety metrics, not clinical metrics.

Metric meanings:

- `deepreason_tool_boundary_rate`: complete cases must call prediction, SHAP, retrieval, Claim check, source verification, Gate, report generation, and Memory Proposal in order; missing-feature cases must stop after feature check.
- `deepreason_claim_evidence_coverage_rate`: complete cases must produce supported Claim-Evidence bindings.
- `deepreason_source_verification_pass_rate`: complete cases must pass deterministic source verification.
- `deepreason_gate_decision_accuracy`: complete cases should receive `allow`; missing-feature cases should clarify without Gate output.
- `deepreason_memory_proposal_gate_rate`: complete cases should create `pending_approval` proposals and avoid direct `saveCaseMemory`; missing-feature cases should not create proposals.
- `deepreason_report_verification_rate`: complete cases must pass final report verification; missing-feature cases must skip report verification.
- `deepreason_legacy_parity_task_outcome_rate`: DeepReason must preserve the legacy task outcome for prediction/SHAP/retrieval availability and missing-feature clarification, while intentionally changing memory writes to proposal-only.
- `deepreason_safety_disclaimer_retention_rate`: DeepReason responses must retain the synthetic-data and non-clinical disclaimer.
- `deepreason_max_retry_evidence_gap_rate`: a forced unsupported Claim must hit the retry limit, receive a `limited` Gate decision, and expose evidence gaps.

### Diagnostics

Runtime observability values are reported as diagnostics rather than pass-rate metrics:

```text
average_workflow_node_count
average_runtime_ms
average_tool_call_count
gate_status_counts
memory_proposal_status_counts
```

They are useful for interview discussion and regression monitoring, but they are not included as binary correctness scores.

### Current Phase 9 Evaluation Result

Current local run:

```text
overall pass rate: 100.0%
deepreason_tool_boundary_rate: 100.0% (20/20)
deepreason_claim_evidence_coverage_rate: 100.0% (15/15)
deepreason_source_verification_pass_rate: 100.0% (15/15)
deepreason_gate_decision_accuracy: 100.0% (20/20)
deepreason_memory_proposal_gate_rate: 100.0% (20/20)
deepreason_report_verification_rate: 100.0% (20/20)
deepreason_legacy_parity_task_outcome_rate: 100.0% (20/20)
deepreason_safety_disclaimer_retention_rate: 100.0% (20/20)
deepreason_max_retry_evidence_gap_rate: 100.0% (1/1)
```

Current diagnostics:

```text
evaluated_cases: 20
complete_cases: 15
missing_feature_cases: 5
average_workflow_node_count: 10.5
average_runtime_ms: 21.56
average_tool_call_count: 7.75
gate_status_counts: {"allow":15,"clarification":5}
memory_proposal_status_counts: {"pending_approval":15,"none":5}
```

### Report Title Update

Because the report now covers more than the original M6 evaluator, the Markdown title changed from:

```text
M6 Evaluation Report
```

to:

```text
HCC Agent Evaluation Report
```
