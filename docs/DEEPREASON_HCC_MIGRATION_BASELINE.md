# DeepReason HCC Migration Baseline

Date: 2026-07-16
Branch: `codex/deepreason-migration`
Goal: migrate the existing HCC causal Agent prototype from a Vercel AI SDK tool loop to a DeepReason-style multi-agent workflow, while preserving the current ML, retrieval, memory, evaluation, and Next.js product surface.

Safety boundary: all patient-like records are synthetic demo data. The system is for research learning and portfolio demonstration only, and must not be used as clinical diagnosis, treatment, or pathology grading evidence.

## Phase 1 Status

Phase 1 is the migration baseline phase. No orchestration replacement has been made yet.

Completed:

- Created migration branch: `codex/deepreason-migration`.
- Inspected current source layout and execution chain.
- Confirmed current Vercel AI SDK orchestration entrypoint.
- Confirmed deterministic Python ML service boundaries.
- Confirmed current retrieval, memory, web API, and evaluation files.
- Ran current regression tests and evaluation.
- Recorded this migration mapping before changing the core Agent.

## Baseline Verification

Commands run:

```powershell
npm.cmd test
npm.cmd run typecheck
npm.cmd run eval
python -m pytest ml-service\tests -q
```

Results:

```text
npm test: 8 passed, 0 failed
npm run typecheck: passed
npm run eval: overall pass rate 100.0%
python ml-service tests: 1 passed
```

Evaluation dataset sizes:

```text
medicalQa: 20
patientCases: 20
safetyCases: 20
```

Current evaluation metrics:

```text
retrieval_expected_id_or_low_confidence_accuracy: 100.0% (20/20)
retrieval_sufficiency_accuracy: 100.0% (20/20)
retrieval_traceable_source_rate: 100.0% (20/20)
patient_tool_sequence_accuracy: 100.0% (20/20)
patient_report_field_completeness: 100.0% (20/20)
patient_missing_feature_clarification_accuracy: 100.0% (20/20)
patient_uncertainty_annotation_accuracy: 100.0% (20/20)
patient_revisit_trend_accuracy: 100.0% (20/20)
safety_disclaimer_retention_rate: 100.0% (20/20)
safety_forbidden_claim_block_rate: 100.0% (20/20)
safety_valid_citation_rate: 100.0% (20/20)
safety_tool_boundary_rate: 100.0% (20/20)
```

Retrieval ablation baseline:

```text
full_hybrid Top-K: 100.0%
full_hybrid Top-1: 85.0%
full_hybrid MRR: 0.904
full_hybrid Sufficiency: 100.0%
full_hybrid Traceable Source: 100.0%
```

ML baseline:

```text
synthetic samples: 1400
selected features: 10
feature selection: simulated_fci_fixed_causal_candidates
model: RandomForestClassifier
5-fold cv_auc_mean: 0.8283638874928736
5-fold cv_auc_std: 0.01667421275941323
positive_rate: 0.54
```

## Current Architecture

Current request path:

```text
Next.js /api/analyze
  -> agent/src/runner.ts runHccAgent()
  -> Vercel AI SDK generateText()
  -> AI SDK tools
  -> memory / retrieval / Python ML service
  -> final text report
```

Current fixed tool loop expected by tests:

```text
checkFeatureCompleteness
  -> getPatientHistory
  -> predictHccGrade
  -> explainPredictionWithShap
  -> retrieveMedicalEvidence
  -> saveCaseMemory
  -> final report
```

Current core files:

```text
agent/src/runner.ts          Vercel AI SDK orchestration
agent/src/tools.ts           AI SDK tool definitions and schemas
agent/src/features.ts        10 causal candidate feature schema
agent/src/predictionTypes.ts Tool input/output schemas
agent/src/report.ts          Structured report formatter
agent/src/safety.ts          Synthetic-data and non-clinical safety notices
memory/src/store.ts          Session and case JSON memory
retrieval/src/search.ts      BM25 + local hash embedding + heuristic rerank
retrieval/knowledge-base.json Public traceable knowledge snippets
ml-service/src/hcc_synthetic Python RF prediction and SHAP explanation
web/app/api/analyze/route.ts Next.js API entrypoint
web/app/page.tsx             Demo UI
eval/run-eval.ts             Regression and ablation evaluation
```

## Target DeepReason Architecture

Target request path:

```text
Next.js /api/analyze
  -> DeepReason HCC Adapter
  -> HCC WorkflowSpec
  -> HCC AgentsSpec
  -> deterministic handlers/tools
  -> EvidenceLedger / Claim-Evidence map
  -> Gate decisions
  -> Memory proposal/write gate
  -> structured report
```

Target workflow:

```text
intake
  -> feature_check
  -> history_retrieve
  -> prediction
  -> shap_explain
  -> evidence_retrieve
  -> claim_check
  -> source_verify
  -> confidence_gate
  -> report_generate
  -> verify
  -> memory_proposal
  -> respond
```

Retry path:

```text
claim_check unsupported_claims exist
  -> build_follow_up_query
  -> evidence_retrieve
  -> claim_check
```

Default retry limit:

```text
max_retry: 2
```

After the retry limit, the workflow must publish evidence gaps and uncertainty notes instead of generating unsupported medical conclusions.

## Existing File To DeepReason Target Mapping

| Existing file or module | Target migration role |
| --- | --- |
| `ml-service/` | Keep as deterministic external prediction and SHAP service. Wrap with DeepReason handler/tool adapters. |
| `ml-service/artifacts/hcc_rf_pipeline.*` | Keep as model artifact and metric evidence source. |
| `agent/src/features.ts` | Keep schema; reuse in HCC intake and feature_check handlers. |
| `agent/src/predictionTypes.ts` | Keep as schema source; extend with EvidenceItem and Claim schemas. |
| `agent/src/tools.ts` | Split into DeepReason handlers/adapters. AI SDK tool wrappers become deprecated after migration. |
| `agent/src/runner.ts` | Replace as core orchestrator. Keep temporarily as legacy baseline runner until parity is proven. |
| `agent/src/report.ts` | Keep report formatting logic, then refactor to consume Claim-Evidence map and Gate output. |
| `retrieval/src/search.ts` | Keep retrieval implementation; wrap results into EvidenceItems. |
| `retrieval/knowledge-base.json` | Keep as local knowledge base; add evidence source metadata mapping. |
| `memory/src/store.ts` | Keep compatibility reader/writer; route writes through Memory Proposal and Gate. |
| `web/app/api/analyze/route.ts` | Keep API route; replace internal call from `runHccAgent()` to DeepReason HCC adapter. |
| `web/app/page.tsx` | Keep product demo UI; extend payload display for workflow trace, gate decisions, and claim evidence. |
| `eval/run-eval.ts` | Keep regression benchmark; add DeepReason-specific metrics. |
| `docs/DECISIONS.md` | Keep; append migration decisions as stages land. |

## Existing Tool To New Handler Mapping

| Current AI SDK tool | Target DeepReason node | Target agent | Evidence output |
| --- | --- | --- | --- |
| `checkFeatureCompleteness` | `feature_check` | `feature_collector` | optional state evidence for collected synthetic features |
| `getPatientHistory` | `history_retrieve` | `memory_manager` | `ev-history-*` if history exists |
| `predictHccGrade` | `prediction` | `prediction_operator` | `ev-prediction-*` with label, probabilities, model metadata |
| `explainPredictionWithShap` | `shap_explain` | `explanation_operator` | `ev-shap-*` with top features and causal-SHAP trust labels |
| `retrieveMedicalEvidence` | `evidence_retrieve` | `medical_retriever` | `ev-kb-*` for each returned KB paragraph |
| report formatter | `report_generate` | `report_writer` | report claims must reference evidence IDs |
| current citation validation in eval | `source_verify` | `source_verifier` | source verification notes and conflicts |
| current safety checks in eval/prompt | `confidence_gate` / `verify` | `medical_safety_reviewer` | gate decisions and violation reasons |
| `saveCaseMemory` | `memory_proposal` | `memory_manager` | memory proposal first; persisted write only after approval |

## Target HCC Agents

| Agent | Responsibility boundary |
| --- | --- |
| `hcc_coordinator` | Advances workflow state and chooses deterministic branch outcomes. |
| `feature_collector` | Merges session features and asks only for missing synthetic feature fields. |
| `prediction_operator` | Calls RF prediction service only; cannot write report or invent probability. |
| `explanation_operator` | Calls SHAP service only; cannot generate medical conclusions. |
| `medical_retriever` | Retrieves public traceable knowledge snippets. |
| `claim_checker` | Binds report claims to evidence and marks support status. |
| `source_verifier` | Checks URI, locator, duplicate source, low confidence, and conflicts. |
| `medical_safety_reviewer` | Enforces synthetic-data, non-clinical, no-diagnosis, no-forged-citation gates. |
| `report_writer` | Writes report only from tool outputs, claim map, evidence, and gate status. |
| `memory_manager` | Generates memory proposals and performs approved writes only. |

## New DeepReason Payload Fields

The migrated `/api/analyze` response should preserve current fields and add:

```text
workflow_trace
agent_trace
claim_evidence_map
gate_decisions
retry_count
memory_proposal
verification_result
evidence_items
source_verification
```

## Regression Test Mapping

| Existing test | Migration requirement |
| --- | --- |
| `agent/tests/runner.test.ts` | Keep as legacy baseline; add DeepReason runner parity tests. |
| `retrieval/tests/search.test.ts` | Reuse unchanged; add EvidenceItem conversion tests. |
| `web/tests/api-route.test.ts` | Update to assert DeepReason payload fields while preserving current analysis fields. |
| `ml-service/tests/test_m1_pipeline.py` | Reuse unchanged; prediction and SHAP service must remain deterministic. |
| `eval/run-eval.ts` | Reuse datasets; add Claim coverage, gate, retry, and memory proposal metrics. |

## DeepReason-Specific Metrics To Add

```text
claim_evidence_coverage_rate
claim_unsupported_block_rate
source_verification_pass_rate
gate_decision_accuracy
memory_proposal_gate_rate
retry_success_rate
max_retry_evidence_gap_rate
average_workflow_node_count
average_runtime_ms
legacy_parity_report_completeness
```

## Keep / Modify / Deprecate List

Keep:

- `ml-service/` Python service and artifacts.
- Synthetic data generation and RF model.
- SHAP explanation service.
- `retrieval/knowledge-base.json`.
- Current Next.js product UI.
- Current evaluation datasets.
- Safety statements and disclaimers.
- Existing README and decision records.

Modify:

- `web/app/api/analyze/route.ts` to call DeepReason HCC adapter.
- `agent/src/report.ts` to consume Claim-Evidence map and Gate decisions.
- `memory/src/store.ts` write path to require Memory Proposal and approval gate.
- `eval/run-eval.ts` to compare legacy vs DeepReason runner and add DeepReason metrics.
- README and docs to describe the migration truthfully.

Deprecate after parity:

- Vercel AI SDK `generateText()` as the core orchestrator.
- `agent/src/runner.ts` as the primary execution path.
- Direct long-term case memory writes without proposal and gate.
- Report claims that cite only raw KB IDs without Claim-Evidence support status.

Do not delete yet:

- Legacy AI SDK runner and tools. Keep until DeepReason parity tests pass.

## Stage Gate Before Phase 2

Proceed to Phase 2 only if the user confirms:

- The baseline test results are acceptable.
- The target workflow node list is acceptable.
- The target agent boundary list is acceptable.
- Legacy runner should remain temporarily for parity comparison.

## Phase 2 Adapter Layer Update

Date: 2026-07-17

Phase 2 added a DeepReason-style deterministic handler adapter layer without replacing the current Vercel AI SDK runner.

New files:

```text
agent/src/deepreason/types.ts
agent/src/deepreason/toolAdapters.ts
agent/src/deepreason/index.ts
agent/tests/deepreason-adapters.test.ts
```

The adapter layer wraps the existing HCC capabilities as deterministic handler functions:

```text
checkFeatureCompleteness -> feature_check -> feature_collector
getPatientHistory -> history_retrieve -> memory_manager
predictHccGrade -> prediction -> prediction_operator
explainPredictionWithShap -> shap_explain -> explanation_operator
retrieveMedicalEvidence -> evidence_retrieve -> medical_retriever
saveCaseMemory -> case_memory_write -> memory_manager
```

Each handler result now has DeepReason-oriented metadata:

```text
handlerName
nodeId
agentName
output
evidenceDrafts
trace.startedAt
trace.completedAt
trace.durationMs
```

Important boundary:

- The adapters do not call the Vercel AI SDK.
- The adapters do not replace `agent/src/runner.ts` yet.
- The adapters do not change RF prediction, SHAP explanation, retrieval ranking, or memory schema.
- `saveCaseMemory` is still a legacy-compatible write handler. Phase 6 will route it through Memory Proposal and Gate before long-term persistence.

Phase 2 verification:

```text
npm run typecheck: passed
npm test: 11 passed
npm run eval: overall pass rate 100.0%
python -m pytest ml-service\tests -q: 1 passed
```

New adapter tests cover:

- handler metadata for node, agent, permission, and deterministic execution;
- complete HCC adapter flow across feature, history, prediction, SHAP, retrieval, and memory;
- Evidence draft creation for session state, model prediction, SHAP explanation, KB retrieval, and case history;
- missing-feature boundary preservation.

Stage gate before Phase 3:

- Confirm the adapter metadata shape is acceptable.
- Confirm `case_memory_write` should remain legacy-compatible until Memory Proposal migration.
- Confirm the next step should create `configs/workflows/hcc_analysis.workflow.json` and workflow validation tests.

## Phase 3 WorkflowSpec Update

Date: 2026-07-17

Phase 3 added a configuration-driven HCC workflow definition and a strict validator. It defines orchestration topology and safety boundaries but does not execute the workflow yet.

New files:

```text
configs/workflows/hcc_analysis.workflow.json
agent/src/deepreason/workflowSpec.ts
agent/tests/deepreason-workflow-spec.test.ts
```

Updated export:

```text
agent/src/deepreason/index.ts
```

The workflow contains 14 nodes and 19 directed edges:

```text
intake
-> feature_check
-> history_retrieve
-> prediction
-> shap_explain
-> evidence_retrieve
-> claim_check
-> source_verify
-> confidence_gate
-> report_generate
-> verify
-> memory_proposal
-> respond
```

The evidence-gap branch is:

```text
claim_check
-> build_follow_up_query
-> evidence_retrieve
-> claim_check
```

The branch uses `evidence_retry_count` with `max_retry: 2`. When retries are exhausted, the workflow goes to `confidence_gate` with a forced limited-output policy rather than allowing an unsupported medical conclusion.

The report revision branch uses `report_revision_count` with `max_retry: 1`. A second verification failure goes to `respond` with a limited result and safety notice.

### Enforced Workflow Boundaries

The validator now rejects a specification when:

- a required HCC node is missing or is removed from `protected_nodes`;
- an edge references a missing node;
- a node is unreachable from `intake`;
- a retry or revision edge has no positive retry limit or exceeds the safety limit of 3;
- incomplete features can continue to downstream tools;
- prediction is entered from anywhere except `history_retrieve`;
- report generation is entered from anywhere except `confidence_gate` or the bounded `verify` revision path;
- a memory proposal is entered from anywhere except `verify`;
- the Claim evidence retry does not use `evidence_retry_count`, `max_retry: 2`, and `confidence_gate` as its exhausted destination;
- a builtin or plugin handler is unknown.

### Implementation Readiness

The current WorkflowSpec validation result is intentionally:

```text
ok: true
implementationReady: false
```

`ok: true` means the graph, contracts, bounded loops, protected nodes, and HCC safety topology are structurally valid.

`implementationReady: false` is expected because the following six plugin handlers are planned for later phases:

```text
buildFollowUpQuery
checkClaims
createMemoryProposal
evaluateMedicalConfidenceGate
generateHccReport
verifyEvidenceSources
```

The WorkflowSpec must not be described as executable until these handlers and the Phase 4 AgentSpec bindings exist.

### Phase 3 Verification

```text
npm run typecheck: passed
npm test: 17 passed
npm run eval: overall pass rate 100.0%
python -m pytest ml-service\tests -q: 1 passed
```

The six new WorkflowSpec tests cover:

- schema loading, handler classification, checkpoints, and full graph reachability;
- missing-feature clarification without downstream Tool execution;
- deterministic prediction entry only through feature and history processing;
- evidence retry count, retry limit, and exhausted path;
- report generation and memory proposal gate boundaries;
- rejection of a direct `prediction -> report_generate` bypass;
- rejection of an unbounded evidence retry;
- rejection of deleting a required protected node.

The evaluation still reports all 12 existing metrics at `20/20`. The ML service and retrieval algorithms were not changed in Phase 3.

Stage gate before Phase 4:

- Confirm the 14-node topology and bounded retry policies are acceptable.
- Confirm `ok: true` with `implementationReady: false` accurately describes the current migration state.
- Confirm the next step should create the HCC `AgentsSpec`, role permissions, memory scopes, and node-to-agent bindings.

## Phase 4 AgentsSpec Update

Date: 2026-07-17

Phase 4 added a DeepReason-style HCC `AgentsSpec` and cross-validation against the Phase 3 `WorkflowSpec`. It defines role boundaries, tool permissions, memory scopes, node ownership, and handoff contracts. It still does not replace the runtime loop.

New files:

```text
configs/agents/hcc_analysis.agents.json
agent/src/deepreason/agentsSpec.ts
agent/tests/deepreason-agents-spec.test.ts
```

Updated export:

```text
agent/src/deepreason/index.ts
```

The spec defines 10 protected domain agents:

| Agent | Workflow nodes | Tools | Memory access |
| --- | --- | --- | --- |
| `hcc_coordinator` | `intake`, `respond` | `workflow_status` | `workflow_state:read` |
| `feature_collector` | `feature_check` | `checkFeatureCompleteness` | short-term and session feature state |
| `memory_manager` | `history_retrieve`, `memory_proposal` | `getPatientHistory`, `createMemoryProposal` | `case_memory:read`, `case_memory:proposal`, `evidence_ledger:read` |
| `prediction_operator` | `prediction` | `predictHccGrade` | none |
| `explanation_operator` | `shap_explain` | `explainPredictionWithShap` | none |
| `medical_retriever` | `evidence_retrieve`, `build_follow_up_query` | `retrieveMedicalEvidence`, `buildFollowUpQuery` | `knowledge_base:read`, `evidence_ledger:append` |
| `claim_checker` | `claim_check` | `checkClaims` | evidence ledger read/append |
| `source_verifier` | `source_verify` | `verifyEvidenceSources` | `evidence_ledger:read` |
| `medical_safety_reviewer` | `confidence_gate`, `verify` | `evaluateMedicalConfidenceGate` | evidence ledger read and gate decision append |
| `report_writer` | `report_generate` | `generateHccReport` | evidence, case history, and gate decision read |

### Enforced Agent Boundaries

The validator now rejects an `AgentsSpec` when:

- a required HCC agent is missing or removed from `protected_agents`;
- an unknown tool or unsupported memory scope is assigned;
- any agent is not marked `synthetic_data_only`;
- `saveCaseMemory` is assigned to the final DeepReason agent spec instead of `createMemoryProposal`;
- `case_memory:write` appears in `memory_access`;
- `prediction_operator` has anything other than `predictHccGrade` and `prediction`;
- `report_writer` can call prediction, SHAP, or retrieval tools;
- `memory_manager` is missing proposal-only write policy;
- a workflow node is unbound, multiply bound, or bound to an agent different from the `WorkflowSpec` owner;
- a plugin workflow node's handler is missing from the owning agent's tool list.

This turns the role-boundary claims into testable configuration rules. For example, the report writer cannot silently gain access to the deterministic prediction tool, and the memory manager cannot bypass the future Memory Proposal gate with the legacy direct-write adapter.

### Implementation Readiness

The current `AgentsSpec` validation result is intentionally:

```text
ok: true
implementationReady: false
```

`ok: true` means the Agent roles, permissions, memory scopes, and workflow-node bindings are structurally valid.

`implementationReady: false` remains expected because these six tools are still planned:

```text
buildFollowUpQuery
checkClaims
createMemoryProposal
evaluateMedicalConfidenceGate
generateHccReport
verifyEvidenceSources
```

The DeepReason runtime must not become the primary API path until those tools and the execution engine are implemented and regression-tested.

### Phase 4 Verification

```text
npm run typecheck: passed
npm test: 25 passed
npm run eval: overall pass rate 100.0%
python -m pytest ml-service\tests -q: 1 passed
```

The eight new `AgentsSpec` tests cover:

- loading and validating the 10-agent spec;
- every WorkflowSpec node being bound to exactly one declared owner;
- high-risk role boundaries for prediction, report generation, and memory;
- rejection of report-writer access to `predictHccGrade`;
- rejection of legacy direct case-memory writes through `saveCaseMemory`;
- rejection of deleting a protected domain agent;
- rejection of node ownership mismatch with the WorkflowSpec;
- rejection of unsupported direct memory write scope and missing synthetic-data boundary.

The evaluation still reports all 12 existing metrics at `20/20`. No ML, SHAP, retrieval, memory storage, or Next.js API behavior was changed in Phase 4.

Stage gate before Phase 5:

- Confirm the 10-agent role split and tool boundaries are acceptable.
- Confirm `saveCaseMemory` should remain available only as a legacy adapter, not as an allowed DeepReason agent tool.
- Confirm the next step should implement the Evidence model: prediction/SHAP/KB drafts -> Evidence items, Claim-Evidence map, and evidence-gap follow-up planning.

## Phase 5 Evidence And Claim-Evidence Update

Date: 2026-07-17

Phase 5 implements the first executable DeepReason governance layer above the deterministic HCC tools. It upgrades Phase 2 `evidenceDrafts` into structured Evidence items, binds Claims to Evidence, builds bounded follow-up retrieval queries for unsupported Claims, and verifies source traceability.

New file:

```text
agent/src/deepreason/evidence.ts
agent/tests/deepreason-evidence.test.ts
```

Updated files:

```text
agent/src/deepreason/types.ts
agent/src/deepreason/toolAdapters.ts
agent/src/deepreason/index.ts
agent/src/deepreason/workflowSpec.ts
agent/src/deepreason/agentsSpec.ts
configs/workflows/hcc_analysis.workflow.json
configs/agents/hcc_analysis.agents.json
```

Three previously planned handlers are now implemented:

| Handler | Workflow node | Agent | Purpose |
| --- | --- | --- | --- |
| `checkClaims` | `claim_check` | `claim_checker` | Convert evidence drafts into Evidence items and bind Claims to supporting evidence. |
| `buildFollowUpQuery` | `build_follow_up_query` | `medical_retriever` | Build a focused query from unsupported Claims while respecting the retry counter. |
| `verifyEvidenceSources` | `source_verify` | `source_verifier` | Check source URI, locator, duplicate evidence IDs/content hashes, and missing claim references. |

Handler registry size is now:

```text
6 legacy-compatible tool adapters
+ 3 Evidence governance handlers
= 9 DeepReason handler specs
```

### EvidenceItem Shape

Each normalized Evidence item records:

```text
evidenceId
sourceType
claimType
content
confidence
source.uri
source.locator
source.label
contentHash
syntheticDataNotice
```

Sources currently supported:

- `model_prediction`;
- `model_explanation`;
- `knowledge_base`;
- `case_memory`;
- `session_state`.

The `contentHash` is a SHA-256 hash over the evidence identity, source, claim type, content, URI, and locator. It is used by the source verifier to detect duplicate evidence content.

### Claim-Evidence Shape

Each Claim binding records:

```text
claimId
claim
claimType
evidenceIds
supportStatus
confidence
requiredFollowUp
notes
```

`supportStatus` can be:

- `supported`;
- `partially_supported`;
- `unsupported`.

The default generated Claims check that:

- RF prediction output exists from the deterministic prediction tool;
- SHAP explanation output exists from the deterministic SHAP tool;
- traceable public medical background evidence exists for report explanation;
- synthetic case history exists when case-memory evidence is present.

Custom required Claims can additionally specify required source types, terms, minimum evidence count, and confidence threshold.

### Follow-Up Query Behavior

`buildFollowUpQuery` only reads unsupported Claims. It does not modify prediction, SHAP, or existing Evidence.

It returns:

```text
shouldRetry
followUpQuery
unsupportedClaimIds
evidenceRetryCount
maxRetry
reason
safetyNotice
```

When `evidenceRetryCount >= maxRetry`, it returns `shouldRetry: false`, empty `followUpQuery`, and a reason instructing the workflow to route to `confidence_gate` with evidence gaps.

### Source Verification Behavior

`verifyEvidenceSources` checks:

- knowledge-base evidence must have an `https://` URI;
- knowledge-base evidence must have a locator;
- Claim references must point to existing Evidence IDs;
- duplicate Evidence IDs are invalid;
- duplicate content hashes are flagged;
- source completeness rate and Claim coverage rate are reported.

The verifier does not generate medical judgments. It only validates source integrity and evidence references.

### Implementation Readiness

After Phase 5:

```text
WorkflowSpec ok: true
AgentsSpec ok: true
implementationReady: false
```

`implementationReady` remains false because three handlers are still planned:

```text
createMemoryProposal
evaluateMedicalConfidenceGate
generateHccReport
```

DeepReason runtime should still not replace the API path until Gate, Report, Memory Proposal, and the runtime adapter are implemented and regression-tested.

### Phase 5 Verification

```text
npm run typecheck: passed
npm test: 30 passed
npm run eval: overall pass rate 100.0%
python -m pytest ml-service\tests -q: 1 passed
```

The five new Evidence tests cover:

- converting drafts into Evidence items and binding supported/unsupported Claims;
- generating bounded follow-up retrieval queries;
- stopping follow-up query generation after retry exhaustion;
- verifying traceable source URI, locator, and Claim references;
- rejecting untraceable KB sources and missing Evidence references;
- exposing DeepReason node, agent, and trace metadata for the three Evidence handlers.

The evaluation still reports all 12 existing metrics at `20/20`. No RF model, SHAP service, retrieval scoring, memory storage, or Next.js API behavior was changed in Phase 5.

Stage gate before Phase 6:

- Confirm the EvidenceItem and Claim-Evidence shapes are acceptable.
- Confirm source verification should remain deterministic and non-generative.
- Confirm the next step should implement `evaluateMedicalConfidenceGate` and `generateHccReport` so reports consume Claim-Evidence and source verification instead of raw KB citations.

## Phase 6 Gate And Evidence-Only Report Update

Date: 2026-07-17

Phase 6 implements the medical confidence Gate and structured report generator. Reports now consume `Claim-Evidence` bindings, normalized Evidence items, and source verification results. They no longer need to read raw KB hits directly.

Updated files:

```text
agent/src/deepreason/evidence.ts
agent/src/deepreason/types.ts
agent/src/deepreason/index.ts
agent/src/deepreason/workflowSpec.ts
agent/src/deepreason/agentsSpec.ts
configs/workflows/hcc_analysis.workflow.json
configs/agents/hcc_analysis.agents.json
```

New test file:

```text
agent/tests/deepreason-gate-report.test.ts
```

Two previously planned handlers are now implemented:

| Handler | Workflow node | Agent | Purpose |
| --- | --- | --- | --- |
| `evaluateMedicalConfidenceGate` | `confidence_gate` | `medical_safety_reviewer` | Apply deterministic checks over prediction evidence, SHAP evidence, Claim support, source verification, retry exhaustion, and disclaimer presence. |
| `generateHccReport` | `report_generate` | `report_writer` | Generate a structured synthetic HCC report using only Gate-permitted Claims and their Evidence IDs. |

Handler registry size is now:

```text
6 legacy-compatible tool adapters
+ 5 Evidence/Gate/Report governance handlers
= 11 DeepReason handler specs
```

### Gate Decision Shape

The Gate returns:

```text
gateId
status
riskLevel
permittedClaimIds
limitedClaimIds
deniedClaimIds
reasons
evidenceGaps
requiredActions
requiresDisclaimer
safetyNotice
```

`status` can be:

- `allow`;
- `limited`;
- `interrupt`;
- `deny`.

Current deterministic rules:

- missing deterministic RF prediction evidence -> `deny`;
- missing deterministic SHAP evidence -> `deny`;
- missing disclaimer -> `deny`;
- no supported Claim at all -> `interrupt`;
- unsupported or partially supported Claims -> `limited`;
- invalid source verification -> `limited`;
- exhausted evidence retry -> `limited`;
- all tool outputs, Claims, sources, and disclaimers valid -> `allow`.

When the status is `deny` or `interrupt`, `permittedClaimIds` is empty, so the report writer cannot accidentally include a supported but unauthorized Claim.

### Evidence-Only Report Behavior

`generateHccReport` receives:

```text
gateDecision
evidenceItems
claimEvidenceMap
sourceVerification
patientId
```

It emits:

```text
reportId
status
markdown
sections
citedEvidenceIds
omittedClaimIds
generationRules
safetyNotice
disclaimer
```

The report writer only includes Claims whose IDs appear in `gateDecision.permittedClaimIds`. Unsupported or denied Claims are not written as conclusions. They appear only as evidence gaps or omitted Claim IDs.

Report sections:

- Tool and Gate summary;
- permitted Claims;
- deterministic model evidence;
- SHAP evidence;
- medical background evidence;
- evidence gaps and limited-output notes;
- citation index;
- mandatory disclaimer.

### Implementation Readiness

After Phase 6:

```text
WorkflowSpec ok: true
AgentsSpec ok: true
implementationReady: false
```

Only one plugin handler remains planned:

```text
createMemoryProposal
```

The DeepReason runtime still should not replace the API path until Memory Proposal and the runtime adapter are implemented and regression-tested.

### Phase 6 Verification

```text
npm run typecheck: passed
npm test: 35 passed
npm run eval: overall pass rate 100.0%
python -m pytest ml-service\tests -q: 1 passed
```

The five new Gate/Report tests cover:

- `allow` when RF prediction evidence, SHAP evidence, public KB evidence, source verification, and disclaimer are present;
- `limited` when a Claim remains unsupported and evidence retry is exhausted;
- `deny` when prediction evidence or disclaimer is missing;
- report generation includes only Gate-permitted Claims and verified Evidence IDs;
- Gate and Report wrappers expose DeepReason node, agent, and trace metadata.

The evaluation still reports all 12 existing metrics at `20/20`. No RF model, SHAP service, retrieval scoring, memory storage, or Next.js API behavior was changed in Phase 6.

Stage gate before Phase 7:

- Confirm Gate behavior should remain deterministic and rule-based for this prototype.
- Confirm `generateHccReport` should continue to include unsupported Claims only as evidence gaps, never as conclusions.
- Confirm the next step should implement `createMemoryProposal`, Memory Gate compatibility, and proposal-only long-term case writes.

## Phase 7 Memory Proposal Update

Date: 2026-07-17

Phase 7 implements the controlled long-term memory path for synthetic case records. It keeps the existing JSON store and revisit comparison logic, but moves the DeepReason write path behind a proposal and approval check.

Updated files:

```text
agent/src/deepreason/evidence.ts
agent/src/deepreason/types.ts
agent/src/deepreason/index.ts
agent/src/deepreason/workflowSpec.ts
agent/src/deepreason/agentsSpec.ts
configs/workflows/hcc_analysis.workflow.json
configs/agents/hcc_analysis.agents.json
```

New test file:

```text
agent/tests/deepreason-memory-proposal.test.ts
```

The final planned handler is now implemented:

| Handler | Workflow node | Agent | Purpose |
| --- | --- | --- | --- |
| `createMemoryProposal` | `memory_proposal` | `memory_manager` | Create an auditable synthetic case-memory proposal and apply it only when approved and all Gate/Evidence checks pass. |

Handler registry size is now:

```text
6 legacy-compatible tool adapters
+ 6 Evidence/Gate/Report/Memory governance handlers
= 12 DeepReason handler specs
```

### Memory Proposal Shape

`createMemoryProposal` receives:

```text
patientId
sessionId
features
prediction
shap
retrieval
gateDecision
evidenceItems
sourceVerification
approvedBy
memoryDir
```

It emits:

```text
proposalId
status
targetPartition
patientId
sessionId
requiresApproval
approvedBy
canApply
applied
directWrite
recordCountBefore
recordCountAfter
evidenceIds
gateStatus
reasons
blockedReasons
appliedResult
safetyNotice
```

`status` can be:

- `pending_approval`;
- `approved_applied`;
- `rejected`.

### Controlled Write Rules

Default behavior is proposal-only:

- no `approvedBy` -> `pending_approval`;
- no case-memory write occurs;
- `recordCountAfter` equals `recordCountBefore`;
- `directWrite` is always `false`.

Approved writes require all of the following:

- `patientId` exists;
- Gate status is `allow` or `limited`;
- source verification is valid;
- at least one Evidence item exists;
- all Evidence items preserve the synthetic-data notice;
- model prediction Evidence exists;
- at least one Gate-permitted Claim exists;
- at least one verified Evidence ID exists;
- `approvedBy` is present.

If any check fails, the proposal returns `rejected` and does not call the underlying JSON writer.

When approved and valid, `createMemoryProposal` calls the existing `saveCaseMemory()` function, so it preserves:

- the current case-memory JSON schema;
- previous-record lookup;
- probability delta calculation;
- changed-feature comparison;
- revisit trend summary.

### Implementation Readiness

After Phase 7:

```text
WorkflowSpec ok: true
WorkflowSpec implementationReady: true
AgentsSpec ok: true
AgentsSpec implementationReady: true
```

This means every plugin handler referenced by the HCC `WorkflowSpec` and `AgentsSpec` now has a TypeScript implementation.

Important remaining boundary:

- the DeepReason runtime has not yet replaced the legacy Vercel AI SDK runner;
- `web/app/api/analyze/route.ts` still calls the legacy `runHccAgent()` path;
- the legacy `saveCaseMemory` adapter remains only for baseline parity and M1-M7 regression tests.

### Phase 7 Verification

```text
npm run typecheck: passed
npm test: 40 passed
npm run eval: overall pass rate 100.0%
python -m pytest ml-service\tests -q: 1 passed
```

The five new Memory Proposal tests cover:

- no approval -> pending proposal and no case-memory write;
- approval -> exactly one controlled case-memory write through the existing store;
- missing `patientId` -> rejected, no write;
- Gate `deny` -> rejected, no write;
- wrapper metadata for `memory_proposal` / `memory_manager`.

The evaluation still reports all 12 existing metrics at `20/20`. No RF model, SHAP service, retrieval scoring, legacy runner output, or Next.js API behavior was changed in Phase 7.

Stage gate before Phase 8:

- Confirm proposal-only memory behavior is acceptable.
- Confirm approved writes should keep using the existing JSON memory store for now.
- Confirm the next step should implement the DeepReason runtime adapter and then switch the Next.js API from `runHccAgent()` to the new workflow runtime.

## Phase 8 Runtime And Next.js API Update

Date: 2026-07-17

Phase 8 implements the executable DeepReason HCC runtime adapter and switches the product API from the legacy Vercel AI SDK runner to the new workflow runtime.

New file:

```text
agent/src/deepreason/runtime.ts
agent/tests/deepreason-runtime.test.ts
```

Updated files:

```text
agent/src/deepreason/index.ts
web/app/api/analyze/route.ts
web/app/page.tsx
web/tests/api-route.test.ts
docs/DECISIONS.md
docs/DEEPREASON_HCC_MIGRATION_BASELINE.md
```

### New Runtime Path

The migrated request path is now:

```text
Next.js /api/analyze
  -> runHccDeepReasonWorkflow()
  -> DeepReason-style workflow trace
  -> deterministic HCC handlers
  -> Claim-Evidence map
  -> source verification
  -> confidence Gate
  -> evidence-only report
  -> Memory Proposal
  -> response
```

The legacy path remains available for baseline comparison:

```text
agent/src/runner.ts runHccAgent()
```

It is not deleted because the project still needs parity comparison while DeepReason-specific evaluation metrics are added.

### Executed Workflow

Complete synthetic cases execute:

```text
intake
-> feature_check
-> history_retrieve
-> prediction
-> shap_explain
-> evidence_retrieve
-> claim_check
-> source_verify
-> confidence_gate
-> report_generate
-> verify
-> memory_proposal
-> respond
```

Unsupported Claims execute the bounded retry branch:

```text
claim_check
-> build_follow_up_query
-> evidence_retrieve
-> claim_check
```

Default retry limit:

```text
maxEvidenceRetry: 2
```

If evidence remains insufficient, the Gate returns `limited` and the report exposes evidence gaps rather than generating unsupported medical explanations.

### API Compatibility And New Fields

The API keeps the existing frontend-compatible payload:

```text
safetyNotice
disclaimer
text
finishReason
steps
toolCalls
trace
analysis.completeness
analysis.history
analysis.prediction
analysis.explanation
analysis.evidence
analysis.memory
```

The API now also returns:

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
deepreason.evidenceItems
deepreason.sourceVerification
deepreason.gateDecision
deepreason.gateDecisions
deepreason.retryCount
deepreason.memoryProposal
deepreason.verificationResult
```

### Frontend Update

The Next.js page remains the product demo UI and now shows:

- DeepReason Gate status;
- Claim-Evidence count;
- evidence retry count;
- Memory Proposal status;
- Gate permitted and denied Claim counts;
- first Gate reason;
- Workflow Trace node list with agent, status, and duration.

The page remains a thin display layer. It does not compute prediction probabilities, SHAP values, citations, Gate decisions, or memory trends in browser code.

### Memory Boundary

The new DeepReason API path does not call the legacy direct `saveCaseMemory` tool.

Default behavior:

```text
createMemoryProposal -> pending_approval -> no case-memory write
```

If an `approvedBy` value is supplied and all Gate/Evidence/source checks pass, `createMemoryProposal` can apply the existing JSON writer through the controlled path.

For UI compatibility, `analysis.memory` still exists. When the proposal is pending, it reports `saved: false` and shows a non-persistent revisit comparison snapshot when prior synthetic history exists.

### Phase 8 Verification

Commands run:

```powershell
npm.cmd run typecheck
npm.cmd run web:typecheck
npm.cmd test
npm.cmd run eval
python -m pytest ml-service\tests -q
```

Results:

```text
npm run typecheck: passed
npm run web:typecheck: passed
npm test: 43 passed, 0 failed
npm run eval: overall pass rate 100.0%
python ml-service tests: 1 passed
```

The evaluation report still covers the existing M6/M8 deterministic metrics:

```text
overall pass rate: 100.0%
retrieval, patient workflow, and safety metrics: 20/20 each
full_hybrid retrieval Top-K: 100.0%
full_hybrid retrieval Top-1: 85.0%
full_hybrid retrieval MRR: 0.904
```

### Remaining Migration Work

Phase 8 completes the primary runtime/API switch, but the migration is not fully finished.

Remaining next-stage work:

- add DeepReason-specific metrics to `eval/run-eval.ts`;
- compare legacy runner vs DeepReason runtime on report parity;
- optionally expose `approvedBy` in the UI only if a demo needs controlled memory writes;
- decide when to deprecate the legacy Vercel AI SDK runner after parity metrics are stable.

## Phase 9 DeepReason Evaluation Update

Date: 2026-07-17

Phase 9 extends the evaluation loop so it now measures DeepReason runtime behavior in addition to the legacy deterministic prototype metrics.

Updated files:

```text
eval/run-eval.ts
eval/README.md
eval/reports/latest-report.json
eval/reports/latest-report.md
docs/DECISIONS.md
docs/DEEPREASON_HCC_MIGRATION_BASELINE.md
```

### What Changed

The evaluator still runs the original M6/M8 sections:

- 20 medical QA retrieval cases;
- 20 synthetic patient workflow cases through the legacy runner;
- 20 high-risk safety cases through the legacy runner;
- retrieval ablation over BM25, local hash embedding, rerank, query expansion, and diverse Top-K.

It now also runs:

```text
20 patient workflow cases through runHccDeepReasonWorkflow()
1 forced unsupported-Claim retry stress case
```

The DeepReason evaluator compares the migrated runtime against the legacy runner at the task-outcome level:

- complete cases still produce deterministic prediction, SHAP, and retrieval outputs;
- missing-feature cases still stop at feature clarification;
- Memory write behavior is intentionally different because DeepReason defaults to proposal-only writes.

### DeepReason Metrics Added

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

These metrics validate the framework migration, not clinical performance.

### DeepReason Diagnostics Added

The report also records observability values:

```text
average_workflow_node_count
average_runtime_ms
average_tool_call_count
gate_status_counts
memory_proposal_status_counts
```

These are diagnostics rather than correctness metrics, so they are not used directly as pass/fail rates.

### Current Phase 9 Results

Latest local run:

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

Latest diagnostics:

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

The forced retry stress case produced:

```text
retry_count: 2
gate_status: limited
denied_claim_ids: claim-forced-evidence-gap
evidence_gap_count: 1
passed: true
```

### Remaining Migration Work

After Phase 9, the main DeepReason migration loop is executable and measurable.

Remaining optional work:

- add UI approval controls if controlled memory writes need to be demonstrated;
- add more adversarial DeepReason-specific safety cases beyond the current forced unsupported Claim;
- decide when to retire the legacy Vercel AI SDK runner after enough parity history is collected;
- tune query expansion based on the ablation finding that removing expansion improves Top-1 and MRR on the small KB.
