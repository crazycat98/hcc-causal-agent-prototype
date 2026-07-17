# Evaluation

M6 built the first deterministic evaluation loop for the prototype. After the DeepReason migration, the same script also evaluates the new workflow runtime, Claim-Evidence layer, Gate behavior, and Memory Proposal boundary.

Safety notice: 演示用合成数据，非真实患者数据；非临床诊断依据。

## Datasets

- `medical-qa.json`: 20 retrieval relevance cases.
- `patient-cases.json`: 20 synthetic patient workflow cases.
- `safety-cases.json`: 20 adversarial safety cases.

## Run

```powershell
npm.cmd run eval
```

The script writes:

- `eval/reports/latest-report.json`
- `eval/reports/latest-report.md`

## Current Metrics

The current local run reports 100% across the measured prototype and DeepReason engineering metrics. This does not imply clinical validity; it only means the local deterministic Agent pipeline and DeepReason workflow passed the designed synthetic-data tests.

## Retrieval Ablation

`npm.cmd run eval` also writes a retrieval ablation table into `eval/reports/latest-report.md`.

Current ablation modes:

- `full_hybrid`
- `remove_bm25`
- `remove_embedding`
- `remove_rerank`
- `remove_query_expansion`
- `remove_diverse_topk`
- `bm25_only`
- `embedding_only`

Current retrieval ranking result on the 20-case medical QA set:

```text
full_hybrid Top-K evidence accuracy=100.0%
full_hybrid Top-1 evidence accuracy=85.0%
full_hybrid MRR=0.904
remove_query_expansion Top-1 evidence accuracy=95.0% (+10.0 pp)
remove_query_expansion MRR=0.967 (+6.3 pp)
```

The result suggests that the current query expansion is too broad for this small knowledge base and can dilute Top-1 ranking quality. This creates an explicit next optimization target.

## DeepReason Runtime Evaluation

`npm.cmd run eval` now also evaluates the migrated DeepReason runtime on the 20 synthetic patient workflow cases and one forced unsupported-claim retry stress case.

Current DeepReason results:

```text
deepreason_tool_boundary_rate=100.0% (20/20)
deepreason_claim_evidence_coverage_rate=100.0% (15/15 complete cases)
deepreason_source_verification_pass_rate=100.0% (15/15 complete cases)
deepreason_gate_decision_accuracy=100.0% (20/20)
deepreason_memory_proposal_gate_rate=100.0% (20/20)
deepreason_report_verification_rate=100.0% (20/20)
deepreason_legacy_parity_task_outcome_rate=100.0% (20/20)
deepreason_safety_disclaimer_retention_rate=100.0% (20/20)
deepreason_max_retry_evidence_gap_rate=100.0% (1/1)
```

Current diagnostics:

```text
evaluated_cases=20
complete_cases=15
missing_feature_cases=5
average_workflow_node_count=10.5
average_runtime_ms=21.56
average_tool_call_count=7.75
gate_status_counts={"allow":15,"clarification":5}
memory_proposal_status_counts={"pending_approval":15,"none":5}
```

These metrics describe framework behavior, not medical accuracy. The `deepreason_memory_proposal_gate_rate` intentionally expects pending proposals rather than direct memory writes unless an approval is supplied.
