# HCC Agent Evaluation Report

Generated at: 2026-07-17T05:17:10.608Z

Overall pass rate: 100.0%

## Metrics

| Metric | Score | Numerator | Denominator |
| --- | ---: | ---: | ---: |
| retrieval_expected_id_or_low_confidence_accuracy | 100.0% | 20 | 20 |
| retrieval_sufficiency_accuracy | 100.0% | 20 | 20 |
| retrieval_traceable_source_rate | 100.0% | 20 | 20 |
| patient_tool_sequence_accuracy | 100.0% | 20 | 20 |
| patient_report_field_completeness | 100.0% | 20 | 20 |
| patient_missing_feature_clarification_accuracy | 100.0% | 20 | 20 |
| patient_uncertainty_annotation_accuracy | 100.0% | 20 | 20 |
| patient_revisit_trend_accuracy | 100.0% | 20 | 20 |
| safety_disclaimer_retention_rate | 100.0% | 20 | 20 |
| safety_forbidden_claim_block_rate | 100.0% | 20 | 20 |
| safety_valid_citation_rate | 100.0% | 20 | 20 |
| safety_tool_boundary_rate | 100.0% | 20 | 20 |
| deepreason_tool_boundary_rate | 100.0% | 20 | 20 |
| deepreason_claim_evidence_coverage_rate | 100.0% | 15 | 15 |
| deepreason_source_verification_pass_rate | 100.0% | 15 | 15 |
| deepreason_gate_decision_accuracy | 100.0% | 20 | 20 |
| deepreason_memory_proposal_gate_rate | 100.0% | 20 | 20 |
| deepreason_report_verification_rate | 100.0% | 20 | 20 |
| deepreason_legacy_parity_task_outcome_rate | 100.0% | 20 | 20 |
| deepreason_safety_disclaimer_retention_rate | 100.0% | 20 | 20 |
| deepreason_max_retry_evidence_gap_rate | 100.0% | 1 | 1 |

## Retrieval Ablation

Delta columns are percentage-point changes versus `full_hybrid`; negative values mean the metric dropped after removing that component.

| Variant | Top-K Evidence | Δ pp | Top-1 Evidence | Δ pp | MRR | Δ pp | Sufficiency | Δ pp | Traceable Source | Δ pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_hybrid | 100.0% | 0.0 | 85.0% | 0.0 | 0.904 | 0.0 | 100.0% | 0.0 | 100.0% | 0.0 |
| remove_bm25 | 100.0% | 0.0 | 90.0% | 5.0 | 0.935 | 3.1 | 100.0% | 0.0 | 100.0% | 0.0 |
| remove_embedding | 100.0% | 0.0 | 90.0% | 5.0 | 0.929 | 2.5 | 100.0% | 0.0 | 100.0% | 0.0 |
| remove_rerank | 100.0% | 0.0 | 85.0% | 0.0 | 0.908 | 0.4 | 100.0% | 0.0 | 100.0% | 0.0 |
| remove_query_expansion | 100.0% | 0.0 | 95.0% | 10.0 | 0.967 | 6.3 | 100.0% | 0.0 | 100.0% | 0.0 |
| remove_diverse_topk | 100.0% | 0.0 | 85.0% | 0.0 | 0.904 | 0.0 | 100.0% | 0.0 | 100.0% | 0.0 |
| bm25_only | 100.0% | 0.0 | 90.0% | 5.0 | 0.929 | 2.5 | 100.0% | 0.0 | 100.0% | 0.0 |
| embedding_only | 100.0% | 0.0 | 90.0% | 5.0 | 0.942 | 3.7 | 100.0% | 0.0 | 100.0% | 0.0 |

## DeepReason Runtime Evaluation

| Metric | Score | Numerator | Denominator |
| --- | ---: | ---: | ---: |
| deepreason_tool_boundary_rate | 100.0% | 20 | 20 |
| deepreason_claim_evidence_coverage_rate | 100.0% | 15 | 15 |
| deepreason_source_verification_pass_rate | 100.0% | 15 | 15 |
| deepreason_gate_decision_accuracy | 100.0% | 20 | 20 |
| deepreason_memory_proposal_gate_rate | 100.0% | 20 | 20 |
| deepreason_report_verification_rate | 100.0% | 20 | 20 |
| deepreason_legacy_parity_task_outcome_rate | 100.0% | 20 | 20 |
| deepreason_safety_disclaimer_retention_rate | 100.0% | 20 | 20 |
| deepreason_max_retry_evidence_gap_rate | 100.0% | 1 | 1 |

### DeepReason Diagnostics

- evaluated_cases: 20
- complete_cases: 15
- missing_feature_cases: 5
- average_workflow_node_count: 10.5
- average_runtime_ms: 21.56
- average_tool_call_count: 7.75
- gate_status_counts: {"allow":15,"clarification":5}
- memory_proposal_status_counts: {"pending_approval":15,"none":5}

### DeepReason Retry Stress

- retry_count: 2
- gate_status: limited
- denied_claim_ids: claim-forced-evidence-gap
- evidence_gap_count: 1
- passed: true

## Safety Note

All evaluated inputs and memory records are synthetic demo data. This evaluation does not validate clinical performance.

