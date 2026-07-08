# Evaluation

M6 builds a small evaluation loop for the prototype.

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

The current local run reports 100% across the measured prototype metrics. This does not imply clinical validity; it only means the local deterministic Agent pipeline passed the designed engineering tests.

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
