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

