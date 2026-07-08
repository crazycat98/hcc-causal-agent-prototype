# Memory Module

M5 implements local JSON memory for the prototype.

Safety notice: 演示用合成数据，非真实患者数据；非临床诊断依据。

## Layers

- Session working memory: keyed by `session_id`, stores the current synthetic feature collection state.
- Cross-session case memory: keyed by `patient_id`, stores previous synthetic analysis records for revisit comparison.

No real patient data should be stored. Runtime JSON files are ignored by git.

