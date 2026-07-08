# Agent Service

M2-M5 wraps deterministic ML, retrieval, and memory modules as Vercel AI SDK tools.

Safety notice: 演示用合成数据，非真实患者数据；非临床诊断依据。

## Current Scope

- Uses `generateText`, `tool`, and `isStepCount` from Vercel AI SDK 7.
- Defines strict Zod schemas for the 10 synthetic causal candidate features.
- Runs a multi-step tool loop:
  1. `checkFeatureCompleteness`
  2. `getPatientHistory`
  3. `predictHccGrade`
  4. `explainPredictionWithShap`
  5. `retrieveMedicalEvidence`
  6. `saveCaseMemory`
  7. final structured report text based only on tool output
- Uses `MockLanguageModelV4` by default for an offline demo with no LLM API key.

AI SDK 7 uses `stopWhen: isStepCount(n)` for loop control. This is the current-version equivalent of the older `maxSteps` examples.

## Demo

Start the M1-M3 ML service in one terminal:

```powershell
npm.cmd run m1:serve
```

Run the Agent demo in another terminal:

```powershell
npm.cmd run m5:demo
```

