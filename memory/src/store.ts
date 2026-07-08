import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { z } from "zod";
import {
  assertCompleteFeatures,
  causalFeatureNames,
  hccFeatureSchema,
  partialHccFeatureSchema,
  type HccFeatures,
  type PartialHccFeatures,
} from "../../agent/src/features.js";
import { SYNTHETIC_DATA_NOTICE } from "../../agent/src/safety.js";

const predictionSnapshotSchema = z
  .object({
    label: z.string(),
    probability_high_grade: z.number(),
    probability_low_or_intermediate: z.number(),
    uncertain_probability_band: z.boolean(),
  })
  .strict();

const shapSnapshotSchema = z
  .object({
    top_features: z.array(
      z
        .object({
          feature: z.string(),
          shap_value: z.number(),
          direction: z.string(),
          trust_level: z.string(),
        })
        .strict(),
    ),
    high_trust_features: z.array(z.string()),
    statistical_only_features: z.array(z.string()),
  })
  .strict();

const retrievalSnapshotSchema = z
  .object({
    confidence: z.enum(["high", "medium", "low"]),
    evidenceSufficient: z.boolean(),
    evidenceIds: z.array(z.string()),
  })
  .strict();

export const caseRecordSchema = z
  .object({
    id: z.string(),
    patientId: z.string(),
    sessionId: z.string(),
    timestamp: z.string(),
    safetyNotice: z.string(),
    features: hccFeatureSchema,
    prediction: predictionSnapshotSchema,
    shap: shapSnapshotSchema.optional(),
    retrieval: retrievalSnapshotSchema.optional(),
  })
  .strict();

const caseMemoryFileSchema = z
  .object({
    casesByPatientId: z.record(z.string(), z.array(caseRecordSchema)),
  })
  .strict();

const sessionRecordSchema = z
  .object({
    sessionId: z.string(),
    patientId: z.string().optional(),
    updatedAt: z.string(),
    features: partialHccFeatureSchema,
  })
  .strict();

const sessionMemoryFileSchema = z
  .object({
    sessionsById: z.record(z.string(), sessionRecordSchema),
  })
  .strict();

export type CaseRecord = z.infer<typeof caseRecordSchema>;

export type PatientHistoryResult = {
  patientId?: string;
  hasHistory: boolean;
  recordCount: number;
  latestRecord?: CaseRecord;
  safetyNotice: string;
};

export type CaseComparison = {
  hasPrevious: boolean;
  previousTimestamp?: string;
  probabilityDelta?: number;
  labelChanged?: boolean;
  changedFeatures: Array<{
    feature: string;
    previous: number;
    current: number;
    delta: number;
  }>;
  summary: string;
};

export type SaveCaseMemoryInput = {
  patientId?: string;
  sessionId: string;
  features: HccFeatures;
  prediction: z.infer<typeof predictionSnapshotSchema>;
  shap?: z.infer<typeof shapSnapshotSchema>;
  retrieval?: z.infer<typeof retrievalSnapshotSchema>;
  memoryDir?: string;
};

function defaultMemoryDir(): string {
  return resolve(process.cwd(), "memory", "data");
}

function ensureMemoryDir(memoryDir?: string): string {
  const dir = memoryDir ? resolve(memoryDir) : defaultMemoryDir();
  mkdirSync(dir, { recursive: true });
  return dir;
}

function readJsonFile<T>(filePath: string, fallback: T, schema: z.ZodType<T>): T {
  if (!existsSync(filePath)) {
    return fallback;
  }

  const parsed = JSON.parse(readFileSync(filePath, "utf-8")) as unknown;
  return schema.parse(parsed);
}

function writeJsonFile(filePath: string, value: unknown): void {
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function sessionFile(memoryDir?: string): string {
  return resolve(ensureMemoryDir(memoryDir), "session-memory.json");
}

function caseFile(memoryDir?: string): string {
  return resolve(ensureMemoryDir(memoryDir), "case-memory.json");
}

function listMissingFeatures(features: PartialHccFeatures): string[] {
  return causalFeatureNames.filter((name) => features[name] === undefined);
}

export function mergeSessionFeatures(options: {
  sessionId: string;
  patientId?: string;
  features: PartialHccFeatures;
  memoryDir?: string;
}) {
  const filePath = sessionFile(options.memoryDir);
  const memory = readJsonFile(
    filePath,
    { sessionsById: {} },
    sessionMemoryFileSchema,
  );
  const existing = memory.sessionsById[options.sessionId];
  const mergedFeatures = partialHccFeatureSchema.parse({
    ...(existing?.features ?? {}),
    ...options.features,
  });
  const missingFeatures = listMissingFeatures(mergedFeatures);
  const now = new Date().toISOString();

  memory.sessionsById[options.sessionId] = {
    sessionId: options.sessionId,
    patientId: options.patientId ?? existing?.patientId,
    updatedAt: now,
    features: mergedFeatures,
  };
  writeJsonFile(filePath, memory);

  return {
    sessionId: options.sessionId,
    patientId: options.patientId ?? existing?.patientId,
    complete: missingFeatures.length === 0,
    missingFeatures,
    receivedFeatures: causalFeatureNames.filter(
      (name) => mergedFeatures[name] !== undefined,
    ),
    requiredFeatures: causalFeatureNames,
    features: mergedFeatures,
    updatedAt: now,
    safetyNotice: SYNTHETIC_DATA_NOTICE,
  };
}

export function getPatientHistory(options: {
  patientId?: string;
  memoryDir?: string;
}): PatientHistoryResult {
  if (!options.patientId) {
    return {
      hasHistory: false,
      recordCount: 0,
      safetyNotice: SYNTHETIC_DATA_NOTICE,
    };
  }

  const memory = readJsonFile(
    caseFile(options.memoryDir),
    { casesByPatientId: {} },
    caseMemoryFileSchema,
  );
  const records = memory.casesByPatientId[options.patientId] ?? [];
  return {
    patientId: options.patientId,
    hasHistory: records.length > 0,
    recordCount: records.length,
    latestRecord: records.at(-1),
    safetyNotice: SYNTHETIC_DATA_NOTICE,
  };
}

function compareCases(previous: CaseRecord | undefined, current: CaseRecord): CaseComparison {
  if (!previous) {
    return {
      hasPrevious: false,
      changedFeatures: [],
      summary: "未找到该 patient_id 的历史分析记录，本次作为首次合成病例分析保存。",
    };
  }

  const changedFeatures = causalFeatureNames
    .map((feature) => {
      const previousValue = previous.features[feature];
      const currentValue = current.features[feature];
      return {
        feature,
        previous: previousValue,
        current: currentValue,
        delta: currentValue - previousValue,
      };
    })
    .filter((item) => Math.abs(item.delta) > 1e-9)
    .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta))
    .slice(0, 5);

  const probabilityDelta =
    current.prediction.probability_high_grade -
    previous.prediction.probability_high_grade;
  const labelChanged = current.prediction.label !== previous.prediction.label;
  const direction =
    probabilityDelta > 0.001
      ? "升高"
      : probabilityDelta < -0.001
        ? "降低"
        : "基本持平";

  return {
    hasPrevious: true,
    previousTimestamp: previous.timestamp,
    probabilityDelta,
    labelChanged,
    changedFeatures,
    summary: `较上次分析，高分级预测概率${direction} ${Math.abs(probabilityDelta * 100).toFixed(1)} 个百分点；预测标签${labelChanged ? "发生变化" : "未变化"}。`,
  };
}

export function saveCaseMemory(options: SaveCaseMemoryInput) {
  if (!options.patientId) {
    return {
      saved: false,
      recordCount: 0,
      comparison: {
        hasPrevious: false,
        changedFeatures: [],
        summary: "未提供 patient_id，未写入跨会话病例记忆。",
      } satisfies CaseComparison,
      safetyNotice: SYNTHETIC_DATA_NOTICE,
    };
  }

  const filePath = caseFile(options.memoryDir);
  const memory = readJsonFile(
    filePath,
    { casesByPatientId: {} },
    caseMemoryFileSchema,
  );
  const records = memory.casesByPatientId[options.patientId] ?? [];
  const features = assertCompleteFeatures(options.features);
  const now = new Date().toISOString();
  const record: CaseRecord = caseRecordSchema.parse({
    id: `${options.patientId}-${now}`,
    patientId: options.patientId,
    sessionId: options.sessionId,
    timestamp: now,
    safetyNotice: SYNTHETIC_DATA_NOTICE,
    features,
    prediction: options.prediction,
    shap: options.shap,
    retrieval: options.retrieval,
  });
  const previousRecord = records.at(-1);
  const comparison = compareCases(previousRecord, record);

  memory.casesByPatientId[options.patientId] = [...records, record];
  writeJsonFile(filePath, memory);

  return {
    saved: true,
    record,
    previousRecord,
    recordCount: records.length + 1,
    comparison,
    safetyNotice: SYNTHETIC_DATA_NOTICE,
  };
}

