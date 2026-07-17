import { z } from "zod";
import {
  getPatientHistory,
  mergeSessionFeatures,
  saveCaseMemory,
} from "../../../memory/src/store.js";
import { retrieveMedicalEvidence } from "../../../retrieval/src/search.js";
import { assertCompleteFeatures, partialHccFeatureSchema } from "../features.js";
import {
  explanationServiceResponseSchema,
  explanationToolInputSchema,
  featureCompletenessOutputSchema,
  patientHistoryOutputSchema,
  predictionServiceResponseSchema,
  predictionToolInputSchema,
  retrievalToolInputSchema,
  retrievalToolOutputSchema,
  saveCaseMemoryOutputSchema,
} from "../predictionTypes.js";
import type {
  HccDeepReasonHandlerName,
  HccDeepReasonHandlerResult,
  HccDeepReasonHandlerSpec,
  HccDeepReasonToolAdapterOptions,
  HccEvidenceDraft,
} from "./types.js";
import { hccDeepReasonEvidenceHandlerSpecs } from "./evidence.js";

const featureCompletenessInputSchema = z
  .object({
    sessionId: z.string().min(1),
    patientId: z.string().min(1).optional(),
    features: partialHccFeatureSchema,
  })
  .strict();

const patientHistoryInputSchema = z
  .object({
    patientId: z.string().min(1).optional(),
  })
  .strict();

const saveCaseMemoryInputSchema = z
  .object({
    patientId: z.string().min(1).optional(),
    sessionId: z.string().min(1),
    features: predictionToolInputSchema.shape.features,
    prediction: predictionServiceResponseSchema.shape.prediction,
    shap: z
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
      .strict()
      .optional(),
    retrieval: z
      .object({
        confidence: z.enum(["high", "medium", "low"]),
        evidenceSufficient: z.boolean(),
        evidenceIds: z.array(z.string()),
      })
      .strict()
      .optional(),
  })
  .strict();

const handlerSpecs = {
  checkFeatureCompleteness: {
    handlerName: "checkFeatureCompleteness",
    nodeId: "feature_check",
    agentName: "feature_collector",
    description:
      "Merge synthetic HCC features into session working memory and report missing causal candidate fields.",
    permissions: ["read_session_memory"],
    deterministic: true,
  },
  getPatientHistory: {
    handlerName: "getPatientHistory",
    nodeId: "history_retrieve",
    agentName: "memory_manager",
    description:
      "Read synthetic cross-session case history for the current patient_id.",
    permissions: ["read_case_memory"],
    deterministic: true,
  },
  predictHccGrade: {
    handlerName: "predictHccGrade",
    nodeId: "prediction",
    agentName: "prediction_operator",
    description:
      "Call deterministic Random Forest prediction service. Never infer probabilities in the adapter.",
    permissions: ["call_prediction_service"],
    deterministic: true,
  },
  explainPredictionWithShap: {
    handlerName: "explainPredictionWithShap",
    nodeId: "shap_explain",
    agentName: "explanation_operator",
    description:
      "Call deterministic SHAP explanation service and preserve causal-SHAP trust labels.",
    permissions: ["call_shap_service"],
    deterministic: true,
  },
  retrieveMedicalEvidence: {
    handlerName: "retrieveMedicalEvidence",
    nodeId: "evidence_retrieve",
    agentName: "medical_retriever",
    description:
      "Run public traceable local medical retrieval and return only retrieved snippets.",
    permissions: ["retrieve_public_knowledge"],
    deterministic: true,
  },
  saveCaseMemory: {
    handlerName: "saveCaseMemory",
    nodeId: "case_memory_write",
    agentName: "memory_manager",
    description:
      "Legacy-compatible synthetic case memory write. Phase 6 will route this through Memory Proposal and Gate.",
    permissions: ["write_case_memory"],
    deterministic: true,
  },
} satisfies Record<
  | "checkFeatureCompleteness"
  | "getPatientHistory"
  | "predictHccGrade"
  | "explainPredictionWithShap"
  | "retrieveMedicalEvidence"
  | "saveCaseMemory",
  HccDeepReasonHandlerSpec
>;

async function postJson(endpoint: string, body: unknown): Promise<unknown> {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload
        ? String((payload as { error: unknown }).error)
        : `HTTP ${response.status}`;
    throw new Error(`DeepReason HCC tool adapter failed: ${message}`);
  }
  return payload;
}

function nowIso() {
  return new Date().toISOString();
}

async function withTrace<TOutput>(
  spec: HccDeepReasonHandlerSpec,
  action: () => Promise<{
    output: TOutput;
    evidenceDrafts?: HccEvidenceDraft[];
  }>,
): Promise<HccDeepReasonHandlerResult<TOutput>> {
  const startedAt = nowIso();
  const started = performance.now();
  const { output, evidenceDrafts = [] } = await action();
  const completedAt = nowIso();
  return {
    handlerName: spec.handlerName,
    nodeId: spec.nodeId,
    agentName: spec.agentName,
    output,
    evidenceDrafts,
    trace: {
      startedAt,
      completedAt,
      durationMs: Number((performance.now() - started).toFixed(3)),
    },
  };
}

function predictionEvidence(output: z.infer<typeof predictionServiceResponseSchema>): HccEvidenceDraft[] {
  return [
    {
      evidenceId: `ev-prediction-${output.patient_id ?? "anonymous"}`,
      sourceType: "model_prediction",
      claimType: "model_output",
      summary: `RF prediction ${output.prediction.label} with high-grade probability ${output.prediction.probability_high_grade}.`,
      confidence: 1,
      uri: "ml-service:/predict",
      locator: "prediction",
    },
  ];
}

function shapEvidence(output: z.infer<typeof explanationServiceResponseSchema>): HccEvidenceDraft[] {
  return [
    {
      evidenceId: `ev-shap-${output.patient_id ?? "anonymous"}`,
      sourceType: "model_explanation",
      claimType: "model_explanation",
      summary: `SHAP ${output.shap.method} returned ${output.shap.top_features.length} top feature contribution(s).`,
      confidence: 1,
      uri: "ml-service:/explain",
      locator: "shap.top_features",
    },
  ];
}

function retrievalEvidence(output: z.infer<typeof retrievalToolOutputSchema>): HccEvidenceDraft[] {
  return output.results.map((hit) => ({
    evidenceId: `ev-kb-${hit.id}`,
    sourceType: "knowledge_base",
    claimType: "medical_background",
    summary: `${hit.title}: ${hit.paragraph}`,
    confidence: hit.scores.final,
    uri: hit.source.url,
    locator: hit.id,
  }));
}

function historyEvidence(output: z.infer<typeof patientHistoryOutputSchema>): HccEvidenceDraft[] {
  if (!output.hasHistory || !output.latestRecord) {
    return [];
  }
  return [
    {
      evidenceId: `ev-history-${output.latestRecord.id}`,
      sourceType: "case_memory",
      claimType: "history",
      summary: `Found ${output.recordCount} synthetic historical record(s) for patient_id ${output.patientId}.`,
      confidence: 1,
      uri: "memory:/case-memory.json",
      locator: output.latestRecord.id,
    },
  ];
}

export function hccDeepReasonToolAdapterHandlerSpecs() {
  return Object.values(handlerSpecs);
}

export function hccDeepReasonHandlerSpecs() {
  return [
    ...hccDeepReasonToolAdapterHandlerSpecs(),
    ...hccDeepReasonEvidenceHandlerSpecs(),
  ];
}

export function createHccDeepReasonToolAdapters(
  options: HccDeepReasonToolAdapterOptions = {},
) {
  const predictionEndpoint =
    options.predictionEndpoint ??
    process.env.ML_PREDICTION_URL ??
    "http://127.0.0.1:8001/predict";
  const explanationEndpoint =
    options.explanationEndpoint ??
    process.env.ML_EXPLANATION_URL ??
    "http://127.0.0.1:8001/explain";

  return {
    checkFeatureCompleteness: async (
      input: z.infer<typeof featureCompletenessInputSchema>,
    ) =>
      withTrace(handlerSpecs.checkFeatureCompleteness, async () => {
        const parsed = featureCompletenessInputSchema.parse(input);
        const output = featureCompletenessOutputSchema.parse(
          mergeSessionFeatures({
            ...parsed,
            memoryDir: options.memoryDir,
          }),
        );
        return {
          output,
          evidenceDrafts: [
            {
              evidenceId: `ev-feature-state-${output.sessionId}`,
              sourceType: "session_state",
              claimType: "feature_state",
              summary: `Synthetic feature state complete=${output.complete}; missing=${output.missingFeatures.join(",") || "none"}.`,
              confidence: 1,
              uri: "memory:/session-memory.json",
              locator: output.sessionId,
            },
          ],
        };
      }),

    getPatientHistory: async (
      input: z.infer<typeof patientHistoryInputSchema>,
    ) =>
      withTrace(handlerSpecs.getPatientHistory, async () => {
        const parsed = patientHistoryInputSchema.parse(input);
        const output = patientHistoryOutputSchema.parse(
          getPatientHistory({
            ...parsed,
            memoryDir: options.memoryDir,
          }),
        );
        return {
          output,
          evidenceDrafts: historyEvidence(output),
        };
      }),

    predictHccGrade: async (input: z.infer<typeof predictionToolInputSchema>) =>
      withTrace(handlerSpecs.predictHccGrade, async () => {
        const parsed = predictionToolInputSchema.parse(input);
        const completeFeatures = assertCompleteFeatures(parsed.features);
        const payload = await postJson(predictionEndpoint, {
          patient_id: parsed.patientId,
          features: completeFeatures,
        });
        const output = predictionServiceResponseSchema.parse(payload);
        return {
          output,
          evidenceDrafts: predictionEvidence(output),
        };
      }),

    explainPredictionWithShap: async (
      input: z.infer<typeof explanationToolInputSchema>,
    ) =>
      withTrace(handlerSpecs.explainPredictionWithShap, async () => {
        const parsed = explanationToolInputSchema.parse(input);
        const completeFeatures = assertCompleteFeatures(parsed.features);
        const payload = await postJson(explanationEndpoint, {
          patient_id: parsed.patientId,
          features: completeFeatures,
          top_n: parsed.topN,
        });
        const output = explanationServiceResponseSchema.parse(payload);
        return {
          output,
          evidenceDrafts: shapEvidence(output),
        };
      }),

    retrieveMedicalEvidence: async (
      input: z.infer<typeof retrievalToolInputSchema>,
    ) =>
      withTrace(handlerSpecs.retrieveMedicalEvidence, async () => {
        const parsed = retrievalToolInputSchema.parse(input);
        const output = retrievalToolOutputSchema.parse(
          retrieveMedicalEvidence({
            query: parsed.query,
            featureNames: parsed.featureNames,
            topK: parsed.topK,
          }),
        );
        return {
          output,
          evidenceDrafts: retrievalEvidence(output),
        };
      }),

    saveCaseMemory: async (input: z.infer<typeof saveCaseMemoryInputSchema>) =>
      withTrace(handlerSpecs.saveCaseMemory, async () => {
        const parsed = saveCaseMemoryInputSchema.parse(input);
        const output = saveCaseMemoryOutputSchema.parse(
          saveCaseMemory({
            ...parsed,
            memoryDir: options.memoryDir,
          }),
        );
        return {
          output,
          evidenceDrafts: [],
        };
      }),
  };
}

export type HccDeepReasonToolAdapters = ReturnType<
  typeof createHccDeepReasonToolAdapters
>;
