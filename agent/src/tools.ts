import { tool } from "ai";
import { z } from "zod";
import {
  getPatientHistory,
  mergeSessionFeatures,
  saveCaseMemory,
} from "../../memory/src/store.js";
import { retrieveMedicalEvidence } from "../../retrieval/src/search.js";
import { assertCompleteFeatures, partialHccFeatureSchema } from "./features.js";
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
} from "./predictionTypes.js";

export type HccAgentToolOptions = {
  predictionEndpoint?: string;
  explanationEndpoint?: string;
  memoryDir?: string;
};

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
    throw new Error(`ML tool failed: ${message}`);
  }

  return payload;
}

export function createHccAgentTools(options: HccAgentToolOptions = {}) {
  const predictionEndpoint =
    options.predictionEndpoint ??
    process.env.ML_PREDICTION_URL ??
    "http://127.0.0.1:8001/predict";
  const explanationEndpoint =
    options.explanationEndpoint ??
    process.env.ML_EXPLANATION_URL ??
    "http://127.0.0.1:8001/explain";

  return {
    checkFeatureCompleteness: tool({
      description:
        "Merge current synthetic features into session working memory, then check whether all 10 causal candidate features are complete.",
      inputSchema: z
        .object({
          sessionId: z.string().min(1),
          patientId: z.string().min(1).optional(),
          features: partialHccFeatureSchema,
        })
        .strict(),
      execute: async ({ sessionId, patientId, features }) => {
        return featureCompletenessOutputSchema.parse(
          mergeSessionFeatures({
            sessionId,
            patientId,
            features,
            memoryDir: options.memoryDir,
          }),
        );
      },
    }),

    getPatientHistory: tool({
      description:
        "Read cross-session synthetic case memory for the same patient_id before analysis.",
      inputSchema: z
        .object({
          patientId: z.string().min(1).optional(),
        })
        .strict(),
      execute: async ({ patientId }) => {
        return patientHistoryOutputSchema.parse(
          getPatientHistory({
            patientId,
            memoryDir: options.memoryDir,
          }),
        );
      },
    }),

    predictHccGrade: tool({
      description:
        "Call the deterministic M1 Random Forest prediction service. The Agent must never invent risk labels or probabilities without this tool output.",
      inputSchema: predictionToolInputSchema,
      strict: true,
      execute: async ({ patientId, features }) => {
        const completeFeatures = assertCompleteFeatures(features);
        const payload = await postJson(predictionEndpoint, {
          patient_id: patientId,
          features: completeFeatures,
        });
        return predictionServiceResponseSchema.parse(payload);
      },
    }),

    explainPredictionWithShap: tool({
      description:
        "Call the deterministic M3 SHAP service and return Top-N model contributions plus causal-candidate consistency labels.",
      inputSchema: explanationToolInputSchema,
      strict: true,
      execute: async ({ patientId, features, topN }) => {
        const completeFeatures = assertCompleteFeatures(features);
        const payload = await postJson(explanationEndpoint, {
          patient_id: patientId,
          features: completeFeatures,
          top_n: topN,
        });
        return explanationServiceResponseSchema.parse(payload);
      },
    }),

    retrieveMedicalEvidence: tool({
      description:
        "Retrieve public, traceable medical background snippets with BM25 + local embedding + rerank. The Agent may only cite returned paragraphs.",
      inputSchema: retrievalToolInputSchema,
      strict: true,
      execute: async ({ query, featureNames, topK }) => {
        return retrievalToolOutputSchema.parse(
          retrieveMedicalEvidence({
            query,
            featureNames,
            topK,
          }),
        );
      },
    }),

    saveCaseMemory: tool({
      description:
        "Persist the completed synthetic analysis and return revisit comparison against previous patient_id record.",
      inputSchema: z
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
        .strict(),
      execute: async (input) => {
        return saveCaseMemoryOutputSchema.parse(
          saveCaseMemory({
            ...input,
            memoryDir: options.memoryDir,
          }),
        );
      },
    }),
  };
}

export type HccAgentTools = ReturnType<typeof createHccAgentTools>;

