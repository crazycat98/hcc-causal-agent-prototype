import { retrievalResultSchema } from "../../retrieval/src/schema.js";
import { z } from "zod";
import { hccFeatureSchema, partialHccFeatureSchema } from "./features.js";

export const predictionToolInputSchema = z
  .object({
    patientId: z.string().min(1).optional(),
    features: hccFeatureSchema,
  })
  .strict();

export type PredictionToolInput = z.infer<typeof predictionToolInputSchema>;

export const predictionServiceResponseSchema = z
  .object({
    safety_notice: z.string(),
    prediction: z.object({
      label: z.enum([
        "synthetic_high_pathology_grade",
        "synthetic_low_or_intermediate_grade",
      ]),
      probability_high_grade: z.number().min(0).max(1),
      probability_low_or_intermediate: z.number().min(0).max(1),
      uncertain_probability_band: z.boolean(),
    }),
    features_used: z.array(z.string()),
    input_echo: hccFeatureSchema,
    model: z.object({
      type: z.string(),
      feature_selection_method: z.string().nullish(),
      cv_auc_mean: z.number().nullish(),
      cv_auc_std: z.number().nullish(),
    }),
    disclaimer: z.string(),
    patient_id: z.string().nullish(),
  })
  .strict();

export type PredictionServiceResponse = z.infer<
  typeof predictionServiceResponseSchema
>;

const shapFeatureSchema = z
  .object({
    feature: z.string(),
    value: z.number(),
    shap_value: z.number(),
    abs_shap_value: z.number(),
    direction: z.enum([
      "pushes_toward_high_grade",
      "pushes_toward_low_or_intermediate",
    ]),
    trust_level: z.enum([
      "high_trust_causal_candidate",
      "statistical_association_only",
    ]),
    consistency_note: z.string(),
  })
  .strict();

export const explanationToolInputSchema = predictionToolInputSchema.extend({
  topN: z.number().int().min(1).max(10).default(5),
});

export const explanationServiceResponseSchema = z
  .object({
    safety_notice: z.string(),
    prediction: predictionServiceResponseSchema.shape.prediction,
    shap: z
      .object({
        method: z.literal("shap.TreeExplainer"),
        target_class: z.literal("synthetic_high_pathology_grade"),
        top_n: z.number().int().min(1).max(10),
        base_value: z.number(),
        top_features: z.array(shapFeatureSchema),
        causal_candidate_features: z.array(z.string()),
        high_trust_features: z.array(z.string()),
        statistical_only_features: z.array(z.string()),
        consistency_summary: z.string(),
        caveat: z.string(),
      })
      .strict(),
    disclaimer: z.string(),
    patient_id: z.string().nullish(),
  })
  .strict();

export type ExplanationServiceResponse = z.infer<
  typeof explanationServiceResponseSchema
>;

export const retrievalToolInputSchema = z
  .object({
    query: z.string().min(1),
    featureNames: z.array(z.string()).default([]),
    topK: z.number().int().min(1).max(8).default(5),
  })
  .strict();

export const retrievalToolOutputSchema = retrievalResultSchema;
export type RetrievalToolOutput = z.infer<typeof retrievalToolOutputSchema>;

export const featureCompletenessOutputSchema = z
  .object({
    sessionId: z.string(),
    patientId: z.string().optional(),
    complete: z.boolean(),
    missingFeatures: z.array(z.string()),
    receivedFeatures: z.array(z.string()),
    requiredFeatures: z.array(z.string()),
    features: partialHccFeatureSchema,
    updatedAt: z.string(),
    safetyNotice: z.string(),
  })
  .strict();

export type FeatureCompletenessOutput = z.infer<
  typeof featureCompletenessOutputSchema
>;

const caseRecordSchema = z
  .object({
    id: z.string(),
    patientId: z.string(),
    sessionId: z.string(),
    timestamp: z.string(),
    safetyNotice: z.string(),
    features: hccFeatureSchema,
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

const caseComparisonSchema = z
  .object({
    hasPrevious: z.boolean(),
    previousTimestamp: z.string().optional(),
    probabilityDelta: z.number().optional(),
    labelChanged: z.boolean().optional(),
    changedFeatures: z.array(
      z
        .object({
          feature: z.string(),
          previous: z.number(),
          current: z.number(),
          delta: z.number(),
        })
        .strict(),
    ),
    summary: z.string(),
  })
  .strict();

export const patientHistoryOutputSchema = z
  .object({
    patientId: z.string().optional(),
    hasHistory: z.boolean(),
    recordCount: z.number().int(),
    latestRecord: caseRecordSchema.optional(),
    safetyNotice: z.string(),
  })
  .strict();

export type PatientHistoryOutput = z.infer<typeof patientHistoryOutputSchema>;

export const saveCaseMemoryOutputSchema = z
  .object({
    saved: z.boolean(),
    record: caseRecordSchema.optional(),
    previousRecord: caseRecordSchema.optional(),
    recordCount: z.number().int(),
    comparison: caseComparisonSchema,
    safetyNotice: z.string(),
  })
  .strict();

export type SaveCaseMemoryOutput = z.infer<typeof saveCaseMemoryOutputSchema>;

