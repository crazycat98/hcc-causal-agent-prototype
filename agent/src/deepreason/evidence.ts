import { createHash } from "node:crypto";
import { z } from "zod";
import {
  getPatientHistory,
  saveCaseMemory,
} from "../../../memory/src/store.js";
import { hccFeatureSchema } from "../features.js";
import {
  predictionServiceResponseSchema,
  saveCaseMemoryOutputSchema,
} from "../predictionTypes.js";
import { REPORT_DISCLAIMER, SYNTHETIC_DATA_NOTICE } from "../safety.js";
import type {
  HccDeepReasonHandlerResult,
  HccDeepReasonHandlerSpec,
  HccEvidenceDraft,
} from "./types.js";

const sourceTypeSchema = z.enum([
  "session_state",
  "case_memory",
  "model_prediction",
  "model_explanation",
  "knowledge_base",
]);
const claimTypeSchema = z.enum([
  "feature_state",
  "history",
  "model_output",
  "model_explanation",
  "medical_background",
]);
const supportStatusSchema = z.enum([
  "supported",
  "partially_supported",
  "unsupported",
]);
const gateStatusSchema = z.enum(["allow", "limited", "interrupt", "deny"]);
const memoryProposalStatusSchema = z.enum([
  "pending_approval",
  "approved_applied",
  "rejected",
]);

const hccEvidenceDraftSchema = z
  .object({
    evidenceId: z.string().min(1),
    sourceType: sourceTypeSchema,
    claimType: claimTypeSchema,
    summary: z.string().min(1),
    confidence: z.number().min(0).max(1),
    uri: z.string().min(1).optional(),
    locator: z.string().min(1).optional(),
  })
  .strict();

export const hccEvidenceItemSchema = z
  .object({
    evidenceId: z.string().min(1),
    sourceType: sourceTypeSchema,
    claimType: claimTypeSchema,
    content: z.string().min(1),
    confidence: z.number().min(0).max(1),
    source: z
      .object({
        uri: z.string().min(1),
        locator: z.string().min(1).optional(),
        label: z.string().min(1).optional(),
      })
      .strict(),
    contentHash: z.string().min(12),
    syntheticDataNotice: z.literal(SYNTHETIC_DATA_NOTICE),
  })
  .strict();

export const hccRequiredClaimSchema = z
  .object({
    claimId: z.string().min(1).optional(),
    claim: z.string().min(1),
    claimType: claimTypeSchema,
    requiredSourceTypes: z.array(sourceTypeSchema).default([]),
    requiredTerms: z.array(z.string().min(1)).default([]),
    minEvidence: z.number().int().min(1).default(1),
    confidenceThreshold: z.number().min(0).max(1).default(0),
  })
  .strict();

export const hccClaimEvidenceSchema = z
  .object({
    claimId: z.string().min(1),
    claim: z.string().min(1),
    claimType: claimTypeSchema,
    evidenceIds: z.array(z.string().min(1)),
    supportStatus: supportStatusSchema,
    confidence: z.number().min(0).max(1),
    requiredFollowUp: z.array(z.string().min(1)),
    notes: z.array(z.string().min(1)),
  })
  .strict();

export const checkClaimsInputSchema = z
  .object({
    evidenceDrafts: z.array(hccEvidenceDraftSchema),
    requiredClaims: z.array(hccRequiredClaimSchema).default([]),
    evidenceRetryCount: z.number().int().min(0).default(0),
  })
  .strict();

export const checkClaimsOutputSchema = z
  .object({
    evidenceItems: z.array(hccEvidenceItemSchema),
    claimEvidenceMap: z.array(hccClaimEvidenceSchema),
    unsupportedClaims: z.array(hccClaimEvidenceSchema),
    evidenceGaps: z.array(z.string().min(1)),
    evidenceRetryCount: z.number().int().min(0),
    supportSummary: z
      .object({
        supported: z.number().int().min(0),
        partiallySupported: z.number().int().min(0),
        unsupported: z.number().int().min(0),
      })
      .strict(),
    safetyNotice: z.literal(SYNTHETIC_DATA_NOTICE),
  })
  .strict();

export const buildFollowUpQueryInputSchema = z
  .object({
    unsupportedClaims: z.array(hccClaimEvidenceSchema),
    evidenceRetryCount: z.number().int().min(0).default(0),
    maxRetry: z.number().int().min(1).max(3).default(2),
    baseQuery: z.string().min(1).optional(),
  })
  .strict();

export const buildFollowUpQueryOutputSchema = z
  .object({
    shouldRetry: z.boolean(),
    followUpQuery: z.string(),
    unsupportedClaimIds: z.array(z.string().min(1)),
    evidenceRetryCount: z.number().int().min(0),
    maxRetry: z.number().int().min(1).max(3),
    reason: z.string().min(1),
    safetyNotice: z.literal(SYNTHETIC_DATA_NOTICE),
  })
  .strict();

export const verifyEvidenceSourcesInputSchema = z
  .object({
    evidenceItems: z.array(hccEvidenceItemSchema),
    claimEvidenceMap: z.array(hccClaimEvidenceSchema),
  })
  .strict();

export const verifyEvidenceSourcesOutputSchema = z
  .object({
    valid: z.boolean(),
    checkedEvidenceCount: z.number().int().min(0),
    verifiedEvidenceIds: z.array(z.string().min(1)),
    invalidEvidenceIds: z.array(z.string().min(1)),
    duplicateEvidenceIds: z.array(z.string().min(1)),
    duplicateContentHashes: z.array(z.string().min(1)),
    missingEvidenceReferences: z.array(z.string().min(1)),
    conflictWarnings: z.array(z.string().min(1)),
    sourceCompletenessRate: z.number().min(0).max(1),
    claimCoverageRate: z.number().min(0).max(1),
    safetyNotice: z.literal(SYNTHETIC_DATA_NOTICE),
  })
  .strict();

export const hccGateDecisionSchema = z
  .object({
    gateId: z.string().min(1),
    status: gateStatusSchema,
    riskLevel: z.enum(["low", "medium", "high"]),
    permittedClaimIds: z.array(z.string().min(1)),
    limitedClaimIds: z.array(z.string().min(1)),
    deniedClaimIds: z.array(z.string().min(1)),
    reasons: z.array(z.string().min(1)),
    evidenceGaps: z.array(z.string().min(1)),
    requiredActions: z.array(z.string().min(1)),
    requiresDisclaimer: z.literal(true),
    safetyNotice: z.literal(SYNTHETIC_DATA_NOTICE),
  })
  .strict();

export const evaluateMedicalConfidenceGateInputSchema = z
  .object({
    evidenceItems: z.array(hccEvidenceItemSchema),
    claimEvidenceMap: z.array(hccClaimEvidenceSchema),
    sourceVerification: verifyEvidenceSourcesOutputSchema,
    disclaimerIncluded: z.boolean().default(true),
    evidenceRetryExhausted: z.boolean().default(false),
  })
  .strict();

export const evaluateMedicalConfidenceGateOutputSchema = hccGateDecisionSchema;

export const hccReportSectionSchema = z
  .object({
    title: z.string().min(1),
    content: z.string().min(1),
    evidenceIds: z.array(z.string().min(1)),
  })
  .strict();

export const generateHccReportInputSchema = z
  .object({
    gateDecision: hccGateDecisionSchema,
    evidenceItems: z.array(hccEvidenceItemSchema),
    claimEvidenceMap: z.array(hccClaimEvidenceSchema),
    sourceVerification: verifyEvidenceSourcesOutputSchema,
    patientId: z.string().min(1).optional(),
  })
  .strict();

export const generateHccReportOutputSchema = z
  .object({
    reportId: z.string().min(1),
    status: gateStatusSchema,
    markdown: z.string().min(1),
    sections: z.array(hccReportSectionSchema),
    citedEvidenceIds: z.array(z.string().min(1)),
    omittedClaimIds: z.array(z.string().min(1)),
    generationRules: z.array(z.string().min(1)),
    safetyNotice: z.literal(SYNTHETIC_DATA_NOTICE),
    disclaimer: z.literal(REPORT_DISCLAIMER),
  })
  .strict();

const memoryShapSnapshotSchema = z
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

const memoryRetrievalSnapshotSchema = z
  .object({
    confidence: z.enum(["high", "medium", "low"]),
    evidenceSufficient: z.boolean(),
    evidenceIds: z.array(z.string().min(1)),
  })
  .strict();

export const createMemoryProposalInputSchema = z
  .object({
    patientId: z.string().min(1).optional(),
    sessionId: z.string().min(1),
    features: hccFeatureSchema,
    prediction: predictionServiceResponseSchema.shape.prediction,
    shap: memoryShapSnapshotSchema.optional(),
    retrieval: memoryRetrievalSnapshotSchema.optional(),
    gateDecision: hccGateDecisionSchema,
    evidenceItems: z.array(hccEvidenceItemSchema),
    sourceVerification: verifyEvidenceSourcesOutputSchema,
    approvedBy: z.string().min(1).optional(),
    memoryDir: z.string().min(1).optional(),
  })
  .strict();

export const createMemoryProposalOutputSchema = z
  .object({
    proposalId: z.string().min(1),
    status: memoryProposalStatusSchema,
    targetPartition: z.literal("case_memory"),
    patientId: z.string().min(1).optional(),
    sessionId: z.string().min(1),
    requiresApproval: z.literal(true),
    approvedBy: z.string().min(1).optional(),
    canApply: z.boolean(),
    applied: z.boolean(),
    directWrite: z.literal(false),
    recordCountBefore: z.number().int().min(0),
    recordCountAfter: z.number().int().min(0),
    evidenceIds: z.array(z.string().min(1)),
    gateStatus: gateStatusSchema,
    reasons: z.array(z.string().min(1)),
    blockedReasons: z.array(z.string().min(1)),
    appliedResult: saveCaseMemoryOutputSchema.optional(),
    safetyNotice: z.literal(SYNTHETIC_DATA_NOTICE),
  })
  .strict();

export type HccEvidenceItem = z.infer<typeof hccEvidenceItemSchema>;
export type HccRequiredClaim = z.infer<typeof hccRequiredClaimSchema>;
export type HccClaimEvidence = z.infer<typeof hccClaimEvidenceSchema>;
export type CheckClaimsInput = z.input<typeof checkClaimsInputSchema>;
export type CheckClaimsOutput = z.infer<typeof checkClaimsOutputSchema>;
export type BuildFollowUpQueryInput = z.input<
  typeof buildFollowUpQueryInputSchema
>;
export type BuildFollowUpQueryOutput = z.infer<
  typeof buildFollowUpQueryOutputSchema
>;
export type VerifyEvidenceSourcesInput = z.input<
  typeof verifyEvidenceSourcesInputSchema
>;
export type VerifyEvidenceSourcesOutput = z.infer<
  typeof verifyEvidenceSourcesOutputSchema
>;
export type HccGateDecision = z.infer<typeof hccGateDecisionSchema>;
export type EvaluateMedicalConfidenceGateInput = z.input<
  typeof evaluateMedicalConfidenceGateInputSchema
>;
export type EvaluateMedicalConfidenceGateOutput = z.infer<
  typeof evaluateMedicalConfidenceGateOutputSchema
>;
export type GenerateHccReportInput = z.input<typeof generateHccReportInputSchema>;
export type GenerateHccReportOutput = z.infer<typeof generateHccReportOutputSchema>;
export type CreateMemoryProposalInput = z.input<
  typeof createMemoryProposalInputSchema
>;
export type CreateMemoryProposalOutput = z.infer<
  typeof createMemoryProposalOutputSchema
>;

const evidenceHandlerSpecs = {
  checkClaims: {
    handlerName: "checkClaims",
    nodeId: "claim_check",
    agentName: "claim_checker",
    description:
      "Bind deterministic model, SHAP, and public KB evidence to structured claims.",
    permissions: ["bind_claim_evidence"],
    deterministic: true,
  },
  buildFollowUpQuery: {
    handlerName: "buildFollowUpQuery",
    nodeId: "build_follow_up_query",
    agentName: "medical_retriever",
    description:
      "Create a focused retrieval query from unsupported claims without changing tool outputs.",
    permissions: ["build_follow_up_query", "retrieve_public_knowledge"],
    deterministic: true,
  },
  verifyEvidenceSources: {
    handlerName: "verifyEvidenceSources",
    nodeId: "source_verify",
    agentName: "source_verifier",
    description:
      "Verify evidence source URI, locator, duplicate IDs, and claim references.",
    permissions: ["verify_evidence_sources"],
    deterministic: true,
  },
  evaluateMedicalConfidenceGate: {
    handlerName: "evaluateMedicalConfidenceGate",
    nodeId: "confidence_gate",
    agentName: "medical_safety_reviewer",
    description:
      "Apply deterministic medical safety, evidence, source, and disclaimer gates.",
    permissions: ["evaluate_medical_confidence_gate"],
    deterministic: true,
  },
  generateHccReport: {
    handlerName: "generateHccReport",
    nodeId: "report_generate",
    agentName: "report_writer",
    description:
      "Generate a structured synthetic HCC report from permitted claims and Evidence IDs only.",
    permissions: ["generate_evidence_report"],
    deterministic: true,
  },
  createMemoryProposal: {
    handlerName: "createMemoryProposal",
    nodeId: "memory_proposal",
    agentName: "memory_manager",
    description:
      "Create an auditable synthetic case memory proposal and apply it only when approved.",
    permissions: ["create_memory_proposal", "apply_approved_memory_proposal"],
    deterministic: true,
  },
} satisfies Record<
  | "checkClaims"
  | "buildFollowUpQuery"
  | "verifyEvidenceSources"
  | "evaluateMedicalConfidenceGate"
  | "generateHccReport"
  | "createMemoryProposal",
  HccDeepReasonHandlerSpec
>;

function nowIso() {
  return new Date().toISOString();
}

async function withTrace<TOutput>(
  spec: HccDeepReasonHandlerSpec,
  action: () => Promise<TOutput>,
): Promise<HccDeepReasonHandlerResult<TOutput>> {
  const startedAt = nowIso();
  const started = performance.now();
  const output = await action();
  const completedAt = nowIso();
  return {
    handlerName: spec.handlerName,
    nodeId: spec.nodeId,
    agentName: spec.agentName,
    output,
    evidenceDrafts: [],
    trace: {
      startedAt,
      completedAt,
      durationMs: Number((performance.now() - started).toFixed(3)),
    },
  };
}

function stableHash(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function normalizeText(value: string): string {
  return value.toLowerCase().replaceAll("_", " ").replaceAll("-", " ");
}

function confidenceAverage(items: HccEvidenceItem[]): number {
  if (items.length === 0) {
    return 0;
  }
  return Number(
    (
      items.reduce((sum, item) => sum + item.confidence, 0) / items.length
    ).toFixed(4),
  );
}

function toEvidenceItem(draft: HccEvidenceDraft): HccEvidenceItem {
  const uri = draft.uri ?? `deepreason:${draft.sourceType}`;
  const label =
    draft.sourceType === "knowledge_base"
      ? "public_rewritten_knowledge_base"
      : draft.sourceType;
  const contentHash = stableHash(
    [draft.evidenceId, draft.sourceType, draft.claimType, draft.summary, uri, draft.locator]
      .filter(Boolean)
      .join("|"),
  );
  return hccEvidenceItemSchema.parse({
    evidenceId: draft.evidenceId,
    sourceType: draft.sourceType,
    claimType: draft.claimType,
    content: draft.summary,
    confidence: draft.confidence,
    source: {
      uri,
      locator: draft.locator,
      label,
    },
    contentHash,
    syntheticDataNotice: SYNTHETIC_DATA_NOTICE,
  });
}

function generatedRequiredClaims(
  evidenceItems: HccEvidenceItem[],
): HccRequiredClaim[] {
  const claims: HccRequiredClaim[] = [
    {
      claimId: "claim-model-output",
      claim: "Random Forest prediction output is available from the deterministic prediction tool.",
      claimType: "model_output",
      requiredSourceTypes: ["model_prediction"],
      requiredTerms: [],
      minEvidence: 1,
      confidenceThreshold: 1,
    },
    {
      claimId: "claim-shap-explanation",
      claim: "SHAP explanation output is available from the deterministic SHAP tool.",
      claimType: "model_explanation",
      requiredSourceTypes: ["model_explanation"],
      requiredTerms: [],
      minEvidence: 1,
      confidenceThreshold: 1,
    },
    {
      claimId: "claim-medical-background",
      claim: "Traceable public medical background evidence is available for report explanation.",
      claimType: "medical_background",
      requiredSourceTypes: ["knowledge_base"],
      requiredTerms: [],
      minEvidence: 1,
      confidenceThreshold: 0.25,
    },
  ];

  const hasHistory = evidenceItems.some((item) => item.sourceType === "case_memory");
  if (hasHistory) {
    claims.push({
      claimId: "claim-case-history",
      claim: "Synthetic case history is available for revisit comparison.",
      claimType: "history",
      requiredSourceTypes: ["case_memory"],
      requiredTerms: [],
      minEvidence: 1,
      confidenceThreshold: 1,
    });
  }

  return claims;
}

function supportingEvidence(
  claim: HccRequiredClaim,
  evidenceItems: HccEvidenceItem[],
): HccEvidenceItem[] {
  const terms = claim.requiredTerms.map(normalizeText);
  return evidenceItems.filter((item) => {
    if (
      claim.requiredSourceTypes.length > 0 &&
      !claim.requiredSourceTypes.includes(item.sourceType)
    ) {
      return false;
    }
    if (item.confidence < claim.confidenceThreshold) {
      return false;
    }
    if (terms.length === 0) {
      return true;
    }
    const haystack = normalizeText(
      [
        item.evidenceId,
        item.content,
        item.source.uri,
        item.source.locator ?? "",
        item.source.label ?? "",
      ].join(" "),
    );
    return terms.some((term) => haystack.includes(term));
  });
}

function followUpForClaim(claim: HccRequiredClaim): string[] {
  const sourceHint =
    claim.requiredSourceTypes.length > 0
      ? `source_type:${claim.requiredSourceTypes.join("|")}`
      : "source_type:any";
  const termHint =
    claim.requiredTerms.length > 0
      ? `terms:${claim.requiredTerms.join(" ")}`
      : claim.claim;
  return [`${sourceHint} ${termHint}`];
}

function bindClaim(
  claim: HccRequiredClaim,
  evidenceItems: HccEvidenceItem[],
  index: number,
): HccClaimEvidence {
  const supporting = supportingEvidence(claim, evidenceItems);
  const evidenceIds = supporting
    .slice(0, Math.max(claim.minEvidence, supporting.length))
    .map((item) => item.evidenceId);
  const supportStatus =
    supporting.length >= claim.minEvidence
      ? "supported"
      : supporting.length > 0
        ? "partially_supported"
        : "unsupported";
  const notes =
    supportStatus === "supported"
      ? ["Evidence requirement satisfied."]
      : [
          `Evidence requirement not satisfied: need ${claim.minEvidence}, found ${supporting.length}.`,
        ];

  return hccClaimEvidenceSchema.parse({
    claimId: claim.claimId ?? `claim-${String(index + 1).padStart(3, "0")}`,
    claim: claim.claim,
    claimType: claim.claimType,
    evidenceIds,
    supportStatus,
    confidence: confidenceAverage(supporting),
    requiredFollowUp:
      supportStatus === "supported" ? [] : followUpForClaim(claim),
    notes,
  });
}

export function checkClaims(input: CheckClaimsInput): CheckClaimsOutput {
  const parsed = checkClaimsInputSchema.parse(input);
  const evidenceItems = parsed.evidenceDrafts.map(toEvidenceItem);
  const requiredClaims =
    parsed.requiredClaims.length > 0
      ? parsed.requiredClaims
      : generatedRequiredClaims(evidenceItems);
  const claimEvidenceMap = requiredClaims.map((claim, index) =>
    bindClaim(claim, evidenceItems, index),
  );
  const unsupportedClaims = claimEvidenceMap.filter(
    (claim) => claim.supportStatus !== "supported",
  );
  const supportSummary = {
    supported: claimEvidenceMap.filter((claim) => claim.supportStatus === "supported")
      .length,
    partiallySupported: claimEvidenceMap.filter(
      (claim) => claim.supportStatus === "partially_supported",
    ).length,
    unsupported: claimEvidenceMap.filter(
      (claim) => claim.supportStatus === "unsupported",
    ).length,
  };

  return checkClaimsOutputSchema.parse({
    evidenceItems,
    claimEvidenceMap,
    unsupportedClaims,
    evidenceGaps: unsupportedClaims.flatMap((claim) => claim.requiredFollowUp),
    evidenceRetryCount: parsed.evidenceRetryCount,
    supportSummary,
    safetyNotice: SYNTHETIC_DATA_NOTICE,
  });
}

function compactQueryParts(parts: string[]): string {
  const seen = new Set<string>();
  const words = parts
    .join(" ")
    .split(/\s+/)
    .map((part) => part.trim())
    .filter(Boolean)
    .filter((part) => {
      const key = part.toLowerCase();
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  return words.slice(0, 80).join(" ");
}

export function buildFollowUpQuery(
  input: BuildFollowUpQueryInput,
): BuildFollowUpQueryOutput {
  const parsed = buildFollowUpQueryInputSchema.parse(input);
  const actionableClaims = parsed.unsupportedClaims.filter(
    (claim) => claim.supportStatus !== "supported",
  );
  const shouldRetry =
    actionableClaims.length > 0 && parsed.evidenceRetryCount < parsed.maxRetry;
  const nextRetryCount = shouldRetry
    ? parsed.evidenceRetryCount + 1
    : parsed.evidenceRetryCount;
  const followUpQuery = shouldRetry
    ? compactQueryParts([
        parsed.baseQuery ?? "HCC pathology grade public evidence",
        ...actionableClaims.flatMap((claim) =>
          claim.requiredFollowUp.length > 0
            ? claim.requiredFollowUp
            : [claim.claim],
        ),
      ])
    : "";
  const reason =
    actionableClaims.length === 0
      ? "No unsupported claims remain."
      : parsed.evidenceRetryCount >= parsed.maxRetry
        ? "Evidence retry limit reached; route to confidence gate with evidence gaps."
        : "Unsupported claims require focused follow-up retrieval.";

  return buildFollowUpQueryOutputSchema.parse({
    shouldRetry,
    followUpQuery,
    unsupportedClaimIds: actionableClaims.map((claim) => claim.claimId),
    evidenceRetryCount: nextRetryCount,
    maxRetry: parsed.maxRetry,
    reason,
    safetyNotice: SYNTHETIC_DATA_NOTICE,
  });
}

function duplicateValues(values: string[]): string[] {
  const seen = new Set<string>();
  const duplicates = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      duplicates.add(value);
    }
    seen.add(value);
  }
  return [...duplicates].sort();
}

export function verifyEvidenceSources(
  input: VerifyEvidenceSourcesInput,
): VerifyEvidenceSourcesOutput {
  const parsed = verifyEvidenceSourcesInputSchema.parse(input);
  const evidenceById = new Map(
    parsed.evidenceItems.map((item) => [item.evidenceId, item]),
  );
  const duplicateEvidenceIds = duplicateValues(
    parsed.evidenceItems.map((item) => item.evidenceId),
  );
  const duplicateContentHashes = duplicateValues(
    parsed.evidenceItems.map((item) => item.contentHash),
  );
  const invalidEvidenceIds = parsed.evidenceItems
    .filter((item) => {
      if (item.sourceType !== "knowledge_base") {
        return false;
      }
      return !item.source.uri.startsWith("https://") || !item.source.locator;
    })
    .map((item) => item.evidenceId);
  const missingEvidenceReferences = parsed.claimEvidenceMap.flatMap((claim) =>
    claim.evidenceIds.filter((evidenceId) => !evidenceById.has(evidenceId)),
  );
  const sourceCompleteCount = parsed.evidenceItems.filter(
    (item) => item.source.uri && item.contentHash && item.syntheticDataNotice,
  ).length;
  const supportedClaims = parsed.claimEvidenceMap.filter(
    (claim) =>
      claim.supportStatus === "supported" && claim.evidenceIds.length > 0,
  ).length;
  const modelPredictionCount = parsed.evidenceItems.filter(
    (item) => item.sourceType === "model_prediction",
  ).length;
  const modelExplanationCount = parsed.evidenceItems.filter(
    (item) => item.sourceType === "model_explanation",
  ).length;
  const conflictWarnings = [
    modelPredictionCount > 1
      ? "Multiple model_prediction evidence items are present; verify they refer to the same synthetic analysis."
      : undefined,
    modelExplanationCount > 1
      ? "Multiple model_explanation evidence items are present; verify they refer to the same synthetic analysis."
      : undefined,
    duplicateContentHashes.length > 0
      ? "Duplicate evidence content hash detected; check for redundant citations."
      : undefined,
  ].filter((item): item is string => Boolean(item));
  const invalidSet = new Set([
    ...invalidEvidenceIds,
    ...duplicateEvidenceIds,
    ...missingEvidenceReferences,
  ]);

  return verifyEvidenceSourcesOutputSchema.parse({
    valid: invalidSet.size === 0,
    checkedEvidenceCount: parsed.evidenceItems.length,
    verifiedEvidenceIds: parsed.evidenceItems
      .map((item) => item.evidenceId)
      .filter((evidenceId) => !invalidSet.has(evidenceId)),
    invalidEvidenceIds: [...new Set(invalidEvidenceIds)].sort(),
    duplicateEvidenceIds,
    duplicateContentHashes,
    missingEvidenceReferences: [...new Set(missingEvidenceReferences)].sort(),
    conflictWarnings,
    sourceCompletenessRate:
      parsed.evidenceItems.length === 0
        ? 0
        : Number((sourceCompleteCount / parsed.evidenceItems.length).toFixed(4)),
    claimCoverageRate:
      parsed.claimEvidenceMap.length === 0
        ? 0
        : Number((supportedClaims / parsed.claimEvidenceMap.length).toFixed(4)),
    safetyNotice: SYNTHETIC_DATA_NOTICE,
  });
}

function claimReferencesInvalidEvidence(
  claim: HccClaimEvidence,
  invalidEvidenceIds: Set<string>,
): boolean {
  return claim.evidenceIds.some((evidenceId) => invalidEvidenceIds.has(evidenceId));
}

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values)].sort();
}

export function evaluateMedicalConfidenceGate(
  input: EvaluateMedicalConfidenceGateInput,
): EvaluateMedicalConfidenceGateOutput {
  const parsed = evaluateMedicalConfidenceGateInputSchema.parse(input);
  const reasons: string[] = [];
  const requiredActions: string[] = [];
  const evidenceGaps: string[] = [];
  const invalidEvidenceIds = new Set([
    ...parsed.sourceVerification.invalidEvidenceIds,
    ...parsed.sourceVerification.duplicateEvidenceIds,
    ...parsed.sourceVerification.missingEvidenceReferences,
  ]);
  const hasPrediction = parsed.evidenceItems.some(
    (item) => item.sourceType === "model_prediction",
  );
  const hasShap = parsed.evidenceItems.some(
    (item) => item.sourceType === "model_explanation",
  );
  const deniedClaimIds = new Set<string>();
  const limitedClaimIds = new Set<string>();
  const permittedClaimIds = new Set<string>();

  if (!parsed.disclaimerIncluded) {
    reasons.push("Required synthetic-data and non-clinical-use disclaimer is missing.");
    requiredActions.push("Attach the mandatory disclaimer before report release.");
  }
  if (!hasPrediction) {
    reasons.push("Missing deterministic Random Forest prediction evidence.");
    requiredActions.push("Call predictHccGrade before report generation.");
  }
  if (!hasShap) {
    reasons.push("Missing deterministic SHAP explanation evidence.");
    requiredActions.push("Call explainPredictionWithShap before report generation.");
  }

  for (const claim of parsed.claimEvidenceMap) {
    if (claim.supportStatus === "unsupported") {
      deniedClaimIds.add(claim.claimId);
      evidenceGaps.push(...claim.requiredFollowUp);
      continue;
    }
    if (claimReferencesInvalidEvidence(claim, invalidEvidenceIds)) {
      limitedClaimIds.add(claim.claimId);
      evidenceGaps.push(
        `Claim ${claim.claimId} references invalid or missing evidence.`,
      );
      continue;
    }
    if (claim.supportStatus === "partially_supported") {
      limitedClaimIds.add(claim.claimId);
      evidenceGaps.push(...claim.requiredFollowUp);
      continue;
    }
    permittedClaimIds.add(claim.claimId);
  }

  if (!parsed.sourceVerification.valid) {
    reasons.push("Source verification reported invalid evidence or missing references.");
    requiredActions.push("Remove invalid citations or re-run retrieval before strong conclusions.");
  }
  if (parsed.evidenceRetryExhausted) {
    reasons.push("Evidence retry limit has been exhausted.");
    requiredActions.push("Expose remaining evidence gaps in the report.");
  }
  if (evidenceGaps.length > 0) {
    reasons.push("One or more Claims are not fully supported by current Evidence.");
  }

  let status: z.infer<typeof gateStatusSchema> = "allow";
  if (!parsed.disclaimerIncluded || !hasPrediction || !hasShap) {
    status = "deny";
  } else if (permittedClaimIds.size === 0) {
    status = "interrupt";
    reasons.push("No supported Claims are available for report generation.");
    requiredActions.push("Collect at least one supported model or evidence Claim.");
  } else if (
    !parsed.sourceVerification.valid ||
    parsed.evidenceRetryExhausted ||
    deniedClaimIds.size > 0 ||
    limitedClaimIds.size > 0
  ) {
    status = "limited";
  }

  if (reasons.length === 0) {
    reasons.push("All required tool outputs, source checks, and evidence bindings passed.");
  }
  if (requiredActions.length === 0) {
    requiredActions.push("Generate report using only permitted Claim-Evidence bindings.");
  }
  const outputPermittedClaimIds =
    status === "allow" || status === "limited" ? [...permittedClaimIds] : [];
  if (status === "deny") {
    for (const claim of parsed.claimEvidenceMap) {
      deniedClaimIds.add(claim.claimId);
    }
  }

  return evaluateMedicalConfidenceGateOutputSchema.parse({
    gateId: `gate-${stableHash(JSON.stringify(parsed)).slice(0, 12)}`,
    status,
    riskLevel: "high",
    permittedClaimIds: uniqueSorted(outputPermittedClaimIds),
    limitedClaimIds: uniqueSorted([...limitedClaimIds]),
    deniedClaimIds: uniqueSorted([...deniedClaimIds]),
    reasons: uniqueSorted(reasons),
    evidenceGaps: uniqueSorted(evidenceGaps),
    requiredActions: uniqueSorted(requiredActions),
    requiresDisclaimer: true,
    safetyNotice: SYNTHETIC_DATA_NOTICE,
  });
}

function formatEvidenceCitation(item: HccEvidenceItem): string {
  const locator = item.source.locator ? `#${item.source.locator}` : "";
  return `${item.evidenceId} (${item.source.uri}${locator})`;
}

function evidenceLine(item: HccEvidenceItem): string {
  return `- [${item.evidenceId}] ${item.content} 来源：${item.source.uri}${item.source.locator ? `#${item.source.locator}` : ""}`;
}

function section(
  title: string,
  contentLines: string[],
  evidenceIds: string[],
): z.infer<typeof hccReportSectionSchema> {
  return hccReportSectionSchema.parse({
    title,
    content: contentLines.join("\n"),
    evidenceIds: uniqueSorted(evidenceIds),
  });
}

export function generateHccReport(
  input: GenerateHccReportInput,
): GenerateHccReportOutput {
  const parsed = generateHccReportInputSchema.parse(input);
  const evidenceById = new Map(
    parsed.evidenceItems.map((item) => [item.evidenceId, item]),
  );
  const permittedClaims = parsed.claimEvidenceMap.filter((claim) =>
    parsed.gateDecision.permittedClaimIds.includes(claim.claimId),
  );
  const omittedClaimIds = parsed.claimEvidenceMap
    .filter((claim) => !parsed.gateDecision.permittedClaimIds.includes(claim.claimId))
    .map((claim) => claim.claimId);
  const citedEvidenceIds = uniqueSorted(
    permittedClaims.flatMap((claim) => claim.evidenceIds),
  );
  const citedEvidenceItems = citedEvidenceIds
    .map((evidenceId) => evidenceById.get(evidenceId))
    .filter((item): item is HccEvidenceItem => Boolean(item));
  const modelEvidence = citedEvidenceItems.filter(
    (item) => item.sourceType === "model_prediction",
  );
  const shapEvidence = citedEvidenceItems.filter(
    (item) => item.sourceType === "model_explanation",
  );
  const kbEvidence = citedEvidenceItems.filter(
    (item) => item.sourceType === "knowledge_base",
  );

  const sections = [
    section(
      "Tool 与 Gate 摘要",
      [
        `patient_id: ${parsed.patientId ?? "未提供"}`,
        `gate_status: ${parsed.gateDecision.status}`,
        `risk_level: ${parsed.gateDecision.riskLevel}`,
        `gate_reasons: ${parsed.gateDecision.reasons.join("；")}`,
      ],
      [],
    ),
    section(
      "允许输出的 Claim",
      permittedClaims.length === 0
        ? ["无可输出 Claim；当前仅能返回受限安全说明。"]
        : permittedClaims.map(
            (claim) =>
              `- ${claim.claimId}: ${claim.claim} 证据：${claim.evidenceIds.join(", ")}`,
          ),
      permittedClaims.flatMap((claim) => claim.evidenceIds),
    ),
    section(
      "确定性模型证据",
      modelEvidence.length === 0
        ? ["未通过 Gate 的预测 Evidence，因此不输出预测概率或标签。"]
        : modelEvidence.map(evidenceLine),
      modelEvidence.map((item) => item.evidenceId),
    ),
    section(
      "SHAP 解释证据",
      shapEvidence.length === 0
        ? ["未通过 Gate 的 SHAP Evidence，因此不输出特征贡献解释。"]
        : shapEvidence.map(evidenceLine),
      shapEvidence.map((item) => item.evidenceId),
    ),
    section(
      "医学背景证据",
      kbEvidence.length === 0
        ? ["现有资料不足，无法给出医学背景解释。"]
        : kbEvidence.map(evidenceLine),
      kbEvidence.map((item) => item.evidenceId),
    ),
    section(
      "证据缺口与受限说明",
      [
        parsed.gateDecision.evidenceGaps.length === 0
          ? "未发现 Gate 要求披露的证据缺口。"
          : `evidence_gaps: ${parsed.gateDecision.evidenceGaps.join("；")}`,
        omittedClaimIds.length === 0
          ? "无被省略 Claim。"
          : `omitted_claim_ids: ${omittedClaimIds.join(", ")}`,
        `source_verification_valid: ${parsed.sourceVerification.valid}`,
        `source_completeness_rate: ${parsed.sourceVerification.sourceCompletenessRate}`,
        `claim_coverage_rate: ${parsed.sourceVerification.claimCoverageRate}`,
      ],
      [],
    ),
  ];

  const markdown = [
    "# 合成病例 DeepReason 可信分析报告",
    "",
    `安全声明：${SYNTHETIC_DATA_NOTICE}`,
    "",
    ...sections.flatMap((item) => [
      `## ${item.title}`,
      item.content,
      "",
    ]),
    "## 引用索引",
    citedEvidenceItems.length === 0
      ? "- 无可引用 Evidence。"
      : citedEvidenceItems
          .map((item) => `- ${formatEvidenceCitation(item)}`)
          .join("\n"),
    "",
    REPORT_DISCLAIMER,
  ].join("\n");

  return generateHccReportOutputSchema.parse({
    reportId: `report-${stableHash(markdown).slice(0, 12)}`,
    status: parsed.gateDecision.status,
    markdown,
    sections,
    citedEvidenceIds,
    omittedClaimIds: uniqueSorted(omittedClaimIds),
    generationRules: [
      "Only permitted Claim-Evidence bindings were included.",
      "Prediction and SHAP statements must come from model Evidence items.",
      "Medical background statements must cite verified knowledge-base Evidence items.",
      "Unsupported Claims are emitted only as evidence gaps.",
    ],
    safetyNotice: SYNTHETIC_DATA_NOTICE,
    disclaimer: REPORT_DISCLAIMER,
  });
}

function defaultRetrievalSnapshot(
  evidenceItems: HccEvidenceItem[],
): z.infer<typeof memoryRetrievalSnapshotSchema> | undefined {
  const knowledgeEvidenceIds = evidenceItems
    .filter((item) => item.sourceType === "knowledge_base")
    .map((item) => item.evidenceId);
  if (knowledgeEvidenceIds.length === 0) {
    return undefined;
  }
  return {
    confidence: "high",
    evidenceSufficient: true,
    evidenceIds: knowledgeEvidenceIds,
  };
}

function memoryProposalBlockers(
  parsed: z.output<typeof createMemoryProposalInputSchema>,
): string[] {
  const blockers: string[] = [];
  if (!parsed.patientId) {
    blockers.push("patient_id is required before writing case memory.");
  }
  if (!["allow", "limited"].includes(parsed.gateDecision.status)) {
    blockers.push(
      `gate status ${parsed.gateDecision.status} does not permit memory write proposal application.`,
    );
  }
  if (!parsed.sourceVerification.valid) {
    blockers.push("source verification must be valid before memory write.");
  }
  if (parsed.evidenceItems.length === 0) {
    blockers.push("at least one Evidence item is required before memory write.");
  }
  if (
    parsed.evidenceItems.some(
      (item) => item.syntheticDataNotice !== SYNTHETIC_DATA_NOTICE,
    )
  ) {
    blockers.push("all Evidence items must preserve the synthetic-data notice.");
  }
  if (
    !parsed.evidenceItems.some((item) => item.sourceType === "model_prediction")
  ) {
    blockers.push("model prediction Evidence is required before memory write.");
  }
  if (parsed.gateDecision.permittedClaimIds.length === 0) {
    blockers.push("at least one Gate-permitted Claim is required before memory write.");
  }
  if (parsed.sourceVerification.verifiedEvidenceIds.length === 0) {
    blockers.push("at least one verified Evidence ID is required before memory write.");
  }
  return uniqueSorted(blockers);
}

export function createMemoryProposal(
  input: CreateMemoryProposalInput,
): CreateMemoryProposalOutput {
  const parsed = createMemoryProposalInputSchema.parse(input);
  const history = getPatientHistory({
    patientId: parsed.patientId,
    memoryDir: parsed.memoryDir,
  });
  const evidenceIds = uniqueSorted(parsed.sourceVerification.verifiedEvidenceIds);
  const blockedReasons = memoryProposalBlockers(parsed);
  const canApply = blockedReasons.length === 0;
  const reasons = [
    "Memory proposal created for synthetic case memory only.",
    "Direct long-term memory write is disabled; approved_by is required for apply.",
    parsed.approvedBy
      ? `Approval supplied by ${parsed.approvedBy}.`
      : "No approval supplied; proposal remains pending and no file write occurs.",
    ...parsed.gateDecision.reasons.map((reason) => `Gate: ${reason}`),
  ];
  const proposalId = `mem-proposal-${stableHash(
    JSON.stringify({
      patientId: parsed.patientId,
      sessionId: parsed.sessionId,
      evidenceIds,
      gateId: parsed.gateDecision.gateId,
      approvedBy: parsed.approvedBy ?? null,
    }),
  ).slice(0, 12)}`;

  if (!parsed.approvedBy || !canApply) {
    return createMemoryProposalOutputSchema.parse({
      proposalId,
      status: canApply ? "pending_approval" : "rejected",
      targetPartition: "case_memory",
      patientId: parsed.patientId,
      sessionId: parsed.sessionId,
      requiresApproval: true,
      approvedBy: parsed.approvedBy,
      canApply,
      applied: false,
      directWrite: false,
      recordCountBefore: history.recordCount,
      recordCountAfter: history.recordCount,
      evidenceIds,
      gateStatus: parsed.gateDecision.status,
      reasons: uniqueSorted(reasons),
      blockedReasons,
      safetyNotice: SYNTHETIC_DATA_NOTICE,
    });
  }

  const appliedResult = saveCaseMemory({
    patientId: parsed.patientId,
    sessionId: parsed.sessionId,
    features: parsed.features,
    prediction: parsed.prediction,
    shap: parsed.shap,
    retrieval:
      parsed.retrieval ?? defaultRetrievalSnapshot(parsed.evidenceItems),
    memoryDir: parsed.memoryDir,
  });

  return createMemoryProposalOutputSchema.parse({
    proposalId,
    status: "approved_applied",
    targetPartition: "case_memory",
    patientId: parsed.patientId,
    sessionId: parsed.sessionId,
    requiresApproval: true,
    approvedBy: parsed.approvedBy,
    canApply,
    applied: true,
    directWrite: false,
    recordCountBefore: history.recordCount,
    recordCountAfter: appliedResult.recordCount,
    evidenceIds,
    gateStatus: parsed.gateDecision.status,
    reasons: uniqueSorted([
      ...reasons,
      "Approved proposal applied through controlled memory writer.",
    ]),
    blockedReasons,
    appliedResult,
    safetyNotice: SYNTHETIC_DATA_NOTICE,
  });
}

export function hccDeepReasonEvidenceHandlerSpecs() {
  return Object.values(evidenceHandlerSpecs);
}

export function createHccDeepReasonEvidenceTools() {
  return {
    checkClaims: async (input: CheckClaimsInput) =>
      withTrace(evidenceHandlerSpecs.checkClaims, async () =>
        checkClaims(input),
      ),
    buildFollowUpQuery: async (input: BuildFollowUpQueryInput) =>
      withTrace(evidenceHandlerSpecs.buildFollowUpQuery, async () =>
        buildFollowUpQuery(input),
      ),
    verifyEvidenceSources: async (input: VerifyEvidenceSourcesInput) =>
      withTrace(evidenceHandlerSpecs.verifyEvidenceSources, async () =>
        verifyEvidenceSources(input),
      ),
    evaluateMedicalConfidenceGate: async (
      input: EvaluateMedicalConfidenceGateInput,
    ) =>
      withTrace(evidenceHandlerSpecs.evaluateMedicalConfidenceGate, async () =>
        evaluateMedicalConfidenceGate(input),
      ),
    generateHccReport: async (input: GenerateHccReportInput) =>
      withTrace(evidenceHandlerSpecs.generateHccReport, async () =>
        generateHccReport(input),
      ),
    createMemoryProposal: async (input: CreateMemoryProposalInput) =>
      withTrace(evidenceHandlerSpecs.createMemoryProposal, async () =>
        createMemoryProposal(input),
      ),
  };
}

export type HccDeepReasonEvidenceTools = ReturnType<
  typeof createHccDeepReasonEvidenceTools
>;
