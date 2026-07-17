import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildFollowUpQuery,
  checkClaims,
  createHccDeepReasonEvidenceTools,
  verifyEvidenceSources,
  type HccEvidenceDraft,
} from "../src/deepreason/index.js";

const evidenceDrafts: HccEvidenceDraft[] = [
  {
    evidenceId: "ev-prediction-demo",
    sourceType: "model_prediction",
    claimType: "model_output",
    summary:
      "RF prediction synthetic_high_pathology_grade with high-grade probability 0.91.",
    confidence: 1,
    uri: "ml-service:/predict",
    locator: "prediction",
  },
  {
    evidenceId: "ev-shap-demo",
    sourceType: "model_explanation",
    claimType: "model_explanation",
    summary:
      "SHAP TreeExplainer returned portal_vein_invasion and AFP as top features.",
    confidence: 1,
    uri: "ml-service:/explain",
    locator: "shap.top_features",
  },
  {
    evidenceId: "ev-kb-afp",
    sourceType: "knowledge_base",
    claimType: "medical_background",
    summary:
      "AFP biomarker background: AFP is a commonly discussed HCC-related tumor marker in public medical education material.",
    confidence: 0.82,
    uri: "https://www.cancer.gov/publications/dictionaries/cancer-terms",
    locator: "KB-HCC-001",
  },
  {
    evidenceId: "ev-kb-pvi",
    sourceType: "knowledge_base",
    claimType: "medical_background",
    summary:
      "Portal vein invasion background: vascular invasion is relevant contextual information for HCC staging discussions.",
    confidence: 0.78,
    uri: "https://www.cancer.gov/types/liver",
    locator: "KB-HCC-002",
  },
];

test("checkClaims converts drafts into EvidenceItems and binds supported and unsupported claims", () => {
  const result = checkClaims({
    evidenceDrafts,
    requiredClaims: [
      {
        claimId: "claim-model",
        claim: "The prediction probability must come from the deterministic RF tool.",
        claimType: "model_output",
        requiredSourceTypes: ["model_prediction"],
        minEvidence: 1,
        confidenceThreshold: 1,
      },
      {
        claimId: "claim-afp",
        claim: "AFP background information is available from a traceable public source.",
        claimType: "medical_background",
        requiredSourceTypes: ["knowledge_base"],
        requiredTerms: ["AFP"],
        minEvidence: 1,
        confidenceThreshold: 0.25,
      },
      {
        claimId: "claim-missing",
        claim:
          "Microvascular invasion background information is available from the current evidence set.",
        claimType: "medical_background",
        requiredSourceTypes: ["knowledge_base"],
        requiredTerms: ["microvascular invasion"],
        minEvidence: 1,
        confidenceThreshold: 0.25,
      },
    ],
  });

  assert.equal(result.evidenceItems.length, 4);
  assert.ok(result.evidenceItems.every((item) => item.contentHash.length === 64));
  assert.equal(result.claimEvidenceMap.length, 3);
  assert.equal(result.supportSummary.supported, 2);
  assert.equal(result.supportSummary.unsupported, 1);
  assert.deepEqual(
    result.unsupportedClaims.map((claim) => claim.claimId),
    ["claim-missing"],
  );
  assert.ok(result.evidenceGaps[0]?.includes("microvascular invasion"));
});

test("buildFollowUpQuery creates a bounded focused query from unsupported claims", () => {
  const claims = checkClaims({
    evidenceDrafts,
    requiredClaims: [
      {
        claimId: "claim-missing",
        claim: "Microvascular invasion background is available.",
        claimType: "medical_background",
        requiredSourceTypes: ["knowledge_base"],
        requiredTerms: ["microvascular invasion"],
      },
    ],
  }).unsupportedClaims;

  const result = buildFollowUpQuery({
    unsupportedClaims: claims,
    evidenceRetryCount: 0,
    maxRetry: 2,
    baseQuery: "HCC pathology grade",
  });

  assert.equal(result.shouldRetry, true);
  assert.equal(result.evidenceRetryCount, 1);
  assert.deepEqual(result.unsupportedClaimIds, ["claim-missing"]);
  assert.ok(result.followUpQuery.includes("microvascular"));

  const exhausted = buildFollowUpQuery({
    unsupportedClaims: claims,
    evidenceRetryCount: 2,
    maxRetry: 2,
  });
  assert.equal(exhausted.shouldRetry, false);
  assert.equal(exhausted.followUpQuery, "");
  assert.ok(exhausted.reason.includes("retry limit"));
});

test("verifyEvidenceSources validates traceable sources and claim references", () => {
  const checked = checkClaims({ evidenceDrafts });
  const verification = verifyEvidenceSources({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
  });

  assert.equal(verification.valid, true);
  assert.equal(verification.checkedEvidenceCount, 4);
  assert.equal(verification.invalidEvidenceIds.length, 0);
  assert.equal(verification.missingEvidenceReferences.length, 0);
  assert.equal(verification.sourceCompletenessRate, 1);
  assert.ok(verification.claimCoverageRate > 0.5);
});

test("verifyEvidenceSources rejects untraceable KB sources and missing evidence references", () => {
  const checked = checkClaims({ evidenceDrafts });
  const tampered = structuredClone(checked.evidenceItems);
  const kb = tampered.find((item) => item.evidenceId === "ev-kb-afp");
  assert.ok(kb);
  kb.source.uri = "memory:/untraceable";
  delete kb.source.locator;
  const claimEvidenceMap = structuredClone(checked.claimEvidenceMap);
  claimEvidenceMap[0]?.evidenceIds.push("ev-does-not-exist");

  const verification = verifyEvidenceSources({
    evidenceItems: tampered,
    claimEvidenceMap,
  });

  assert.equal(verification.valid, false);
  assert.deepEqual(verification.invalidEvidenceIds, ["ev-kb-afp"]);
  assert.deepEqual(verification.missingEvidenceReferences, ["ev-does-not-exist"]);
});

test("Evidence tool wrappers expose DeepReason node, agent, and trace metadata", async () => {
  const tools = createHccDeepReasonEvidenceTools();
  const checked = await tools.checkClaims({ evidenceDrafts });
  assert.equal(checked.handlerName, "checkClaims");
  assert.equal(checked.nodeId, "claim_check");
  assert.equal(checked.agentName, "claim_checker");
  assert.ok(checked.output.evidenceItems.length > 0);
  assert.ok(checked.trace.durationMs >= 0);

  const followUp = await tools.buildFollowUpQuery({
    unsupportedClaims: checked.output.unsupportedClaims,
  });
  assert.equal(followUp.nodeId, "build_follow_up_query");
  assert.equal(followUp.agentName, "medical_retriever");

  const source = await tools.verifyEvidenceSources({
    evidenceItems: checked.output.evidenceItems,
    claimEvidenceMap: checked.output.claimEvidenceMap,
  });
  assert.equal(source.nodeId, "source_verify");
  assert.equal(source.agentName, "source_verifier");
});
