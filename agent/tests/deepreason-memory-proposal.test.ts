import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { getPatientHistory } from "../../memory/src/store.js";
import {
  checkClaims,
  createHccDeepReasonEvidenceTools,
  createMemoryProposal,
  evaluateMedicalConfidenceGate,
  verifyEvidenceSources,
  type HccEvidenceDraft,
} from "../src/deepreason/index.js";
import type { HccFeatures } from "../src/features.js";

const demoFeatures: HccFeatures = {
  tumor_size_cm: 6.2,
  afp_ng_ml: 420,
  alt_u_l: 61,
  ast_u_l: 72,
  bilirubin_umol_l: 24,
  albumin_g_l: 36,
  platelet_10e9_l: 128,
  portal_vein_invasion: 1,
  radiomics_entropy: 5.4,
  radiomics_glcm_contrast: 112,
};

const prediction = {
  label: "synthetic_high_pathology_grade" as const,
  probability_high_grade: 0.91,
  probability_low_or_intermediate: 0.09,
  uncertain_probability_band: false,
};

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
    summary: "SHAP TreeExplainer returned portal_vein_invasion as a top feature.",
    confidence: 1,
    uri: "ml-service:/explain",
    locator: "shap.top_features",
  },
  {
    evidenceId: "ev-kb-afp",
    sourceType: "knowledge_base",
    claimType: "medical_background",
    summary: "AFP is discussed as HCC-related public medical background.",
    confidence: 0.82,
    uri: "https://www.cancer.gov/publications/dictionaries/cancer-terms",
    locator: "KB-HCC-001",
  },
];

function tempMemoryDir() {
  return mkdtempSync(join(tmpdir(), "hcc-memory-proposal-"));
}

function approvedGateBundle() {
  const checked = checkClaims({ evidenceDrafts });
  const sourceVerification = verifyEvidenceSources({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
  });
  const gateDecision = evaluateMedicalConfidenceGate({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
    sourceVerification,
  });
  assert.equal(gateDecision.status, "allow");
  return { checked, sourceVerification, gateDecision };
}

test("createMemoryProposal without approval creates a pending proposal and does not write case memory", () => {
  const memoryDir = tempMemoryDir();
  const { checked, sourceVerification, gateDecision } = approvedGateBundle();

  const proposal = createMemoryProposal({
    patientId: "synthetic-memory-001",
    sessionId: "session-memory-001",
    features: demoFeatures,
    prediction,
    gateDecision,
    evidenceItems: checked.evidenceItems,
    sourceVerification,
    memoryDir,
  });

  assert.equal(proposal.status, "pending_approval");
  assert.equal(proposal.applied, false);
  assert.equal(proposal.directWrite, false);
  assert.equal(proposal.canApply, true);
  assert.equal(proposal.recordCountBefore, 0);
  assert.equal(proposal.recordCountAfter, 0);

  const history = getPatientHistory({
    patientId: "synthetic-memory-001",
    memoryDir,
  });
  assert.equal(history.hasHistory, false);
  assert.equal(history.recordCount, 0);
});

test("createMemoryProposal with approval applies the existing memory writer once", () => {
  const memoryDir = tempMemoryDir();
  const { checked, sourceVerification, gateDecision } = approvedGateBundle();

  const proposal = createMemoryProposal({
    patientId: "synthetic-memory-002",
    sessionId: "session-memory-002",
    features: demoFeatures,
    prediction,
    gateDecision,
    evidenceItems: checked.evidenceItems,
    sourceVerification,
    approvedBy: "reviewer@example.test",
    memoryDir,
  });

  assert.equal(proposal.status, "approved_applied");
  assert.equal(proposal.applied, true);
  assert.equal(proposal.approvedBy, "reviewer@example.test");
  assert.equal(proposal.recordCountBefore, 0);
  assert.equal(proposal.recordCountAfter, 1);
  assert.equal(proposal.appliedResult?.saved, true);

  const history = getPatientHistory({
    patientId: "synthetic-memory-002",
    memoryDir,
  });
  assert.equal(history.hasHistory, true);
  assert.equal(history.recordCount, 1);
});

test("createMemoryProposal rejects approved writes without patient_id", () => {
  const memoryDir = tempMemoryDir();
  const { checked, sourceVerification, gateDecision } = approvedGateBundle();

  const proposal = createMemoryProposal({
    sessionId: "session-memory-missing-patient",
    features: demoFeatures,
    prediction,
    gateDecision,
    evidenceItems: checked.evidenceItems,
    sourceVerification,
    approvedBy: "reviewer@example.test",
    memoryDir,
  });

  assert.equal(proposal.status, "rejected");
  assert.equal(proposal.applied, false);
  assert.equal(proposal.canApply, false);
  assert.ok(
    proposal.blockedReasons.includes(
      "patient_id is required before writing case memory.",
    ),
  );
});

test("createMemoryProposal rejects approved writes when Gate denies release", () => {
  const memoryDir = tempMemoryDir();
  const checked = checkClaims({
    evidenceDrafts: evidenceDrafts.filter(
      (item) => item.sourceType !== "model_prediction",
    ),
  });
  const sourceVerification = verifyEvidenceSources({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
  });
  const gateDecision = evaluateMedicalConfidenceGate({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
    sourceVerification,
    disclaimerIncluded: false,
  });
  assert.equal(gateDecision.status, "deny");

  const proposal = createMemoryProposal({
    patientId: "synthetic-memory-denied",
    sessionId: "session-memory-denied",
    features: demoFeatures,
    prediction,
    gateDecision,
    evidenceItems: checked.evidenceItems,
    sourceVerification,
    approvedBy: "reviewer@example.test",
    memoryDir,
  });

  assert.equal(proposal.status, "rejected");
  assert.equal(proposal.applied, false);
  assert.ok(
    proposal.blockedReasons.some((reason) =>
      reason.includes("does not permit memory write"),
    ),
  );
  assert.equal(
    getPatientHistory({ patientId: "synthetic-memory-denied", memoryDir })
      .recordCount,
    0,
  );
});

test("Memory proposal wrapper exposes DeepReason metadata", async () => {
  const memoryDir = tempMemoryDir();
  const tools = createHccDeepReasonEvidenceTools();
  const checked = await tools.checkClaims({ evidenceDrafts });
  const source = await tools.verifyEvidenceSources({
    evidenceItems: checked.output.evidenceItems,
    claimEvidenceMap: checked.output.claimEvidenceMap,
  });
  const gate = await tools.evaluateMedicalConfidenceGate({
    evidenceItems: checked.output.evidenceItems,
    claimEvidenceMap: checked.output.claimEvidenceMap,
    sourceVerification: source.output,
  });

  const proposal = await tools.createMemoryProposal({
    patientId: "synthetic-memory-wrapper",
    sessionId: "session-memory-wrapper",
    features: demoFeatures,
    prediction,
    gateDecision: gate.output,
    evidenceItems: checked.output.evidenceItems,
    sourceVerification: source.output,
    memoryDir,
  });

  assert.equal(proposal.handlerName, "createMemoryProposal");
  assert.equal(proposal.nodeId, "memory_proposal");
  assert.equal(proposal.agentName, "memory_manager");
  assert.equal(proposal.output.status, "pending_approval");
});
