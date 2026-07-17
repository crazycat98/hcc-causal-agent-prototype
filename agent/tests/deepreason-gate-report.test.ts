import assert from "node:assert/strict";
import { test } from "node:test";
import { REPORT_DISCLAIMER } from "../src/safety.js";
import {
  checkClaims,
  createHccDeepReasonEvidenceTools,
  evaluateMedicalConfidenceGate,
  generateHccReport,
  verifyEvidenceSources,
  type HccEvidenceDraft,
} from "../src/deepreason/index.js";

const completeDrafts: HccEvidenceDraft[] = [
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
      "AFP biomarker background: AFP is discussed as HCC-related public medical background information.",
    confidence: 0.82,
    uri: "https://www.cancer.gov/publications/dictionaries/cancer-terms",
    locator: "KB-HCC-001",
  },
];

function checkedComplete() {
  const checked = checkClaims({ evidenceDrafts: completeDrafts });
  const sourceVerification = verifyEvidenceSources({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
  });
  return { checked, sourceVerification };
}

test("evaluateMedicalConfidenceGate allows fully supported tool and source evidence", () => {
  const { checked, sourceVerification } = checkedComplete();
  const gate = evaluateMedicalConfidenceGate({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
    sourceVerification,
    disclaimerIncluded: true,
  });

  assert.equal(gate.status, "allow");
  assert.equal(gate.riskLevel, "high");
  assert.deepEqual(gate.deniedClaimIds, []);
  assert.ok(gate.permittedClaimIds.includes("claim-model-output"));
  assert.ok(gate.permittedClaimIds.includes("claim-shap-explanation"));
  assert.ok(gate.permittedClaimIds.includes("claim-medical-background"));
  assert.equal(gate.evidenceGaps.length, 0);
});

test("evaluateMedicalConfidenceGate downgrades to limited when a claim is unsupported", () => {
  const checked = checkClaims({
    evidenceDrafts: completeDrafts,
    requiredClaims: [
      {
        claimId: "claim-model",
        claim: "Prediction output comes from the RF tool.",
        claimType: "model_output",
        requiredSourceTypes: ["model_prediction"],
      },
      {
        claimId: "claim-missing",
        claim: "Microvascular invasion background is available.",
        claimType: "medical_background",
        requiredSourceTypes: ["knowledge_base"],
        requiredTerms: ["microvascular invasion"],
      },
    ],
  });
  const sourceVerification = verifyEvidenceSources({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
  });

  const gate = evaluateMedicalConfidenceGate({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
    sourceVerification,
    evidenceRetryExhausted: true,
  });

  assert.equal(gate.status, "limited");
  assert.deepEqual(gate.permittedClaimIds, ["claim-model"]);
  assert.deepEqual(gate.deniedClaimIds, ["claim-missing"]);
  assert.ok(gate.evidenceGaps.some((gap) => gap.includes("microvascular")));
  assert.ok(gate.reasons.some((reason) => reason.includes("retry limit")));
});

test("evaluateMedicalConfidenceGate denies report release without prediction evidence or disclaimer", () => {
  const checked = checkClaims({
    evidenceDrafts: completeDrafts.filter(
      (item) => item.sourceType !== "model_prediction",
    ),
  });
  const sourceVerification = verifyEvidenceSources({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
  });

  const gate = evaluateMedicalConfidenceGate({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
    sourceVerification,
    disclaimerIncluded: false,
  });

  assert.equal(gate.status, "deny");
  assert.deepEqual(gate.permittedClaimIds, []);
  assert.ok(
    gate.requiredActions.includes("Call predictHccGrade before report generation."),
  );
  assert.ok(
    gate.requiredActions.includes(
      "Attach the mandatory disclaimer before report release.",
    ),
  );
});

test("generateHccReport includes only Gate-permitted claims and verified Evidence IDs", () => {
  const checked = checkClaims({
    evidenceDrafts: completeDrafts,
    requiredClaims: [
      {
        claimId: "claim-model",
        claim: "Prediction output comes from the RF tool.",
        claimType: "model_output",
        requiredSourceTypes: ["model_prediction"],
      },
      {
        claimId: "claim-missing",
        claim: "Microvascular invasion background is available.",
        claimType: "medical_background",
        requiredSourceTypes: ["knowledge_base"],
        requiredTerms: ["microvascular invasion"],
      },
    ],
  });
  const sourceVerification = verifyEvidenceSources({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
  });
  const gate = evaluateMedicalConfidenceGate({
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
    sourceVerification,
    evidenceRetryExhausted: true,
  });

  const report = generateHccReport({
    gateDecision: gate,
    evidenceItems: checked.evidenceItems,
    claimEvidenceMap: checked.claimEvidenceMap,
    sourceVerification,
    patientId: "synthetic-patient-001",
  });

  assert.equal(report.status, "limited");
  assert.ok(report.markdown.includes("synthetic-patient-001"));
  assert.ok(report.markdown.includes("claim-model"));
  assert.ok(!report.markdown.includes("claim-missing: Microvascular"));
  assert.ok(report.omittedClaimIds.includes("claim-missing"));
  assert.deepEqual(report.citedEvidenceIds, ["ev-prediction-demo"]);
  assert.equal(report.disclaimer, REPORT_DISCLAIMER);
  assert.ok(report.markdown.includes("免责声明"));
});

test("Gate and report wrappers expose DeepReason metadata", async () => {
  const tools = createHccDeepReasonEvidenceTools();
  const checked = await tools.checkClaims({ evidenceDrafts: completeDrafts });
  const source = await tools.verifyEvidenceSources({
    evidenceItems: checked.output.evidenceItems,
    claimEvidenceMap: checked.output.claimEvidenceMap,
  });
  const gate = await tools.evaluateMedicalConfidenceGate({
    evidenceItems: checked.output.evidenceItems,
    claimEvidenceMap: checked.output.claimEvidenceMap,
    sourceVerification: source.output,
  });
  assert.equal(gate.nodeId, "confidence_gate");
  assert.equal(gate.agentName, "medical_safety_reviewer");

  const report = await tools.generateHccReport({
    gateDecision: gate.output,
    evidenceItems: checked.output.evidenceItems,
    claimEvidenceMap: checked.output.claimEvidenceMap,
    sourceVerification: source.output,
  });
  assert.equal(report.nodeId, "report_generate");
  assert.equal(report.agentName, "report_writer");
  assert.ok(report.output.markdown.includes("合成病例 DeepReason 可信分析报告"));
});
