import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import {
  runHccDeepReasonWorkflow,
  type HccRequiredClaim,
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

let server: Server;
let baseUrl = "";

function predictionPayload(body: { patient_id?: string; features: HccFeatures }) {
  return {
    safety_notice: "演示用合成数据，非真实患者数据；非临床诊断依据。",
    prediction: {
      label: "synthetic_high_pathology_grade",
      probability_high_grade: 0.91,
      probability_low_or_intermediate: 0.09,
      uncertain_probability_band: false,
    },
    features_used: Object.keys(body.features),
    input_echo: body.features,
    model: {
      type: "RandomForestClassifier",
      feature_selection_method: "simulated_fci_fixed_causal_candidates",
      cv_auc_mean: 0.828,
      cv_auc_std: 0.017,
    },
    disclaimer:
      "该输出仅用于科研学习与工程演示，不作为任何临床诊断、治疗或分级依据。",
    patient_id: body.patient_id,
  };
}

function explanationPayload(body: { patient_id?: string; features: HccFeatures }) {
  return {
    safety_notice: "演示用合成数据，非真实患者数据；非临床诊断依据。",
    prediction: predictionPayload(body).prediction,
    shap: {
      method: "shap.TreeExplainer",
      target_class: "synthetic_high_pathology_grade",
      top_n: 5,
      base_value: 0.51,
      top_features: [
        {
          feature: "portal_vein_invasion",
          value: body.features.portal_vein_invasion,
          shap_value: 0.19,
          abs_shap_value: 0.19,
          direction: "pushes_toward_high_grade",
          trust_level: "high_trust_causal_candidate",
          consistency_note:
            "该特征同时位于 SHAP Top-N 与模拟 FCI 因果候选集合中。",
        },
        {
          feature: "afp_ng_ml",
          value: body.features.afp_ng_ml,
          shap_value: 0.08,
          abs_shap_value: 0.08,
          direction: "pushes_toward_high_grade",
          trust_level: "high_trust_causal_candidate",
          consistency_note:
            "该特征同时位于 SHAP Top-N 与模拟 FCI 因果候选集合中。",
        },
      ],
      causal_candidate_features: Object.keys(body.features),
      high_trust_features: ["portal_vein_invasion", "afp_ng_ml"],
      statistical_only_features: [],
      consistency_summary:
        "SHAP Top-N 与模拟 FCI 因果候选集合存在交集；交集特征标记为高可信解释线索。",
      caveat:
        "SHAP 解释反映模型内部贡献，不等同于真实医学因果效应；本原型仅用于工程演示。",
    },
    disclaimer:
      "该输出仅用于科研学习与工程演示，不作为任何临床诊断、治疗或分级依据。",
    patient_id: body.patient_id,
  };
}

before(async () => {
  server = createServer((req, res) => {
    if (req.method !== "POST" || (req.url !== "/predict" && req.url !== "/explain")) {
      res.writeHead(404).end();
      return;
    }

    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      const body = JSON.parse(Buffer.concat(chunks).toString("utf-8")) as {
        patient_id?: string;
        features: HccFeatures;
      };
      const payload =
        req.url === "/predict" ? predictionPayload(body) : explanationPayload(body);

      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(payload));
    });
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve());
  });

  const address = server.address();
  assert(address && typeof address === "object");
  baseUrl = `http://127.0.0.1:${address.port}`;
});

after(async () => {
  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
});

function tempMemoryDir() {
  return mkdtempSync(join(tmpdir(), "hcc-deepreason-runtime-"));
}

function workflowNodes(result: Awaited<ReturnType<typeof runHccDeepReasonWorkflow>>) {
  return result.workflow_trace.map((item) => item.nodeId);
}

test("DeepReason runtime executes complete HCC workflow and keeps legacy analysis fields", async () => {
  const result = await runHccDeepReasonWorkflow({
    sessionId: "runtime-session",
    patientId: "runtime-patient",
    features: demoFeatures,
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
    memoryDir: tempMemoryDir(),
  });

  assert.equal(result.finishReason, "stop");
  assert.equal(
    result.analysis.prediction?.prediction.label,
    "synthetic_high_pathology_grade",
  );
  assert.equal(
    result.analysis.explanation?.shap.top_features[0]?.feature,
    "portal_vein_invasion",
  );
  assert.ok(result.analysis.evidence?.results.length);
  assert.equal(result.analysis.memory?.saved, false);
  assert.equal(result.memory_proposal?.status, "pending_approval");
  assert.equal(result.deepreason.gateDecision?.status, "allow");
  assert.ok(result.claim_evidence_map.length > 0);
  assert.ok(result.evidence_items.some((item) => item.sourceType === "model_prediction"));
  assert.match(result.text, /免责声明/);
  assert.deepEqual(
    workflowNodes(result).filter((node) =>
      [
        "feature_check",
        "history_retrieve",
        "prediction",
        "shap_explain",
        "evidence_retrieve",
        "claim_check",
        "source_verify",
        "confidence_gate",
        "report_generate",
        "verify",
        "memory_proposal",
        "respond",
      ].includes(node),
    ),
    [
      "feature_check",
      "history_retrieve",
      "prediction",
      "shap_explain",
      "evidence_retrieve",
      "claim_check",
      "source_verify",
      "confidence_gate",
      "report_generate",
      "verify",
      "memory_proposal",
      "respond",
    ],
  );
  assert.ok(
    result.toolCalls.some((call) => call.toolName === "createMemoryProposal"),
  );
  assert.ok(
    !result.toolCalls.some((call) => call.toolName === "saveCaseMemory"),
  );
});

test("DeepReason runtime stops after feature_check when synthetic features are missing", async () => {
  const result = await runHccDeepReasonWorkflow({
    sessionId: "runtime-missing-session",
    patientId: "runtime-missing-patient",
    features: {
      tumor_size_cm: 6.2,
      afp_ng_ml: 420,
    },
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
    memoryDir: tempMemoryDir(),
  });

  assert.equal(result.finishReason, "clarification_required");
  assert.equal(result.analysis.completeness?.complete, false);
  assert.ok(result.analysis.completeness?.missingFeatures.includes("alt_u_l"));
  assert.deepEqual(
    result.toolCalls.map((call) => call.toolName),
    ["checkFeatureCompleteness"],
  );
  assert.ok(!workflowNodes(result).includes("prediction"));
  assert.match(result.text, /缺少以下合成特征/);
});

test("DeepReason runtime performs bounded evidence retry and limits unsupported claims", async () => {
  const requiredClaims: HccRequiredClaim[] = [
    {
      claimId: "claim-model-output",
      claim: "Prediction output is available from the RF tool.",
      claimType: "model_output",
      requiredSourceTypes: ["model_prediction"],
      requiredTerms: [],
      minEvidence: 1,
      confidenceThreshold: 1,
    },
    {
      claimId: "claim-shap-explanation",
      claim: "SHAP explanation output is available from the SHAP tool.",
      claimType: "model_explanation",
      requiredSourceTypes: ["model_explanation"],
      requiredTerms: [],
      minEvidence: 1,
      confidenceThreshold: 1,
    },
    {
      claimId: "claim-impossible-background",
      claim: "A nonexistent prototype-only biomarker has public HCC background evidence.",
      claimType: "medical_background",
      requiredSourceTypes: ["knowledge_base"],
      requiredTerms: ["zzzx_nonexistent_hcc_marker"],
      minEvidence: 1,
      confidenceThreshold: 0.25,
    },
  ];

  const result = await runHccDeepReasonWorkflow({
    sessionId: "runtime-retry-session",
    patientId: "runtime-retry-patient",
    features: demoFeatures,
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
    memoryDir: tempMemoryDir(),
    requiredClaims,
    maxEvidenceRetry: 2,
  });

  assert.equal(result.retry_count, 2);
  assert.equal(result.deepreason.gateDecision?.status, "limited");
  assert.ok(
    result.claim_evidence_map.some(
      (claim) =>
        claim.claimId === "claim-impossible-background" &&
        claim.supportStatus === "unsupported",
    ),
  );
  assert.ok(
    result.deepreason.gateDecision?.deniedClaimIds.includes(
      "claim-impossible-background",
    ),
  );
  assert.ok(
    result.toolCalls.filter((call) => call.toolName === "buildFollowUpQuery")
      .length >= 2,
  );
  assert.match(result.text, /免责声明/);
});
