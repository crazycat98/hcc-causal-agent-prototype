import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import type { HccFeatures } from "../../agent/src/features.js";
import { POST } from "../app/api/analyze/route.js";

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
      ],
      causal_candidate_features: Object.keys(body.features),
      high_trust_features: ["portal_vein_invasion"],
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
      const body = JSON.parse(Buffer.concat(chunks).toString("utf-8"));
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
  process.env.ML_PREDICTION_URL = `${baseUrl}/predict`;
  process.env.ML_EXPLANATION_URL = `${baseUrl}/explain`;
  process.env.AGENT_MEMORY_DIR = mkdtempSync(join(tmpdir(), "hcc-web-memory-"));
});

after(async () => {
  delete process.env.ML_PREDICTION_URL;
  delete process.env.ML_EXPLANATION_URL;
  delete process.env.AGENT_MEMORY_DIR;

  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
});

test("M8 API route returns DeepReason workflow report and legacy-compatible outputs", async () => {
  const response = await POST(
    new Request("http://127.0.0.1:3000/api/analyze", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        sessionId: "web-test-session",
        patientId: "web-test-patient",
        features: demoFeatures,
      }),
    }),
  );
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.equal(
    payload.safetyNotice,
    "演示用合成数据，非真实患者数据；非临床诊断依据。",
  );
  assert.equal(
    payload.analysis.prediction.prediction.label,
    "synthetic_high_pathology_grade",
  );
  assert.equal(payload.analysis.explanation.shap.top_features[0].feature, "portal_vein_invasion");
  assert.ok(payload.analysis.evidence.results.length > 0);
  assert.equal(payload.analysis.memory.saved, false);
  assert.match(payload.text, /免责声明/);
  assert.ok(payload.deepreason.workflowTrace.length > 0);
  assert.ok(payload.deepreason.agentTrace.length > 0);
  assert.equal(payload.deepreason.gateDecision.status, "allow");
  assert.ok(payload.deepreason.claimEvidenceMap.length > 0);
  assert.ok(payload.deepreason.evidenceItems.length > 0);
  assert.equal(payload.deepreason.memoryProposal.status, "pending_approval");
  assert.equal(payload.deepreason.memoryProposal.applied, false);
  assert.equal(payload.deepreason.retryCount, 0);
  assert.equal(payload.deepreason.verificationResult.passed, true);
  assert.ok(
    payload.toolCalls.some(
      (call: { toolName: string }) => call.toolName === "createMemoryProposal",
    ),
  );
  assert.ok(
    !payload.toolCalls.some(
      (call: { toolName: string }) => call.toolName === "saveCaseMemory",
    ),
  );
});
