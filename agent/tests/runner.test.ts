import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer, type Server } from "node:http";
import { after, before, test } from "node:test";
import type { HccFeatures } from "../src/features.js";
import { runHccAgent } from "../src/runner.js";

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
let predictRequestCount = 0;
let explainRequestCount = 0;

function probabilityFor(features: HccFeatures) {
  return features.tumor_size_cm >= 6 ? 0.91 : 0.47;
}

function predictionPayload(body: { patient_id?: string; features: HccFeatures }) {
  const probability = probabilityFor(body.features);
  return {
    safety_notice: "演示用合成数据，非真实患者数据；非临床诊断依据。",
    prediction: {
      label:
        probability >= 0.5
          ? "synthetic_high_pathology_grade"
          : "synthetic_low_or_intermediate_grade",
      probability_high_grade: probability,
      probability_low_or_intermediate: 1 - probability,
      uncertain_probability_band: probability >= 0.4 && probability <= 0.6,
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
          consistency_note: "该特征同时位于 SHAP Top-N 与模拟 FCI 因果候选集合中。",
        },
        {
          feature: "afp_ng_ml",
          value: body.features.afp_ng_ml,
          shap_value: 0.08,
          abs_shap_value: 0.08,
          direction: "pushes_toward_high_grade",
          trust_level: "high_trust_causal_candidate",
          consistency_note: "该特征同时位于 SHAP Top-N 与模拟 FCI 因果候选集合中。",
        },
      ],
      causal_candidate_features: Object.keys(body.features),
      high_trust_features: ["portal_vein_invasion", "afp_ng_ml"],
      statistical_only_features: [],
      consistency_summary:
        "SHAP Top-N 与模拟 FCI 因果候选集合存在交集；交集特征标记为高可信解释线索。",
      caveat: "SHAP 解释反映模型内部贡献，不等同于真实医学因果效应；本原型仅用于工程演示。",
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
        req.url === "/predict"
          ? (predictRequestCount += 1, predictionPayload(body))
          : (explainRequestCount += 1, explanationPayload(body));

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
  return mkdtempSync(join(tmpdir(), "hcc-agent-memory-"));
}

test("M5 Agent checks, reads history, predicts, explains, retrieves, saves memory, and cites KB paragraphs", async () => {
  predictRequestCount = 0;
  explainRequestCount = 0;
  const result = await runHccAgent({
    sessionId: "session-complete",
    patientId: "test-001",
    features: demoFeatures,
    memoryDir: tempMemoryDir(),
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
  });

  assert.equal(result.finishReason, "stop");
  assert.equal(result.steps, 7);
  assert.deepEqual(
    result.toolCalls.map((call) => call.toolName),
    [
      "checkFeatureCompleteness",
      "getPatientHistory",
      "predictHccGrade",
      "explainPredictionWithShap",
      "retrieveMedicalEvidence",
      "saveCaseMemory",
    ],
  );
  assert.match(result.text, /probability_high_grade: 91\.0%/);
  assert.match(result.text, /医学依据检索/);
  assert.match(result.text, /\[KB-/);
  assert.match(result.text, /会话与病例记忆/);
  assert.match(result.text, /首次合成病例分析/);
  assert.equal(predictRequestCount, 1);
  assert.equal(explainRequestCount, 1);
});

test("M5 session memory lets a second turn complete missing features without re-asking known fields", async () => {
  predictRequestCount = 0;
  explainRequestCount = 0;
  const memoryDir = tempMemoryDir();
  const sessionId = "session-two-turn";

  const first = await runHccAgent({
    sessionId,
    patientId: "test-session-memory",
    features: {
      tumor_size_cm: 6.2,
      afp_ng_ml: 420,
    },
    memoryDir,
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
  });

  assert.deepEqual(
    first.toolCalls.map((call) => call.toolName),
    ["checkFeatureCompleteness"],
  );
  assert.match(first.text, /缺少以下合成特征/);
  assert.equal(predictRequestCount, 0);

  const second = await runHccAgent({
    sessionId,
    patientId: "test-session-memory",
    features: {
      alt_u_l: 61,
      ast_u_l: 72,
      bilirubin_umol_l: 24,
      albumin_g_l: 36,
      platelet_10e9_l: 128,
      portal_vein_invasion: 1,
      radiomics_entropy: 5.4,
      radiomics_glcm_contrast: 112,
    },
    memoryDir,
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
  });

  assert.equal(second.finishReason, "stop");
  assert.ok(
    second.toolCalls.some((call) => call.toolName === "predictHccGrade"),
  );
  assert.match(second.text, /probability_high_grade/);
  assert.equal(predictRequestCount, 1);
});

test("M5 cross-session case memory reports revisit trend for the same patient_id", async () => {
  const memoryDir = tempMemoryDir();
  const patientId = "test-revisit";

  await runHccAgent({
    sessionId: "visit-1",
    patientId,
    features: {
      ...demoFeatures,
      tumor_size_cm: 4.2,
    },
    memoryDir,
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
  });

  const second = await runHccAgent({
    sessionId: "visit-2",
    patientId,
    features: demoFeatures,
    memoryDir,
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
  });

  assert.match(second.text, /较上次分析/);
  assert.match(second.text, /probability_delta/);
  assert.match(second.text, /tumor_size_cm/);
});

test("M5 Agent asks for missing features and does not call downstream tools", async () => {
  predictRequestCount = 0;
  explainRequestCount = 0;
  const result = await runHccAgent({
    sessionId: "session-missing",
    patientId: "test-missing",
    features: {
      tumor_size_cm: 3.1,
      afp_ng_ml: 60,
    },
    memoryDir: tempMemoryDir(),
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
  });

  assert.equal(result.finishReason, "stop");
  assert.equal(result.steps, 2);
  assert.deepEqual(
    result.toolCalls.map((call) => call.toolName),
    ["checkFeatureCompleteness"],
  );
  assert.match(result.text, /缺少以下合成特征/);
  assert.match(result.text, /alt_u_l/);
  assert.equal(predictRequestCount, 0);
  assert.equal(explainRequestCount, 0);
});

