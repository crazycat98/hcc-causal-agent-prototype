import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import {
  createHccDeepReasonToolAdapters,
  hccDeepReasonHandlerSpecs,
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
let predictRequestCount = 0;
let explainRequestCount = 0;

function predictionPayload(body: { patient_id?: string; features: HccFeatures }) {
  predictRequestCount += 1;
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
  explainRequestCount += 1;
  return {
    safety_notice: "演示用合成数据，非真实患者数据；非临床诊断依据。",
    prediction: {
      label: "synthetic_high_pathology_grade",
      probability_high_grade: 0.91,
      probability_low_or_intermediate: 0.09,
      uncertain_probability_band: false,
    },
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
        "SHAP Top-N 与模拟 FCI 因果候选集合存在交集。",
      caveat:
        "SHAP 解释反映模型内部贡献，不等同于真实医学因果效应。",
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
  return mkdtempSync(join(tmpdir(), "hcc-deepreason-adapters-"));
}

test("DeepReason handler specs expose node, agent, permission, and deterministic metadata", () => {
  const specs = hccDeepReasonHandlerSpecs();
  assert.equal(specs.length, 12);
  assert.ok(specs.every((spec) => spec.deterministic));
  assert.deepEqual(
    specs.map((spec) => spec.nodeId),
    [
      "feature_check",
      "history_retrieve",
      "prediction",
      "shap_explain",
      "evidence_retrieve",
      "case_memory_write",
      "claim_check",
      "build_follow_up_query",
      "source_verify",
      "confidence_gate",
      "report_generate",
      "memory_proposal",
    ],
  );
});

test("DeepReason adapters wrap existing HCC tools without using the Vercel AI SDK loop", async () => {
  predictRequestCount = 0;
  explainRequestCount = 0;
  const memoryDir = tempMemoryDir();
  const adapters = createHccDeepReasonToolAdapters({
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
    memoryDir,
  });

  const featureCheck = await adapters.checkFeatureCompleteness({
    sessionId: "dr-session",
    patientId: "dr-patient",
    features: demoFeatures,
  });
  assert.equal(featureCheck.handlerName, "checkFeatureCompleteness");
  assert.equal(featureCheck.nodeId, "feature_check");
  assert.equal(featureCheck.agentName, "feature_collector");
  assert.equal(featureCheck.output.complete, true);
  assert.equal(featureCheck.evidenceDrafts[0]?.sourceType, "session_state");

  const historyBefore = await adapters.getPatientHistory({
    patientId: "dr-patient",
  });
  assert.equal(historyBefore.nodeId, "history_retrieve");
  assert.equal(historyBefore.output.hasHistory, false);
  assert.equal(historyBefore.evidenceDrafts.length, 0);

  const prediction = await adapters.predictHccGrade({
    patientId: "dr-patient",
    features: demoFeatures,
  });
  assert.equal(prediction.nodeId, "prediction");
  assert.equal(prediction.output.prediction.probability_high_grade, 0.91);
  assert.equal(prediction.evidenceDrafts[0]?.sourceType, "model_prediction");
  assert.equal(predictRequestCount, 1);

  const shap = await adapters.explainPredictionWithShap({
    patientId: "dr-patient",
    features: demoFeatures,
    topN: 5,
  });
  assert.equal(shap.nodeId, "shap_explain");
  assert.equal(shap.output.shap.top_features[0]?.feature, "portal_vein_invasion");
  assert.equal(shap.evidenceDrafts[0]?.sourceType, "model_explanation");
  assert.equal(explainRequestCount, 1);

  const retrieval = await adapters.retrieveMedicalEvidence({
    query: "HCC pathology grade AFP portal vein invasion radiomics",
    featureNames: ["afp_ng_ml", "portal_vein_invasion"],
    topK: 5,
  });
  assert.equal(retrieval.nodeId, "evidence_retrieve");
  assert.ok(retrieval.output.results.length > 0);
  assert.ok(retrieval.evidenceDrafts.length > 0);
  assert.ok(retrieval.evidenceDrafts.every((item) => item.sourceType === "knowledge_base"));

  const saved = await adapters.saveCaseMemory({
    patientId: "dr-patient",
    sessionId: "dr-session",
    features: demoFeatures,
    prediction: prediction.output.prediction,
    shap: {
      top_features: shap.output.shap.top_features.map((item) => ({
        feature: item.feature,
        shap_value: item.shap_value,
        direction: item.direction,
        trust_level: item.trust_level,
      })),
      high_trust_features: shap.output.shap.high_trust_features,
      statistical_only_features: shap.output.shap.statistical_only_features,
    },
    retrieval: {
      confidence: retrieval.output.confidence,
      evidenceSufficient: retrieval.output.evidenceSufficient,
      evidenceIds: retrieval.output.results.map((hit) => hit.id),
    },
  });
  assert.equal(saved.nodeId, "case_memory_write");
  assert.equal(saved.output.saved, true);

  const historyAfter = await adapters.getPatientHistory({
    patientId: "dr-patient",
  });
  assert.equal(historyAfter.output.hasHistory, true);
  assert.equal(historyAfter.evidenceDrafts[0]?.sourceType, "case_memory");
});

test("DeepReason feature adapter preserves missing-feature clarification boundary", async () => {
  const adapters = createHccDeepReasonToolAdapters({
    predictionEndpoint: `${baseUrl}/predict`,
    explanationEndpoint: `${baseUrl}/explain`,
    memoryDir: tempMemoryDir(),
  });
  const result = await adapters.checkFeatureCompleteness({
    sessionId: "dr-missing-session",
    patientId: "dr-missing-patient",
    features: {
      tumor_size_cm: 6.2,
      afp_ng_ml: 420,
    },
  });

  assert.equal(result.output.complete, false);
  assert.ok(result.output.missingFeatures.includes("alt_u_l"));
  assert.equal(result.nodeId, "feature_check");
  assert.equal(result.evidenceDrafts[0]?.claimType, "feature_state");
});
