import assert from "node:assert/strict";
import { test } from "node:test";
import { retrieveMedicalEvidence } from "../src/search.js";

test("hybrid retrieval returns cited evidence for SHAP features", () => {
  const result = retrieveMedicalEvidence({
    query: "HCC 高分级 SHAP 解释 AFP portal vein radiomics entropy GLCM",
    featureNames: [
      "afp_ng_ml",
      "portal_vein_invasion",
      "radiomics_entropy",
      "radiomics_glcm_contrast",
    ],
    topK: 5,
  });

  assert.equal(result.retrievalMethod, "bm25+local_hash_embedding+heuristic_rerank");
  assert.ok(result.evidenceSufficient);
  assert.equal(result.results.length, 5);
  assert.ok(result.results.some((hit) => hit.id === "KB-HCC-003"));
  assert.ok(result.results.some((hit) => hit.source.url.startsWith("https://")));
});

test("hybrid retrieval reports low confidence for unrelated questions", () => {
  const result = retrieveMedicalEvidence({
    query: "spacecraft propulsion orchid taxonomy",
    featureNames: [],
    topK: 3,
  });

  assert.equal(result.confidence, "low");
  assert.equal(result.evidenceSufficient, false);
});

test("retrieval ablation mode returns comparable cited results", () => {
  const result = retrieveMedicalEvidence({
    query: "AFP HCC tumor marker",
    featureNames: ["afp_ng_ml"],
    topK: 5,
    ablationMode: "no_query_expansion",
  });

  assert.equal(result.retrievalMethod, "ablation:no_query_expansion");
  assert.equal(result.results.length, 5);
  assert.ok(result.results.every((hit) => hit.source.url.startsWith("https://")));
});
