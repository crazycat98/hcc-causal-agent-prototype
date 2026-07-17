import assert from "node:assert/strict";
import { test } from "node:test";
import {
  loadHccWorkflowSpec,
  validateHccWorkflowSpec,
  type HccWorkflowSpec,
} from "../src/deepreason/index.js";

function cloneSpec(spec: HccWorkflowSpec): HccWorkflowSpec {
  return structuredClone(spec);
}

function reachableNodeIds(spec: HccWorkflowSpec): Set<string> {
  const adjacency = new Map<string, string[]>();
  for (const edge of spec.edges) {
    adjacency.set(edge.from, [...(adjacency.get(edge.from) ?? []), edge.to]);
  }

  const reached = new Set<string>();
  const queue = [spec.start_node];
  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || reached.has(current)) {
      continue;
    }
    reached.add(current);
    queue.push(...(adjacency.get(current) ?? []));
  }
  return reached;
}

test("HCC WorkflowSpec is structurally valid with all plugin handlers implemented", () => {
  const spec = loadHccWorkflowSpec();
  const validation = validateHccWorkflowSpec(spec);

  assert.equal(spec.name, "hcc-deepreason-analysis-workflow");
  assert.equal(spec.nodes.length, 14);
  assert.equal(spec.edges.length, 19);
  assert.equal(validation.ok, true, validation.errors.join("\n"));
  assert.equal(validation.implementationReady, true);
  assert.deepEqual(validation.plannedHandlers, []);
  assert.equal(reachableNodeIds(spec).size, spec.nodes.length);
  assert.ok(validation.checkpoints.includes("prediction"));
  assert.ok(validation.checkpoints.includes("confidence_gate"));
  assert.ok(validation.checkpoints.includes("memory_proposal"));
});

test("HCC WorkflowSpec preserves clarification, deterministic prediction, and bounded evidence retry", () => {
  const spec = loadHccWorkflowSpec();
  const missingFeatureEdge = spec.edges.find(
    (edge) => edge.id === "feature_check_to_respond_missing",
  );
  assert.equal(missingFeatureEdge?.from, "feature_check");
  assert.equal(missingFeatureEdge?.to, "respond");
  assert.equal(missingFeatureEdge?.gate_policy.forbid_downstream_tools, true);

  const predictionInbound = spec.edges
    .filter((edge) => edge.to === "prediction")
    .map((edge) => edge.from);
  assert.deepEqual(predictionInbound, ["history_retrieve"]);

  const claimRetry = spec.edges.find(
    (edge) => edge.id === "claim_check_to_follow_up",
  );
  assert.equal(claimRetry?.type, "retry");
  assert.equal(claimRetry?.planner_contract.retry_counter, "evidence_retry_count");
  assert.equal(claimRetry?.planner_contract.max_retry, 2);
  assert.equal(claimRetry?.planner_contract.on_exhausted, "confidence_gate");

  const exhaustedEdge = spec.edges.find(
    (edge) => edge.id === "claim_check_to_gate_exhausted",
  );
  assert.equal(exhaustedEdge?.to, "confidence_gate");
  assert.equal(exhaustedEdge?.gate_policy.force_limited_output, true);
});

test("HCC WorkflowSpec gates report generation and memory proposal", () => {
  const spec = loadHccWorkflowSpec();
  const reportInbound = spec.edges
    .filter((edge) => edge.to === "report_generate")
    .map((edge) => edge.from)
    .sort();
  assert.deepEqual(reportInbound, ["confidence_gate", "verify"]);

  const memoryInbound = spec.edges
    .filter((edge) => edge.to === "memory_proposal")
    .map((edge) => edge.from);
  assert.deepEqual(memoryInbound, ["verify"]);

  assert.ok(
    spec.edges.some(
      (edge) =>
        edge.from === "confidence_gate" &&
        edge.to === "respond" &&
        edge.type === "branch",
    ),
  );
});

test("Workflow validation rejects a direct prediction-to-report bypass", () => {
  const spec = cloneSpec(loadHccWorkflowSpec());
  const template = spec.edges.find((edge) => edge.id === "prediction_to_shap");
  assert.ok(template);
  spec.edges.push({
    ...template,
    id: "prediction_to_report_unsafe",
    from: "prediction",
    to: "report_generate",
  });

  const validation = validateHccWorkflowSpec(spec);
  assert.equal(validation.ok, false);
  assert.ok(
    validation.errors.includes(
      "report_generate has an unsafe inbound edge that bypasses gate or revision",
    ),
  );
});

test("Workflow validation rejects unbounded evidence retry", () => {
  const spec = cloneSpec(loadHccWorkflowSpec());
  const retryEdge = spec.edges.find(
    (edge) => edge.id === "claim_check_to_follow_up",
  );
  assert.ok(retryEdge);
  delete retryEdge.planner_contract.max_retry;

  const validation = validateHccWorkflowSpec(spec);
  assert.equal(validation.ok, false);
  assert.ok(
    validation.errors.some((error) =>
      error.includes("must declare a positive max_retry"),
    ),
  );
  assert.ok(
    validation.errors.includes("claim_check evidence retry must use max_retry 2"),
  );
});

test("Workflow validation rejects deletion of a required protected node", () => {
  const spec = cloneSpec(loadHccWorkflowSpec());
  spec.nodes = spec.nodes.filter((node) => node.id !== "source_verify");

  const validation = validateHccWorkflowSpec(spec);
  assert.equal(validation.ok, false);
  assert.ok(
    validation.errors.includes(
      "protected node is missing from nodes: source_verify",
    ),
  );
  assert.ok(
    validation.errors.includes(
      "required HCC workflow node is missing: source_verify",
    ),
  );
});
