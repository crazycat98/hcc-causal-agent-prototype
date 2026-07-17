import assert from "node:assert/strict";
import { test } from "node:test";
import {
  loadHccAgentsSpec,
  loadHccWorkflowSpec,
  validateHccAgentsSpec,
  type HccAgentsSpec,
} from "../src/deepreason/index.js";

function cloneSpec(spec: HccAgentsSpec): HccAgentsSpec {
  return structuredClone(spec);
}

function agentById(spec: HccAgentsSpec, id: string) {
  const agent = spec.agents.find((item) => item.id === id);
  assert.ok(agent, `missing agent ${id}`);
  return agent;
}

test("HCC AgentsSpec is structurally valid with all plugin tools implemented", () => {
  const agentsSpec = loadHccAgentsSpec();
  const workflowSpec = loadHccWorkflowSpec();
  const validation = validateHccAgentsSpec(agentsSpec, workflowSpec);

  assert.equal(agentsSpec.name, "hcc-deepreason-analysis-agents");
  assert.equal(validation.ok, true, validation.errors.join("\n"));
  assert.equal(validation.implementationReady, true);
  assert.equal(validation.agentCount, 10);
  assert.deepEqual(validation.plannedTools, []);
  assert.deepEqual(
    validation.boundWorkflowNodes,
    workflowSpec.nodes.map((node) => node.id).sort(),
  );
});

test("HCC AgentsSpec binds every WorkflowSpec node to its declared owner", () => {
  const agentsSpec = loadHccAgentsSpec();
  const workflowSpec = loadHccWorkflowSpec();

  for (const workflowNode of workflowSpec.nodes) {
    const owners = agentsSpec.agents.filter((agent) =>
      agent.workflow_nodes.includes(workflowNode.id),
    );
    assert.equal(owners.length, 1, workflowNode.id);
    assert.equal(owners[0]?.id, workflowNode.agent, workflowNode.id);
    if (workflowNode.handler_kind === "plugin_tool") {
      assert.ok(
        owners[0]?.tools.includes(workflowNode.handler),
        `${owners[0]?.id} must expose ${workflowNode.handler}`,
      );
    }
  }
});

test("HCC AgentsSpec encodes high-risk role boundaries", () => {
  const agentsSpec = loadHccAgentsSpec();

  const prediction = agentById(agentsSpec, "prediction_operator");
  assert.deepEqual(prediction.tools, ["predictHccGrade"]);
  assert.equal(prediction.permissions.forbid_llm_prediction, true);
  assert.equal(prediction.memory_access.length, 0);

  const report = agentById(agentsSpec, "report_writer");
  assert.deepEqual(report.tools, ["generateHccReport"]);
  assert.equal(report.permissions.evidence_only_generation, true);
  assert.equal(report.permissions.requires_disclaimer, true);
  assert.equal(report.permissions.cannot_invent_citations, true);

  const memory = agentById(agentsSpec, "memory_manager");
  assert.deepEqual(memory.tools, ["getPatientHistory", "createMemoryProposal"]);
  assert.equal(memory.permissions.direct_case_memory_write, false);
  assert.equal(memory.permissions.requires_approval_for_write, true);
  assert.ok(memory.memory_access.includes("case_memory:proposal"));
  assert.ok(!memory.memory_access.includes("case_memory:write"));
});

test("Agent validation rejects report writer access to deterministic prediction", () => {
  const agentsSpec = cloneSpec(loadHccAgentsSpec());
  agentById(agentsSpec, "report_writer").tools.push("predictHccGrade");

  const validation = validateHccAgentsSpec(agentsSpec, loadHccWorkflowSpec());
  assert.equal(validation.ok, false);
  assert.ok(
    validation.errors.includes(
      "agent report_writer tools must be exactly: generateHccReport",
    ),
  );
  assert.ok(
    validation.errors.includes(
      "agent report_writer has forbidden tool(s): predictHccGrade",
    ),
  );
});

test("Agent validation rejects legacy direct case memory write assignment", () => {
  const agentsSpec = cloneSpec(loadHccAgentsSpec());
  agentById(agentsSpec, "memory_manager").tools.push("saveCaseMemory");

  const validation = validateHccAgentsSpec(agentsSpec, loadHccWorkflowSpec());
  assert.equal(validation.ok, false);
  assert.ok(
    validation.errors.some((error) =>
      error.includes("cannot use legacy direct-write tool saveCaseMemory"),
    ),
  );
});

test("Agent validation rejects deletion of a protected domain agent", () => {
  const agentsSpec = cloneSpec(loadHccAgentsSpec());
  agentsSpec.agents = agentsSpec.agents.filter(
    (agent) => agent.id !== "source_verifier",
  );

  const validation = validateHccAgentsSpec(agentsSpec, loadHccWorkflowSpec());
  assert.equal(validation.ok, false);
  assert.ok(
    validation.errors.includes("required HCC agent is missing: source_verifier"),
  );
  assert.ok(
    validation.errors.includes(
      "protected agent is missing from agents: source_verifier",
    ),
  );
});

test("Agent validation rejects WorkflowSpec node ownership mismatch", () => {
  const agentsSpec = cloneSpec(loadHccAgentsSpec());
  const prediction = agentById(agentsSpec, "prediction_operator");
  prediction.workflow_nodes = [];
  agentById(agentsSpec, "report_writer").workflow_nodes.push("prediction");

  const validation = validateHccAgentsSpec(agentsSpec, loadHccWorkflowSpec());
  assert.equal(validation.ok, false);
  assert.ok(
    validation.errors.includes(
      "agent prediction_operator workflow_nodes must be exactly: prediction",
    ),
  );
  assert.ok(
    validation.errors.includes(
      "workflow node prediction is assigned to prediction_operator but bound to report_writer",
    ),
  );
});

test("Agent validation rejects unsupported memory scope and missing synthetic-data boundary", () => {
  const agentsSpec = cloneSpec(loadHccAgentsSpec());
  const memory = agentById(agentsSpec, "memory_manager");
  memory.memory_access.push("case_memory:write");
  memory.permissions.synthetic_data_only = false;

  const validation = validateHccAgentsSpec(agentsSpec, loadHccWorkflowSpec());
  assert.equal(validation.ok, false);
  assert.ok(
    validation.errors.includes(
      "agent memory_manager references unsupported memory scope: case_memory:write",
    ),
  );
  assert.ok(
    validation.errors.includes(
      "case_memory:write is forbidden; use case_memory:proposal",
    ),
  );
  assert.ok(
    validation.errors.includes(
      "agent memory_manager must be restricted to synthetic data only",
    ),
  );
});
