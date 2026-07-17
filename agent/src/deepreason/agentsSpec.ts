import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { z } from "zod";
import { hccDeepReasonHandlerSpecs } from "./toolAdapters.js";
import type { HccWorkflowSpec } from "./workflowSpec.js";

const idPattern = /^[a-z][a-z0-9_]*$/;
const freeformRecordSchema = z.record(z.string(), z.unknown());
const modelRoleSchema = z.enum(["planner", "worker", "critic", "grader"]);

const builtinAgentTools = new Set(["workflow_status"]);
const plannedAgentTools = new Set<string>([]);
const finalSpecForbiddenTools = new Set(["saveCaseMemory"]);
const allowedMemoryScopes = new Set([
  "short_term:read",
  "short_term:write",
  "session_state:read",
  "session_state:write",
  "case_memory:read",
  "case_memory:proposal",
  "evidence_ledger:read",
  "evidence_ledger:append",
  "knowledge_base:read",
  "gate_decisions:read",
  "gate_decisions:append",
  "workflow_state:read",
]);
const requiredHccAgentIds = [
  "hcc_coordinator",
  "feature_collector",
  "memory_manager",
  "prediction_operator",
  "explanation_operator",
  "medical_retriever",
  "claim_checker",
  "source_verifier",
  "medical_safety_reviewer",
  "report_writer",
];

export const hccAgentRoleSpecSchema = z
  .object({
    id: z.string().regex(idPattern),
    label: z.string().min(1),
    description: z.string().min(1),
    responsibilities: z.array(z.string().min(1)).min(1),
    model_role: modelRoleSchema,
    tools: z.array(z.string().min(1)),
    permissions: freeformRecordSchema,
    memory_access: z.array(z.string().min(1)),
    workflow_nodes: z.array(z.string().regex(idPattern)),
    handoff_contract: freeformRecordSchema,
    ui: freeformRecordSchema,
  })
  .strict();

export const hccAgentsSpecSchema = z
  .object({
    version: z.literal("1.0"),
    name: z.string().min(1),
    revision: z.string().min(1),
    protected_agents: z.array(z.string().regex(idPattern)),
    agents: z.array(hccAgentRoleSpecSchema).min(1),
  })
  .strict();

export type HccAgentRoleSpec = z.infer<typeof hccAgentRoleSpecSchema>;
export type HccAgentsSpec = z.infer<typeof hccAgentsSpecSchema>;

export type HccAgentsValidation = {
  ok: boolean;
  implementationReady: boolean;
  errors: string[];
  warnings: string[];
  plannedTools: string[];
  agentCount: number;
  boundWorkflowNodes: string[];
};

function duplicates(values: string[]): string[] {
  const seen = new Set<string>();
  const duplicated = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      duplicated.add(value);
    }
    seen.add(value);
  }
  return [...duplicated].sort();
}

function sameMembers(actual: string[], expected: string[]): boolean {
  if (actual.length !== expected.length) {
    return false;
  }
  const actualSet = new Set(actual);
  return expected.every((item) => actualSet.has(item));
}

function describeExpected(expected: string[]): string {
  return expected.slice().sort().join(", ");
}

function requireExactTools(
  agent: HccAgentRoleSpec | undefined,
  expected: string[],
  errors: string[],
): void {
  if (!agent) {
    return;
  }
  if (!sameMembers(agent.tools, expected)) {
    errors.push(
      `agent ${agent.id} tools must be exactly: ${describeExpected(expected)}`,
    );
  }
}

function requireExactNodes(
  agent: HccAgentRoleSpec | undefined,
  expected: string[],
  errors: string[],
): void {
  if (!agent) {
    return;
  }
  if (!sameMembers(agent.workflow_nodes, expected)) {
    errors.push(
      `agent ${agent.id} workflow_nodes must be exactly: ${describeExpected(expected)}`,
    );
  }
}

function requirePermission(
  agent: HccAgentRoleSpec | undefined,
  key: string,
  expected: unknown,
  errors: string[],
): void {
  if (!agent) {
    return;
  }
  if (agent.permissions[key] !== expected) {
    errors.push(`agent ${agent.id} permission ${key} must be ${String(expected)}`);
  }
}

function forbidTools(
  agent: HccAgentRoleSpec | undefined,
  tools: string[],
  errors: string[],
): void {
  if (!agent) {
    return;
  }
  const forbidden = tools.filter((tool) => agent.tools.includes(tool));
  if (forbidden.length > 0) {
    errors.push(
      `agent ${agent.id} has forbidden tool(s): ${forbidden.sort().join(", ")}`,
    );
  }
}

function requireMemoryIncludes(
  agent: HccAgentRoleSpec | undefined,
  scopes: string[],
  errors: string[],
): void {
  if (!agent) {
    return;
  }
  for (const scope of scopes) {
    if (!agent.memory_access.includes(scope)) {
      errors.push(`agent ${agent.id} must include memory scope: ${scope}`);
    }
  }
}

export function validateHccAgentsSpec(
  input: unknown,
  workflowSpec?: HccWorkflowSpec,
): HccAgentsValidation {
  const parsed = hccAgentsSpecSchema.safeParse(input);
  if (!parsed.success) {
    return {
      ok: false,
      implementationReady: false,
      errors: parsed.error.issues.map(
        (issue) => `${issue.path.join(".") || "agents"}: ${issue.message}`,
      ),
      warnings: [],
      plannedTools: [],
      agentCount: 0,
      boundWorkflowNodes: [],
    };
  }

  const spec = parsed.data;
  const errors: string[] = [];
  const warnings: string[] = [];
  const implementedTools = new Set<string>(
    hccDeepReasonHandlerSpecs().map((handler) => handler.handlerName),
  );
  const knownTools = new Set<string>([
    ...implementedTools,
    ...plannedAgentTools,
    ...builtinAgentTools,
  ]);
  const plannedTools = new Set<string>();
  const agentIds = spec.agents.map((agent) => agent.id);
  const agentMap = new Map(spec.agents.map((agent) => [agent.id, agent]));

  for (const duplicate of duplicates(agentIds)) {
    errors.push(`duplicate agent id: ${duplicate}`);
  }
  for (const duplicate of duplicates(spec.protected_agents)) {
    errors.push(`duplicate protected agent id: ${duplicate}`);
  }

  for (const requiredAgentId of requiredHccAgentIds) {
    if (!agentMap.has(requiredAgentId)) {
      errors.push(`required HCC agent is missing: ${requiredAgentId}`);
    }
    if (!spec.protected_agents.includes(requiredAgentId)) {
      errors.push(`required HCC agent is not protected: ${requiredAgentId}`);
    }
  }
  for (const protectedAgent of spec.protected_agents) {
    if (!agentMap.has(protectedAgent)) {
      errors.push(`protected agent is missing from agents: ${protectedAgent}`);
    }
  }

  for (const agent of spec.agents) {
    if (duplicates(agent.tools).length > 0) {
      errors.push(`agent ${agent.id} has duplicate tools`);
    }
    if (duplicates(agent.workflow_nodes).length > 0) {
      errors.push(`agent ${agent.id} has duplicate workflow_nodes`);
    }
    if (agent.permissions.synthetic_data_only !== true) {
      errors.push(`agent ${agent.id} must be restricted to synthetic data only`);
    }
    for (const tool of agent.tools) {
      if (!knownTools.has(tool)) {
        errors.push(`agent ${agent.id} references unknown tool: ${tool}`);
      }
      if (plannedAgentTools.has(tool)) {
        plannedTools.add(tool);
      }
      if (finalSpecForbiddenTools.has(tool)) {
        errors.push(
          `agent ${agent.id} cannot use legacy direct-write tool ${tool}; use createMemoryProposal`,
        );
      }
    }
    for (const scope of agent.memory_access) {
      if (!allowedMemoryScopes.has(scope)) {
        errors.push(`agent ${agent.id} references unsupported memory scope: ${scope}`);
      }
      if (scope === "case_memory:write") {
        errors.push("case_memory:write is forbidden; use case_memory:proposal");
      }
    }
  }

  const dangerousGenerationTools = [
    "predictHccGrade",
    "explainPredictionWithShap",
    "generateHccReport",
  ];
  requireExactNodes(agentMap.get("hcc_coordinator"), ["intake", "respond"], errors);
  requireExactTools(agentMap.get("hcc_coordinator"), ["workflow_status"], errors);
  requirePermission(
    agentMap.get("hcc_coordinator"),
    "cannot_generate_medical_claims",
    true,
    errors,
  );

  requireExactNodes(agentMap.get("feature_collector"), ["feature_check"], errors);
  requireExactTools(
    agentMap.get("feature_collector"),
    ["checkFeatureCompleteness"],
    errors,
  );
  requirePermission(
    agentMap.get("feature_collector"),
    "forbid_downstream_when_missing",
    true,
    errors,
  );

  requireExactNodes(
    agentMap.get("memory_manager"),
    ["history_retrieve", "memory_proposal"],
    errors,
  );
  requireExactTools(
    agentMap.get("memory_manager"),
    ["getPatientHistory", "createMemoryProposal"],
    errors,
  );
  requirePermission(agentMap.get("memory_manager"), "direct_case_memory_write", false, errors);
  requirePermission(
    agentMap.get("memory_manager"),
    "requires_approval_for_write",
    true,
    errors,
  );
  requireMemoryIncludes(
    agentMap.get("memory_manager"),
    ["case_memory:read", "case_memory:proposal"],
    errors,
  );

  requireExactNodes(agentMap.get("prediction_operator"), ["prediction"], errors);
  requireExactTools(agentMap.get("prediction_operator"), ["predictHccGrade"], errors);
  requirePermission(
    agentMap.get("prediction_operator"),
    "forbid_llm_prediction",
    true,
    errors,
  );
  requirePermission(
    agentMap.get("prediction_operator"),
    "requires_complete_features",
    true,
    errors,
  );

  requireExactNodes(agentMap.get("explanation_operator"), ["shap_explain"], errors);
  requireExactTools(
    agentMap.get("explanation_operator"),
    ["explainPredictionWithShap"],
    errors,
  );
  requirePermission(
    agentMap.get("explanation_operator"),
    "forbid_llm_shap_values",
    true,
    errors,
  );

  requireExactNodes(
    agentMap.get("medical_retriever"),
    ["evidence_retrieve", "build_follow_up_query"],
    errors,
  );
  requireExactTools(
    agentMap.get("medical_retriever"),
    ["retrieveMedicalEvidence", "buildFollowUpQuery"],
    errors,
  );
  requirePermission(
    agentMap.get("medical_retriever"),
    "public_traceable_sources_only",
    true,
    errors,
  );
  forbidTools(
    agentMap.get("medical_retriever"),
    dangerousGenerationTools,
    errors,
  );

  requireExactNodes(agentMap.get("claim_checker"), ["claim_check"], errors);
  requireExactTools(agentMap.get("claim_checker"), ["checkClaims"], errors);
  requirePermission(
    agentMap.get("claim_checker"),
    "requires_evidence_binding",
    true,
    errors,
  );
  requirePermission(
    agentMap.get("claim_checker"),
    "cannot_generate_medical_judgment",
    true,
    errors,
  );

  requireExactNodes(agentMap.get("source_verifier"), ["source_verify"], errors);
  requireExactTools(agentMap.get("source_verifier"), ["verifyEvidenceSources"], errors);
  requirePermission(
    agentMap.get("source_verifier"),
    "cannot_generate_medical_judgment",
    true,
    errors,
  );

  requireExactNodes(
    agentMap.get("medical_safety_reviewer"),
    ["confidence_gate", "verify"],
    errors,
  );
  requireExactTools(
    agentMap.get("medical_safety_reviewer"),
    ["evaluateMedicalConfidenceGate"],
    errors,
  );
  requirePermission(
    agentMap.get("medical_safety_reviewer"),
    "requires_disclaimer",
    true,
    errors,
  );
  requirePermission(
    agentMap.get("medical_safety_reviewer"),
    "cannot_modify_tool_outputs",
    true,
    errors,
  );

  requireExactNodes(agentMap.get("report_writer"), ["report_generate"], errors);
  requireExactTools(agentMap.get("report_writer"), ["generateHccReport"], errors);
  requirePermission(
    agentMap.get("report_writer"),
    "evidence_only_generation",
    true,
    errors,
  );
  requirePermission(agentMap.get("report_writer"), "requires_disclaimer", true, errors);
  forbidTools(
    agentMap.get("report_writer"),
    ["predictHccGrade", "explainPredictionWithShap", "retrieveMedicalEvidence"],
    errors,
  );

  const boundWorkflowNodes = spec.agents.flatMap((agent) => agent.workflow_nodes);
  if (workflowSpec) {
    const workflowNodeMap = new Map(workflowSpec.nodes.map((node) => [node.id, node]));
    const workflowNodeIds = workflowSpec.nodes.map((node) => node.id);
    const workflowAgentIds = new Set(workflowSpec.nodes.map((node) => node.agent));

    for (const workflowAgentId of workflowAgentIds) {
      if (!agentMap.has(workflowAgentId)) {
        errors.push(`workflow references agent not defined in AgentsSpec: ${workflowAgentId}`);
      }
    }
    for (const nodeId of boundWorkflowNodes) {
      if (!workflowNodeMap.has(nodeId)) {
        errors.push(`AgentsSpec binds unknown workflow node: ${nodeId}`);
      }
    }
    for (const nodeId of workflowNodeIds) {
      const owners = spec.agents.filter((agent) =>
        agent.workflow_nodes.includes(nodeId),
      );
      if (owners.length === 0) {
        errors.push(`workflow node is not bound to an agent: ${nodeId}`);
        continue;
      }
      if (owners.length > 1) {
        errors.push(
          `workflow node is bound to multiple agents: ${nodeId} -> ${owners
            .map((agent) => agent.id)
            .sort()
            .join(", ")}`,
        );
        continue;
      }
      const workflowNode = workflowNodeMap.get(nodeId);
      const owner = owners[0];
      if (workflowNode && owner && workflowNode.agent !== owner.id) {
        errors.push(
          `workflow node ${nodeId} is assigned to ${workflowNode.agent} but bound to ${owner.id}`,
        );
      }
      if (
        workflowNode &&
        owner &&
        workflowNode.handler_kind === "plugin_tool" &&
        !owner.tools.includes(workflowNode.handler)
      ) {
        errors.push(
          `agent ${owner.id} must list handler ${workflowNode.handler} for workflow node ${nodeId}`,
        );
      }
    }
  } else {
    warnings.push("workflowSpec was not provided; node binding checks were skipped");
  }

  const sortedPlannedTools = [...plannedTools].sort();
  if (sortedPlannedTools.length > 0) {
    warnings.push(
      `planned tools are not implemented yet: ${sortedPlannedTools.join(", ")}`,
    );
  }

  return {
    ok: errors.length === 0,
    implementationReady: errors.length === 0 && sortedPlannedTools.length === 0,
    errors,
    warnings,
    plannedTools: sortedPlannedTools,
    agentCount: spec.agents.length,
    boundWorkflowNodes: [...new Set(boundWorkflowNodes)].sort(),
  };
}

export function loadHccAgentsSpec(
  filePath = resolve(
    process.cwd(),
    "configs",
    "agents",
    "hcc_analysis.agents.json",
  ),
): HccAgentsSpec {
  const raw: unknown = JSON.parse(readFileSync(filePath, "utf-8"));
  return hccAgentsSpecSchema.parse(raw);
}
