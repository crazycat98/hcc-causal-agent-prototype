import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { z } from "zod";
import { hccDeepReasonHandlerSpecs } from "./toolAdapters.js";

const idPattern = /^[a-z][a-z0-9_]*$/;
const supportedHandlerKinds = new Set(["builtin", "plugin_tool"]);
const knownBuiltinHandlers = new Set(["intake", "verify", "respond"]);
const plannedPluginHandlers = new Set<string>([]);

const freeformRecordSchema = z.record(z.string(), z.unknown());

export const hccWorkflowNodeSchema = z
  .object({
    id: z.string().regex(idPattern),
    label: z.string().min(1),
    agent: z.string().regex(idPattern),
    description: z.string(),
    work: z.string(),
    input_contract: z.string(),
    output_contract: z.string(),
    handler_kind: z.enum(["builtin", "plugin_tool"]),
    handler: z.string().min(1),
    checkpoint: z.boolean(),
    gate_policy: freeformRecordSchema,
    ui: freeformRecordSchema,
  })
  .strict();

export const hccWorkflowEdgeSchema = z
  .object({
    id: z.string().regex(idPattern),
    from: z.string().regex(idPattern),
    to: z.string().regex(idPattern),
    type: z.enum(["flow", "branch", "retry", "revise", "loop"]),
    condition: z.string().min(1),
    handoff_contract: freeformRecordSchema,
    gate_policy: freeformRecordSchema,
    planner_contract: freeformRecordSchema,
    reviewer_required: z.boolean(),
  })
  .strict();

export const hccWorkflowSpecSchema = z
  .object({
    version: z.literal("1.0"),
    name: z.string().min(1),
    revision: z.string().min(1),
    start_node: z.string().regex(idPattern),
    terminal_nodes: z.array(z.string().regex(idPattern)).min(1),
    protected_nodes: z.array(z.string().regex(idPattern)),
    nodes: z.array(hccWorkflowNodeSchema).min(1),
    edges: z.array(hccWorkflowEdgeSchema).min(1),
  })
  .strict();

export type HccWorkflowSpec = z.infer<typeof hccWorkflowSpecSchema>;
export type HccWorkflowNode = z.infer<typeof hccWorkflowNodeSchema>;
export type HccWorkflowEdge = z.infer<typeof hccWorkflowEdgeSchema>;

export type HccWorkflowValidation = {
  ok: boolean;
  implementationReady: boolean;
  errors: string[];
  warnings: string[];
  plannedHandlers: string[];
  checkpoints: string[];
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

function reachableNodes(startNode: string, edges: HccWorkflowEdge[]): Set<string> {
  const adjacency = new Map<string, string[]>();
  for (const edge of edges) {
    adjacency.set(edge.from, [...(adjacency.get(edge.from) ?? []), edge.to]);
  }
  const reached = new Set<string>();
  const queue = [startNode];
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

function edgeExists(
  edges: HccWorkflowEdge[],
  from: string,
  to: string,
  type?: HccWorkflowEdge["type"],
): boolean {
  return edges.some(
    (edge) =>
      edge.from === from &&
      edge.to === to &&
      (type === undefined || edge.type === type),
  );
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : undefined;
}

export function validateHccWorkflowSpec(input: unknown): HccWorkflowValidation {
  const parsed = hccWorkflowSpecSchema.safeParse(input);
  if (!parsed.success) {
    return {
      ok: false,
      implementationReady: false,
      errors: parsed.error.issues.map(
        (issue) => `${issue.path.join(".") || "workflow"}: ${issue.message}`,
      ),
      warnings: [],
      plannedHandlers: [],
      checkpoints: [],
    };
  }

  const spec = parsed.data;
  const errors: string[] = [];
  const warnings: string[] = [];
  const nodeIds = spec.nodes.map((node) => node.id);
  const edgeIds = spec.edges.map((edge) => edge.id);
  const nodeIdSet = new Set(nodeIds);
  const implementedPluginHandlers = new Set<string>(
    hccDeepReasonHandlerSpecs().map((handler) => handler.handlerName),
  );
  const plannedHandlers = new Set<string>();

  for (const duplicate of duplicates(nodeIds)) {
    errors.push(`duplicate node id: ${duplicate}`);
  }
  for (const duplicate of duplicates(edgeIds)) {
    errors.push(`duplicate edge id: ${duplicate}`);
  }
  if (!nodeIdSet.has(spec.start_node)) {
    errors.push(`start_node is missing from nodes: ${spec.start_node}`);
  }
  for (const terminal of spec.terminal_nodes) {
    if (!nodeIdSet.has(terminal)) {
      errors.push(`terminal node is missing from nodes: ${terminal}`);
    }
  }
  for (const protectedNode of spec.protected_nodes) {
    if (!nodeIdSet.has(protectedNode)) {
      errors.push(`protected node is missing from nodes: ${protectedNode}`);
    }
  }

  for (const node of spec.nodes) {
    if (!supportedHandlerKinds.has(node.handler_kind)) {
      errors.push(
        `unsupported handler_kind for ${node.id}: ${node.handler_kind}`,
      );
      continue;
    }
    if (
      node.handler_kind === "builtin" &&
      !knownBuiltinHandlers.has(node.handler)
    ) {
      errors.push(`unknown builtin handler for ${node.id}: ${node.handler}`);
    }
    if (node.handler_kind === "plugin_tool") {
      if (implementedPluginHandlers.has(node.handler)) {
        if (node.ui.implementation_status !== "implemented") {
          warnings.push(
            `implemented plugin handler ${node.handler} is not marked implemented`,
          );
        }
      } else if (plannedPluginHandlers.has(node.handler)) {
        plannedHandlers.add(node.handler);
        if (node.ui.implementation_status !== "planned") {
          warnings.push(
            `planned plugin handler ${node.handler} is not marked planned`,
          );
        }
      } else {
        errors.push(`unknown plugin_tool handler for ${node.id}: ${node.handler}`);
      }
    }
  }

  for (const edge of spec.edges) {
    if (!nodeIdSet.has(edge.from)) {
      errors.push(`edge ${edge.id} references missing from node: ${edge.from}`);
    }
    if (!nodeIdSet.has(edge.to)) {
      errors.push(`edge ${edge.id} references missing to node: ${edge.to}`);
    }
    if (edge.type === "retry" || edge.type === "revise") {
      const maxRetry = positiveInteger(edge.planner_contract.max_retry);
      if (maxRetry === undefined) {
        errors.push(`bounded edge ${edge.id} must declare a positive max_retry`);
      } else if (maxRetry > 3) {
        errors.push(`bounded edge ${edge.id} exceeds max_retry safety limit: ${maxRetry}`);
      }
      if (typeof edge.planner_contract.retry_counter !== "string") {
        errors.push(`bounded edge ${edge.id} must declare retry_counter`);
      }
    }
  }

  if (nodeIdSet.has(spec.start_node)) {
    const reachable = reachableNodes(spec.start_node, spec.edges);
    const unreachable = nodeIds.filter((nodeId) => !reachable.has(nodeId));
    if (unreachable.length > 0) {
      errors.push(`unreachable nodes from start_node: ${unreachable.join(", ")}`);
    }
    if (!spec.terminal_nodes.some((terminal) => reachable.has(terminal))) {
      errors.push("no terminal node is reachable from start_node");
    }
  }

  const requiredNodes = [
    "intake",
    "feature_check",
    "history_retrieve",
    "prediction",
    "shap_explain",
    "evidence_retrieve",
    "claim_check",
    "build_follow_up_query",
    "source_verify",
    "confidence_gate",
    "report_generate",
    "verify",
    "memory_proposal",
    "respond",
  ];
  for (const requiredNode of requiredNodes) {
    if (!nodeIdSet.has(requiredNode)) {
      errors.push(`required HCC workflow node is missing: ${requiredNode}`);
    }
    if (!spec.protected_nodes.includes(requiredNode)) {
      errors.push(`required HCC workflow node is not protected: ${requiredNode}`);
    }
  }

  const missingFeatureEdge = spec.edges.find(
    (edge) =>
      edge.from === "feature_check" &&
      edge.to === "respond" &&
      edge.type === "branch",
  );
  if (!missingFeatureEdge) {
    errors.push("missing feature clarification branch to respond");
  } else if (missingFeatureEdge.gate_policy.forbid_downstream_tools !== true) {
    errors.push("missing feature clarification branch must forbid downstream tools");
  }
  if (!edgeExists(spec.edges, "feature_check", "history_retrieve", "branch")) {
    errors.push("complete features must hand off to history_retrieve");
  }
  if (!edgeExists(spec.edges, "history_retrieve", "prediction", "flow")) {
    errors.push("prediction must follow history_retrieve");
  }
  if (!edgeExists(spec.edges, "prediction", "shap_explain", "flow")) {
    errors.push("prediction must hand off to shap_explain");
  }
  const claimRetryEdge = spec.edges.find(
    (edge) =>
      edge.from === "claim_check" &&
      edge.to === "build_follow_up_query" &&
      edge.type === "retry",
  );
  if (!claimRetryEdge) {
    errors.push("claim_check must have a bounded evidence retry edge");
  } else {
    if (claimRetryEdge.planner_contract.max_retry !== 2) {
      errors.push("claim_check evidence retry must use max_retry 2");
    }
    if (
      claimRetryEdge.planner_contract.retry_counter !== "evidence_retry_count"
    ) {
      errors.push(
        "claim_check evidence retry must use evidence_retry_count",
      );
    }
    if (claimRetryEdge.planner_contract.on_exhausted !== "confidence_gate") {
      errors.push("claim_check evidence retry must exhaust to confidence_gate");
    }
  }
  if (!edgeExists(spec.edges, "build_follow_up_query", "evidence_retrieve", "retry")) {
    errors.push("follow-up query must retry evidence retrieval");
  }
  if (!edgeExists(spec.edges, "claim_check", "confidence_gate", "branch")) {
    errors.push("exhausted evidence retry must reach confidence_gate");
  }
  if (!edgeExists(spec.edges, "confidence_gate", "report_generate", "branch")) {
    errors.push("report generation must be gated by confidence_gate");
  }
  if (!edgeExists(spec.edges, "verify", "memory_proposal", "branch")) {
    errors.push("memory proposal must follow successful verification");
  }
  if (
    spec.edges.some(
      (edge) =>
        edge.to === "report_generate" &&
        !["confidence_gate", "verify"].includes(edge.from),
    )
  ) {
    errors.push("report_generate has an unsafe inbound edge that bypasses gate or revision");
  }
  if (
    spec.edges.some(
      (edge) =>
        edge.to === "prediction" && edge.from !== "history_retrieve",
    )
  ) {
    errors.push("prediction has an unsafe inbound edge that bypasses feature/history path");
  }
  if (
    spec.edges.some(
      (edge) => edge.to === "memory_proposal" && edge.from !== "verify",
    )
  ) {
    errors.push("memory_proposal has an unsafe inbound edge that bypasses verification");
  }

  const sortedPlannedHandlers = [...plannedHandlers].sort();
  if (sortedPlannedHandlers.length > 0) {
    warnings.push(
      `planned handlers are not implemented yet: ${sortedPlannedHandlers.join(", ")}`,
    );
  }

  return {
    ok: errors.length === 0,
    implementationReady:
      errors.length === 0 && sortedPlannedHandlers.length === 0,
    errors,
    warnings,
    plannedHandlers: sortedPlannedHandlers,
    checkpoints: spec.nodes
      .filter((node) => node.checkpoint)
      .map((node) => node.id),
  };
}

export function loadHccWorkflowSpec(
  filePath = resolve(
    process.cwd(),
    "configs",
    "workflows",
    "hcc_analysis.workflow.json",
  ),
): HccWorkflowSpec {
  const raw: unknown = JSON.parse(readFileSync(filePath, "utf-8"));
  return hccWorkflowSpecSchema.parse(raw);
}
