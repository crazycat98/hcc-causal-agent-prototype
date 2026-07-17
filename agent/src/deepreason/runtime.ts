import {
  assertCompleteFeatures,
  causalFeatureNames,
  type HccFeatures,
  type PartialHccFeatures,
} from "../features.js";
import {
  saveCaseMemoryOutputSchema,
  type ExplanationServiceResponse,
  type FeatureCompletenessOutput,
  type PatientHistoryOutput,
  type PredictionServiceResponse,
  type RetrievalToolOutput,
  type SaveCaseMemoryOutput,
} from "../predictionTypes.js";
import { REPORT_DISCLAIMER, SYNTHETIC_DATA_NOTICE } from "../safety.js";
import {
  createHccDeepReasonEvidenceTools,
  type CheckClaimsOutput,
  type CreateMemoryProposalOutput,
  type HccClaimEvidence,
  type HccEvidenceItem,
  type HccGateDecision,
  type HccRequiredClaim,
  type VerifyEvidenceSourcesOutput,
} from "./evidence.js";
import {
  createHccDeepReasonToolAdapters,
  type HccDeepReasonToolAdapters,
} from "./toolAdapters.js";
import type {
  HccDeepReasonHandlerResult,
  HccEvidenceDraft,
} from "./types.js";

type WorkflowTraceStatus = "completed" | "blocked" | "skipped";

export type HccWorkflowTraceItem = {
  nodeId: string;
  agentName: string;
  handlerName: string;
  status: WorkflowTraceStatus;
  startedAt: string;
  completedAt: string;
  durationMs: number;
  outputSummary: string;
};

export type HccAgentTraceItem = HccWorkflowTraceItem & {
  toolCallId?: string;
  input?: unknown;
};

export type HccReportVerificationResult = {
  nodeId: "verify";
  agentName: "medical_safety_reviewer";
  status: "passed" | "failed" | "skipped";
  passed: boolean;
  checkedAt: string;
  revisionCount: number;
  checkedRules: string[];
  failures: string[];
  safetyNotice: typeof SYNTHETIC_DATA_NOTICE;
  disclaimer: typeof REPORT_DISCLAIMER;
};

export type RunHccDeepReasonWorkflowOptions = {
  sessionId?: string;
  patientId?: string;
  features: PartialHccFeatures;
  predictionEndpoint?: string;
  explanationEndpoint?: string;
  memoryDir?: string;
  userInstruction?: string;
  approvedBy?: string;
  requiredClaims?: HccRequiredClaim[];
  maxEvidenceRetry?: number;
  topK?: number;
  topN?: number;
};

export type HccDeepReasonAnalysisOutputs = {
  completeness?: FeatureCompletenessOutput;
  history?: PatientHistoryOutput;
  prediction?: PredictionServiceResponse;
  explanation?: ExplanationServiceResponse;
  evidence?: RetrievalToolOutput;
  memory?: SaveCaseMemoryOutput;
};

export type HccDeepReasonWorkflowResult = {
  safetyNotice: typeof SYNTHETIC_DATA_NOTICE;
  disclaimer: typeof REPORT_DISCLAIMER;
  text: string;
  finishReason: "stop" | "clarification_required" | "blocked";
  steps: number;
  toolCalls: Array<{
    toolName: string;
    toolCallId: string;
    input: unknown;
  }>;
  toolResults: Array<{
    toolName: string;
    toolCallId: string;
    output: unknown;
  }>;
  trace: Array<Record<string, unknown>>;
  analysis: HccDeepReasonAnalysisOutputs;
  workflow_trace: HccWorkflowTraceItem[];
  agent_trace: HccAgentTraceItem[];
  claim_evidence_map: HccClaimEvidence[];
  evidence_items: HccEvidenceItem[];
  gate_decisions: HccGateDecision[];
  retry_count: number;
  memory_proposal?: CreateMemoryProposalOutput;
  verification_result: HccReportVerificationResult;
  source_verification?: VerifyEvidenceSourcesOutput;
  deepreason: {
    workflowTrace: HccWorkflowTraceItem[];
    agentTrace: HccAgentTraceItem[];
    claimEvidenceMap: HccClaimEvidence[];
    evidenceItems: HccEvidenceItem[];
    sourceVerification?: VerifyEvidenceSourcesOutput;
    gateDecision?: HccGateDecision;
    gateDecisions: HccGateDecision[];
    retryCount: number;
    memoryProposal?: CreateMemoryProposalOutput;
    verificationResult: HccReportVerificationResult;
  };
};

function nowIso(): string {
  return new Date().toISOString();
}

function syntheticMissingFeatureText(
  completeness: FeatureCompletenessOutput,
): string {
  return [
    "# 合成病例 DeepReason 澄清请求",
    "",
    `安全声明：${SYNTHETIC_DATA_NOTICE}`,
    "",
    "当前合成特征尚不完整，DeepReason 工作流已停在 feature_check 节点，未调用预测、SHAP、检索或长期记忆写入工具。",
    "",
    "## 缺少以下合成特征",
    ...completeness.missingFeatures.map((feature) => `- ${feature}`),
    "",
    "请补充上述字段后再次提交。系统会先合并会话内已获得的合成特征，不会重复要求已经提供的字段。",
    "",
    REPORT_DISCLAIMER,
  ].join("\n");
}

function blockedText(gateDecision?: HccGateDecision): string {
  return [
    "# 合成病例 DeepReason 受限响应",
    "",
    `安全声明：${SYNTHETIC_DATA_NOTICE}`,
    "",
    "## Gate 未放行报告生成",
    ...(gateDecision?.reasons.length
      ? gateDecision.reasons.map((reason) => `- ${reason}`)
      : ["- 当前工作流没有足够证据生成结构化报告。"]),
    "",
    "系统不会在缺少确定性 Tool 输出或证据不足时补写预测概率、SHAP 值、引用或医学诊断结论。",
    "",
    REPORT_DISCLAIMER,
  ].join("\n");
}

function summarizeOutput(output: unknown): string {
  if (!output || typeof output !== "object") {
    return "completed";
  }
  if ("complete" in output && "missingFeatures" in output) {
    const completeness = output as FeatureCompletenessOutput;
    return completeness.complete
      ? "all 10 synthetic causal candidate features available"
      : `missing ${completeness.missingFeatures.length} feature(s): ${completeness.missingFeatures.join(", ")}`;
  }
  if ("hasHistory" in output && "recordCount" in output) {
    const history = output as PatientHistoryOutput;
    return history.hasHistory
      ? `found ${history.recordCount} synthetic historical record(s)`
      : "no synthetic case history found";
  }
  if ("prediction" in output && "model" in output) {
    const prediction = output as PredictionServiceResponse;
    return `${prediction.prediction.label}, high-grade probability ${prediction.prediction.probability_high_grade}`;
  }
  if ("shap" in output) {
    const explanation = output as ExplanationServiceResponse;
    return `${explanation.shap.method}, ${explanation.shap.top_features.length} top feature(s)`;
  }
  if ("results" in output && "confidence" in output) {
    const retrieval = output as RetrievalToolOutput;
    return `${retrieval.results.length} retrieved snippet(s), confidence ${retrieval.confidence}`;
  }
  if ("unsupportedClaims" in output && "claimEvidenceMap" in output) {
    const checked = output as CheckClaimsOutput;
    return `${checked.claimEvidenceMap.length} claim(s), ${checked.unsupportedClaims.length} unsupported`;
  }
  if ("valid" in output && "checkedEvidenceCount" in output) {
    const source = output as VerifyEvidenceSourcesOutput;
    return `source verification ${source.valid ? "valid" : "invalid"}, ${source.checkedEvidenceCount} evidence item(s)`;
  }
  if ("status" in output && "permittedClaimIds" in output) {
    const gate = output as HccGateDecision;
    return `gate ${gate.status}, ${gate.permittedClaimIds.length} permitted claim(s)`;
  }
  if ("status" in output && "proposalId" in output) {
    const proposal = output as CreateMemoryProposalOutput;
    return `memory proposal ${proposal.status}, applied=${proposal.applied}`;
  }
  return "completed";
}

function builtinTrace(
  nodeId: string,
  agentName: string,
  handlerName: string,
  status: WorkflowTraceStatus,
  outputSummary: string,
): HccWorkflowTraceItem {
  const timestamp = nowIso();
  return {
    nodeId,
    agentName,
    handlerName,
    status,
    startedAt: timestamp,
    completedAt: timestamp,
    durationMs: 0,
    outputSummary,
  };
}

function uniqueEvidenceDrafts(drafts: HccEvidenceDraft[]): HccEvidenceDraft[] {
  const byId = new Map<string, HccEvidenceDraft>();
  for (const draft of drafts) {
    if (!byId.has(draft.evidenceId)) {
      byId.set(draft.evidenceId, draft);
    }
  }
  return [...byId.values()];
}

function shapSnapshot(explanation: ExplanationServiceResponse) {
  return {
    top_features: explanation.shap.top_features.map((feature) => ({
      feature: feature.feature,
      shap_value: feature.shap_value,
      direction: feature.direction,
      trust_level: feature.trust_level,
    })),
    high_trust_features: explanation.shap.high_trust_features,
    statistical_only_features: explanation.shap.statistical_only_features,
  };
}

function retrievalSnapshot(retrieval: RetrievalToolOutput) {
  return {
    confidence: retrieval.confidence,
    evidenceSufficient: retrieval.evidenceSufficient,
    evidenceIds: retrieval.results.map((hit) => hit.id),
  };
}

function buildInitialRetrievalQuery(
  prediction: PredictionServiceResponse,
  explanation: ExplanationServiceResponse,
): { query: string; featureNames: string[] } {
  const featureNames = explanation.shap.top_features
    .map((feature) => feature.feature)
    .slice(0, 5);
  const query = [
    "HCC pathology grade public background",
    prediction.prediction.label,
    ...featureNames,
  ].join(" ");
  return { query, featureNames };
}

function compareHistoryWithoutWrite(options: {
  history?: PatientHistoryOutput;
  features: HccFeatures;
  prediction: PredictionServiceResponse["prediction"];
  proposal?: CreateMemoryProposalOutput;
}): SaveCaseMemoryOutput {
  const latest = options.history?.latestRecord;
  const proposalReason =
    options.proposal?.status === "pending_approval"
      ? "DeepReason 已生成长期记忆写入提案；因未提供 approved_by，本次未写入病例记忆。"
      : options.proposal?.status === "rejected"
        ? `DeepReason 记忆写入提案被拒绝：${options.proposal.blockedReasons.join("；") || "未满足写入条件"}。`
        : "DeepReason 未执行长期病例记忆写入。";

  if (!latest) {
    return saveCaseMemoryOutputSchema.parse({
      saved: false,
      recordCount: options.history?.recordCount ?? 0,
      comparison: {
        hasPrevious: false,
        changedFeatures: [],
        summary: `未找到该 patient_id 的历史分析记录。${proposalReason}`,
      },
      safetyNotice: SYNTHETIC_DATA_NOTICE,
    });
  }

  const changedFeatures = causalFeatureNames
    .map((feature) => {
      const previous = latest.features[feature];
      const current = options.features[feature];
      return {
        feature,
        previous,
        current,
        delta: current - previous,
      };
    })
    .filter((item) => Math.abs(item.delta) > 1e-9)
    .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta))
    .slice(0, 5);
  const probabilityDelta =
    options.prediction.probability_high_grade -
    latest.prediction.probability_high_grade;
  const labelChanged = options.prediction.label !== latest.prediction.label;
  const direction =
    probabilityDelta > 0.001
      ? "升高"
      : probabilityDelta < -0.001
        ? "降低"
        : "基本持平";

  return saveCaseMemoryOutputSchema.parse({
    saved: false,
    previousRecord: latest,
    recordCount: options.history?.recordCount ?? 0,
    comparison: {
      hasPrevious: true,
      previousTimestamp: latest.timestamp,
      probabilityDelta,
      labelChanged,
      changedFeatures,
      summary: `较上次合成分析，高分级预测概率${direction} ${Math.abs(probabilityDelta * 100).toFixed(1)} 个百分点；预测标签${labelChanged ? "发生变化" : "未变化"}。${proposalReason}`,
    },
    safetyNotice: SYNTHETIC_DATA_NOTICE,
  });
}

function verifyReport(options: {
  reportMarkdown?: string;
  gateDecision?: HccGateDecision;
  claimEvidenceMap: HccClaimEvidence[];
}): HccReportVerificationResult {
  const failures: string[] = [];
  if (!options.reportMarkdown) {
    failures.push("report markdown is missing.");
  }
  if (options.reportMarkdown && !options.reportMarkdown.includes(REPORT_DISCLAIMER)) {
    failures.push("mandatory disclaimer is missing from report markdown.");
  }
  if (!options.gateDecision) {
    failures.push("gate decision is missing.");
  }
  if (
    options.gateDecision &&
    ["deny", "interrupt"].includes(options.gateDecision.status)
  ) {
    failures.push(`gate status ${options.gateDecision.status} does not permit report release.`);
  }
  const unsupportedClaimIds = new Set(
    options.claimEvidenceMap
      .filter((claim) => claim.supportStatus !== "supported")
      .map((claim) => claim.claimId),
  );
  const leakedUnsupported = options.gateDecision?.permittedClaimIds.filter((claimId) =>
    unsupportedClaimIds.has(claimId),
  ) ?? [];
  if (leakedUnsupported.length > 0) {
    failures.push(
      `unsupported claims leaked into permitted set: ${leakedUnsupported.join(", ")}.`,
    );
  }

  return {
    nodeId: "verify",
    agentName: "medical_safety_reviewer",
    status: failures.length === 0 ? "passed" : "failed",
    passed: failures.length === 0,
    checkedAt: nowIso(),
    revisionCount: 0,
    checkedRules: [
      "report must include mandatory disclaimer",
      "report must have a Gate decision",
      "Gate must be allow or limited for report release",
      "unsupported Claims cannot appear in permittedClaimIds",
    ],
    failures,
    safetyNotice: SYNTHETIC_DATA_NOTICE,
    disclaimer: REPORT_DISCLAIMER,
  };
}

export async function runHccDeepReasonWorkflow(
  options: RunHccDeepReasonWorkflowOptions,
): Promise<HccDeepReasonWorkflowResult> {
  const sessionId = options.sessionId ?? "demo-session";
  const maxEvidenceRetry = options.maxEvidenceRetry ?? 2;
  const topK = options.topK ?? 5;
  const topN = options.topN ?? 5;
  const adapters = createHccDeepReasonToolAdapters({
    predictionEndpoint: options.predictionEndpoint,
    explanationEndpoint: options.explanationEndpoint,
    memoryDir: options.memoryDir,
  });
  const evidenceTools = createHccDeepReasonEvidenceTools();

  const intakeTrace = builtinTrace(
    "intake",
    "hcc_coordinator",
    "intake",
    "completed",
    "normalized synthetic HCC analysis request",
  );
  const workflowTrace: HccWorkflowTraceItem[] = [intakeTrace];
  const agentTrace: HccAgentTraceItem[] = [intakeTrace];
  const trace: Array<Record<string, unknown>> = [];
  const toolCalls: HccDeepReasonWorkflowResult["toolCalls"] = [];
  const toolResults: HccDeepReasonWorkflowResult["toolResults"] = [];
  const evidenceDrafts: HccEvidenceDraft[] = [];
  let toolCallCounter = 0;

  function recordToolResult<TOutput>(
    result: HccDeepReasonHandlerResult<TOutput>,
    input: unknown,
  ) {
    const toolCallId = `dr-${String(toolCallCounter + 1).padStart(2, "0")}-${result.handlerName}`;
    toolCallCounter += 1;
    const workflowItem: HccWorkflowTraceItem = {
      nodeId: result.nodeId,
      agentName: result.agentName,
      handlerName: result.handlerName,
      status: "completed",
      startedAt: result.trace.startedAt,
      completedAt: result.trace.completedAt,
      durationMs: result.trace.durationMs,
      outputSummary: summarizeOutput(result.output),
    };
    workflowTrace.push(workflowItem);
    agentTrace.push({
      ...workflowItem,
      toolCallId,
      input,
    });
    toolCalls.push({
      toolName: result.handlerName,
      toolCallId,
      input,
    });
    toolResults.push({
      toolName: result.handlerName,
      toolCallId,
      output: result.output,
    });
    trace.push({
      event: "tool_start",
      toolName: result.handlerName,
      toolCallId,
      input,
    });
    trace.push({
      event: "tool_end",
      toolName: result.handlerName,
      toolCallId,
      toolExecutionMs: result.trace.durationMs,
      output: result.output,
    });
    evidenceDrafts.push(...result.evidenceDrafts);
  }

  const featureInput = {
    sessionId,
    patientId: options.patientId,
    features: options.features,
  };
  const completenessResult = await adapters.checkFeatureCompleteness(featureInput);
  recordToolResult(completenessResult, featureInput);
  const completeness = completenessResult.output;

  if (!completeness.complete) {
    workflowTrace.push(
      builtinTrace(
        "respond",
        "hcc_coordinator",
        "respond",
        "blocked",
        "requesting missing synthetic features; downstream tools were not called",
      ),
    );
    agentTrace.push(workflowTrace.at(-1)!);
    const verificationResult: HccReportVerificationResult = {
      nodeId: "verify",
      agentName: "medical_safety_reviewer",
      status: "skipped",
      passed: false,
      checkedAt: nowIso(),
      revisionCount: 0,
      checkedRules: [],
      failures: ["feature_check reported missing synthetic features."],
      safetyNotice: SYNTHETIC_DATA_NOTICE,
      disclaimer: REPORT_DISCLAIMER,
    };
    const deepreason = {
      workflowTrace,
      agentTrace,
      claimEvidenceMap: [],
      evidenceItems: [],
      gateDecisions: [],
      retryCount: 0,
      verificationResult,
    };
    return {
      safetyNotice: SYNTHETIC_DATA_NOTICE,
      disclaimer: REPORT_DISCLAIMER,
      text: syntheticMissingFeatureText(completeness),
      finishReason: "clarification_required",
      steps: workflowTrace.length,
      toolCalls,
      toolResults,
      trace,
      analysis: {
        completeness,
      },
      workflow_trace: workflowTrace,
      agent_trace: agentTrace,
      claim_evidence_map: [],
      evidence_items: [],
      gate_decisions: [],
      retry_count: 0,
      verification_result: verificationResult,
      deepreason,
    };
  }

  const features = assertCompleteFeatures(completeness.features);

  const historyInput = { patientId: options.patientId };
  const historyResult = await adapters.getPatientHistory(historyInput);
  recordToolResult(historyResult, historyInput);
  const history = historyResult.output;

  const predictionInput = {
    patientId: options.patientId,
    features,
  };
  const predictionResult = await adapters.predictHccGrade(predictionInput);
  recordToolResult(predictionResult, predictionInput);
  const prediction = predictionResult.output;

  const explanationInput = {
    patientId: options.patientId,
    features,
    topN,
  };
  const explanationResult = await adapters.explainPredictionWithShap(
    explanationInput,
  );
  recordToolResult(explanationResult, explanationInput);
  const explanation = explanationResult.output;

  const retrievalSeed = buildInitialRetrievalQuery(prediction, explanation);
  const retrievalInput = {
    query: retrievalSeed.query,
    featureNames: retrievalSeed.featureNames,
    topK,
  };
  const retrievalResult = await adapters.retrieveMedicalEvidence(retrievalInput);
  recordToolResult(retrievalResult, retrievalInput);
  let retrieval = retrievalResult.output;

  let checked: CheckClaimsOutput | undefined;
  let retryCount = 0;
  let evidenceRetryExhausted = false;
  let baseQuery = retrievalSeed.query;

  while (true) {
    const checkInput = {
      evidenceDrafts: uniqueEvidenceDrafts(evidenceDrafts),
      requiredClaims: options.requiredClaims ?? [],
      evidenceRetryCount: retryCount,
    };
    const checkedResult = await evidenceTools.checkClaims(checkInput);
    recordToolResult(checkedResult, checkInput);
    checked = checkedResult.output;

    if (checked.unsupportedClaims.length === 0) {
      break;
    }

    const followUpInput = {
      unsupportedClaims: checked.unsupportedClaims,
      evidenceRetryCount: retryCount,
      maxRetry: maxEvidenceRetry,
      baseQuery,
    };
    const followUpResult = await evidenceTools.buildFollowUpQuery(followUpInput);
    recordToolResult(followUpResult, followUpInput);
    retryCount = followUpResult.output.evidenceRetryCount;

    if (!followUpResult.output.shouldRetry) {
      evidenceRetryExhausted = true;
      break;
    }

    baseQuery = followUpResult.output.followUpQuery;
    const followUpRetrievalInput = {
      query: followUpResult.output.followUpQuery,
      featureNames: retrievalSeed.featureNames,
      topK,
    };
    const followUpRetrievalResult = await adapters.retrieveMedicalEvidence(
      followUpRetrievalInput,
    );
    recordToolResult(followUpRetrievalResult, followUpRetrievalInput);
    retrieval = followUpRetrievalResult.output;
  }

  const checkedOutput = checked;
  if (!checkedOutput) {
    throw new Error("DeepReason runtime failed to produce Claim-Evidence output.");
  }

  const sourceInput = {
    evidenceItems: checkedOutput.evidenceItems,
    claimEvidenceMap: checkedOutput.claimEvidenceMap,
  };
  const sourceResult = await evidenceTools.verifyEvidenceSources(sourceInput);
  recordToolResult(sourceResult, sourceInput);
  const sourceVerification = sourceResult.output;

  const gateInput = {
    evidenceItems: checkedOutput.evidenceItems,
    claimEvidenceMap: checkedOutput.claimEvidenceMap,
    sourceVerification,
    disclaimerIncluded: true,
    evidenceRetryExhausted,
  };
  const gateResult = await evidenceTools.evaluateMedicalConfidenceGate(gateInput);
  recordToolResult(gateResult, gateInput);
  const gateDecision = gateResult.output;

  let reportMarkdown: string | undefined;
  if (gateDecision.status === "allow" || gateDecision.status === "limited") {
    const reportInput = {
      gateDecision,
      evidenceItems: checkedOutput.evidenceItems,
      claimEvidenceMap: checkedOutput.claimEvidenceMap,
      sourceVerification,
      patientId: options.patientId,
    };
    const reportResult = await evidenceTools.generateHccReport(reportInput);
    recordToolResult(reportResult, reportInput);
    reportMarkdown = reportResult.output.markdown;
  }

  const verificationResult = verifyReport({
    reportMarkdown,
    gateDecision,
    claimEvidenceMap: checkedOutput.claimEvidenceMap,
  });
  workflowTrace.push(
    builtinTrace(
      "verify",
      "medical_safety_reviewer",
      "verify",
      verificationResult.passed ? "completed" : "blocked",
      verificationResult.passed
        ? "report, Gate, disclaimer, and Claim-Evidence bindings verified"
        : verificationResult.failures.join("; "),
    ),
  );
  agentTrace.push(workflowTrace.at(-1)!);

  let memoryProposal: CreateMemoryProposalOutput | undefined;
  if (verificationResult.passed) {
    const memoryInput = {
      patientId: options.patientId,
      sessionId,
      features,
      prediction: prediction.prediction,
      shap: shapSnapshot(explanation),
      retrieval: retrievalSnapshot(retrieval),
      gateDecision,
      evidenceItems: checkedOutput.evidenceItems,
      sourceVerification,
      approvedBy: options.approvedBy,
      memoryDir: options.memoryDir,
    };
    const memoryResult = await evidenceTools.createMemoryProposal(memoryInput);
    recordToolResult(memoryResult, memoryInput);
    memoryProposal = memoryResult.output;
  }

  const memoryCompatibility =
    memoryProposal?.appliedResult ??
    compareHistoryWithoutWrite({
      history,
      features,
      prediction: prediction.prediction,
      proposal: memoryProposal,
    });

  workflowTrace.push(
    builtinTrace(
      "respond",
      "hcc_coordinator",
      "respond",
      verificationResult.passed ? "completed" : "blocked",
      verificationResult.passed
        ? "returning DeepReason HCC report with trace, Gate, Evidence, and Memory Proposal"
        : "returning blocked response with Gate reasons and evidence gaps",
    ),
  );
  agentTrace.push(workflowTrace.at(-1)!);

  const gateDecisions = [gateDecision];
  const deepreason = {
    workflowTrace,
    agentTrace,
    claimEvidenceMap: checkedOutput.claimEvidenceMap,
    evidenceItems: checkedOutput.evidenceItems,
    sourceVerification,
    gateDecision,
    gateDecisions,
    retryCount,
    memoryProposal,
    verificationResult,
  };

  return {
    safetyNotice: SYNTHETIC_DATA_NOTICE,
    disclaimer: REPORT_DISCLAIMER,
    text: reportMarkdown ?? blockedText(gateDecision),
    finishReason: verificationResult.passed ? "stop" : "blocked",
    steps: workflowTrace.length,
    toolCalls,
    toolResults,
    trace,
    analysis: {
      completeness,
      history,
      prediction,
      explanation,
      evidence: retrieval,
      memory: memoryCompatibility,
    },
    workflow_trace: workflowTrace,
    agent_trace: agentTrace,
    claim_evidence_map: checkedOutput.claimEvidenceMap,
    evidence_items: checkedOutput.evidenceItems,
    gate_decisions: gateDecisions,
    retry_count: retryCount,
    memory_proposal: memoryProposal,
    verification_result: verificationResult,
    source_verification: sourceVerification,
    deepreason,
  };
}
