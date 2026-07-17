import type {
  ExplanationServiceResponse,
  FeatureCompletenessOutput,
  PatientHistoryOutput,
  PredictionServiceResponse,
  RetrievalToolOutput,
  SaveCaseMemoryOutput,
} from "../predictionTypes.js";

export type HccDeepReasonAgentName =
  | "feature_collector"
  | "memory_manager"
  | "prediction_operator"
  | "explanation_operator"
  | "medical_retriever"
  | "claim_checker"
  | "source_verifier"
  | "medical_safety_reviewer"
  | "report_writer";

export type HccDeepReasonNodeId =
  | "feature_check"
  | "history_retrieve"
  | "prediction"
  | "shap_explain"
  | "evidence_retrieve"
  | "claim_check"
  | "build_follow_up_query"
  | "source_verify"
  | "confidence_gate"
  | "report_generate"
  | "memory_proposal"
  | "case_memory_write";

export type HccDeepReasonHandlerName =
  | "checkFeatureCompleteness"
  | "getPatientHistory"
  | "predictHccGrade"
  | "explainPredictionWithShap"
  | "retrieveMedicalEvidence"
  | "checkClaims"
  | "buildFollowUpQuery"
  | "verifyEvidenceSources"
  | "evaluateMedicalConfidenceGate"
  | "generateHccReport"
  | "createMemoryProposal"
  | "saveCaseMemory";

export type HccDeepReasonPermission =
  | "read_session_memory"
  | "read_case_memory"
  | "call_prediction_service"
  | "call_shap_service"
  | "retrieve_public_knowledge"
  | "bind_claim_evidence"
  | "build_follow_up_query"
  | "verify_evidence_sources"
  | "evaluate_medical_confidence_gate"
  | "generate_evidence_report"
  | "create_memory_proposal"
  | "apply_approved_memory_proposal"
  | "write_case_memory";

export type HccDeepReasonHandlerSpec = {
  handlerName: HccDeepReasonHandlerName;
  nodeId: HccDeepReasonNodeId;
  agentName: HccDeepReasonAgentName;
  description: string;
  permissions: HccDeepReasonPermission[];
  deterministic: true;
};

export type HccEvidenceDraft = {
  evidenceId: string;
  sourceType:
    | "session_state"
    | "case_memory"
    | "model_prediction"
    | "model_explanation"
    | "knowledge_base";
  claimType:
    | "feature_state"
    | "history"
    | "model_output"
    | "model_explanation"
    | "medical_background";
  summary: string;
  confidence: number;
  uri?: string;
  locator?: string;
};

export type HccDeepReasonHandlerResult<TOutput> = {
  handlerName: HccDeepReasonHandlerName;
  nodeId: HccDeepReasonNodeId;
  agentName: HccDeepReasonAgentName;
  output: TOutput;
  evidenceDrafts: HccEvidenceDraft[];
  trace: {
    startedAt: string;
    completedAt: string;
    durationMs: number;
  };
};

export type HccDeepReasonToolAdapterOptions = {
  predictionEndpoint?: string;
  explanationEndpoint?: string;
  memoryDir?: string;
};

export type HccDeepReasonAdapterOutputs = {
  checkFeatureCompleteness: FeatureCompletenessOutput;
  getPatientHistory: PatientHistoryOutput;
  predictHccGrade: PredictionServiceResponse;
  explainPredictionWithShap: ExplanationServiceResponse;
  retrieveMedicalEvidence: RetrievalToolOutput;
  saveCaseMemory: SaveCaseMemoryOutput;
};
