import type { LanguageModelV4CallOptions } from "@ai-sdk/provider";
import { MockLanguageModelV4 } from "ai/test";
import type { PartialHccFeatures } from "./features.js";
import { listMissingFeatures } from "./features.js";
import type {
  ExplanationServiceResponse,
  FeatureCompletenessOutput,
  PatientHistoryOutput,
  PredictionServiceResponse,
  RetrievalToolOutput,
  SaveCaseMemoryOutput,
} from "./predictionTypes.js";
import { formatPredictionReport } from "./report.js";
import { REPORT_DISCLAIMER, SYNTHETIC_DATA_NOTICE } from "./safety.js";

type MockAgentInput = {
  sessionId: string;
  patientId?: string;
  features: PartialHccFeatures;
};

const usage = {
  inputTokens: {
    total: 50,
    noCache: 50,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: {
    total: 50,
    text: 50,
    reasoning: undefined,
  },
};

function getToolResult(options: LanguageModelV4CallOptions, toolName: string) {
  for (const message of options.prompt) {
    if (message.role !== "tool") {
      continue;
    }

    for (const part of message.content) {
      if (part.type === "tool-result" && part.toolName === toolName) {
        return part.output;
      }
    }
  }

  return undefined;
}

function jsonToolOutputValue(output: unknown): unknown {
  if (
    output &&
    typeof output === "object" &&
    "type" in output &&
    (output as { type?: unknown }).type === "json"
  ) {
    return (output as { value?: unknown }).value;
  }

  return undefined;
}

function retrievalQuery(explanation: ExplanationServiceResponse): string {
  const topFeatures = explanation.shap.top_features
    .map((item) => item.feature)
    .join(" ");
  return [
    "HCC 病理分级 预测解释",
    explanation.prediction.label,
    topFeatures,
    "AFP portal vein radiomics entropy GLCM tumor size liver function",
  ].join(" ");
}

function completedFeatures(checkResult: FeatureCompletenessOutput) {
  return checkResult.features as Required<FeatureCompletenessOutput["features"]>;
}

export function createMockHccAgentModel(input: MockAgentInput) {
  return new MockLanguageModelV4({
    provider: "local-m5-demo",
    modelId: "mock-hcc-agent",
    doGenerate: async (options) => {
      const checkResult = jsonToolOutputValue(
        getToolResult(options, "checkFeatureCompleteness"),
      ) as FeatureCompletenessOutput | undefined;

      const historyResult = jsonToolOutputValue(
        getToolResult(options, "getPatientHistory"),
      ) as PatientHistoryOutput | undefined;

      const predictionResult = jsonToolOutputValue(
        getToolResult(options, "predictHccGrade"),
      ) as PredictionServiceResponse | undefined;

      const explanationResult = jsonToolOutputValue(
        getToolResult(options, "explainPredictionWithShap"),
      ) as ExplanationServiceResponse | undefined;

      const retrievalResult = jsonToolOutputValue(
        getToolResult(options, "retrieveMedicalEvidence"),
      ) as RetrievalToolOutput | undefined;

      const saveMemoryResult = jsonToolOutputValue(
        getToolResult(options, "saveCaseMemory"),
      ) as SaveCaseMemoryOutput | undefined;

      if (!checkResult) {
        return {
          content: [
            {
              type: "tool-call",
              toolCallId: "call_check_features",
              toolName: "checkFeatureCompleteness",
              input: JSON.stringify({
                sessionId: input.sessionId,
                patientId: input.patientId,
                features: input.features,
              }),
            },
          ],
          finishReason: { unified: "tool-calls", raw: "tool_calls" },
          usage,
          warnings: [],
        };
      }

      if (!checkResult.complete) {
        const missingFeatures =
          checkResult.missingFeatures ?? listMissingFeatures(input.features);
        return {
          content: [
            {
              type: "text",
              text: [
                `安全声明：${SYNTHETIC_DATA_NOTICE}`,
                "",
                "当前无法进行预测，因为缺少以下合成特征：",
                missingFeatures.map((name) => `- ${name}`).join("\n"),
                "",
                "请补充缺失字段后再运行预测；已提供字段已写入会话内工作记忆，不会重复询问。",
                "",
                REPORT_DISCLAIMER,
              ].join("\n"),
            },
          ],
          finishReason: { unified: "stop", raw: "stop" },
          usage,
          warnings: [],
        };
      }

      const features = completedFeatures(checkResult);

      if (!historyResult) {
        return {
          content: [
            {
              type: "tool-call",
              toolCallId: "call_get_history",
              toolName: "getPatientHistory",
              input: JSON.stringify({
                patientId: input.patientId,
              }),
            },
          ],
          finishReason: { unified: "tool-calls", raw: "tool_calls" },
          usage,
          warnings: [],
        };
      }

      if (!predictionResult) {
        return {
          content: [
            {
              type: "tool-call",
              toolCallId: "call_predict_grade",
              toolName: "predictHccGrade",
              input: JSON.stringify({
                patientId: input.patientId,
                features,
              }),
            },
          ],
          finishReason: { unified: "tool-calls", raw: "tool_calls" },
          usage,
          warnings: [],
        };
      }

      if (!explanationResult) {
        return {
          content: [
            {
              type: "tool-call",
              toolCallId: "call_explain_shap",
              toolName: "explainPredictionWithShap",
              input: JSON.stringify({
                patientId: input.patientId,
                features,
                topN: 5,
              }),
            },
          ],
          finishReason: { unified: "tool-calls", raw: "tool_calls" },
          usage,
          warnings: [],
        };
      }

      if (!retrievalResult) {
        const featureNames = explanationResult.shap.top_features.map(
          (item) => item.feature,
        );
        return {
          content: [
            {
              type: "tool-call",
              toolCallId: "call_retrieve_evidence",
              toolName: "retrieveMedicalEvidence",
              input: JSON.stringify({
                query: retrievalQuery(explanationResult),
                featureNames,
                topK: 5,
              }),
            },
          ],
          finishReason: { unified: "tool-calls", raw: "tool_calls" },
          usage,
          warnings: [],
        };
      }

      if (!saveMemoryResult) {
        return {
          content: [
            {
              type: "tool-call",
              toolCallId: "call_save_case_memory",
              toolName: "saveCaseMemory",
              input: JSON.stringify({
                patientId: input.patientId,
                sessionId: input.sessionId,
                features,
                prediction: predictionResult.prediction,
                shap: {
                  top_features: explanationResult.shap.top_features.map((item) => ({
                    feature: item.feature,
                    shap_value: item.shap_value,
                    direction: item.direction,
                    trust_level: item.trust_level,
                  })),
                  high_trust_features: explanationResult.shap.high_trust_features,
                  statistical_only_features:
                    explanationResult.shap.statistical_only_features,
                },
                retrieval: {
                  confidence: retrievalResult.confidence,
                  evidenceSufficient: retrievalResult.evidenceSufficient,
                  evidenceIds: retrievalResult.results.map((hit) => hit.id),
                },
              }),
            },
          ],
          finishReason: { unified: "tool-calls", raw: "tool_calls" },
          usage,
          warnings: [],
        };
      }

      return {
        content: [
          {
            type: "text",
            text: formatPredictionReport(
              predictionResult,
              explanationResult,
              retrievalResult,
              historyResult,
              saveMemoryResult,
            ),
          },
        ],
        finishReason: { unified: "stop", raw: "stop" },
        usage,
        warnings: [],
      };
    },
  });
}

