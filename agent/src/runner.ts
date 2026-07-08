import { generateText, isStepCount } from "ai";
import type { LanguageModel } from "ai";
import type { PartialHccFeatures } from "./features.js";
import { createMockHccAgentModel } from "./mockModel.js";
import { HCC_AGENT_SYSTEM_PROMPT } from "./systemPrompt.js";
import { createHccAgentTools } from "./tools.js";

export type RunHccAgentOptions = {
  sessionId?: string;
  patientId?: string;
  features: PartialHccFeatures;
  model?: LanguageModel;
  predictionEndpoint?: string;
  explanationEndpoint?: string;
  memoryDir?: string;
  userInstruction?: string;
};

export async function runHccAgent(options: RunHccAgentOptions) {
  const sessionId = options.sessionId ?? "demo-session";
  const tools = createHccAgentTools({
    predictionEndpoint: options.predictionEndpoint,
    explanationEndpoint: options.explanationEndpoint,
    memoryDir: options.memoryDir,
  });
  const model =
    options.model ??
    createMockHccAgentModel({
      sessionId,
      patientId: options.patientId,
      features: options.features,
    });
  const trace: Array<Record<string, unknown>> = [];

  const result = await generateText({
    model,
    tools,
    system: HCC_AGENT_SYSTEM_PROMPT,
    prompt: [
      "请基于以下演示用合成数据完成 HCC 病理分级预测流程。",
      "必须先合并会话内特征记忆并检查 10 个因果候选特征是否完整。",
      "特征完整时，依次调用历史病例记忆、预测 Tool、SHAP 解释 Tool、医学依据检索 Tool、病例记忆写入 Tool。",
      "不要自行编造预测概率、SHAP 值、医学机制、引用、记忆记录或诊断结论。",
      options.userInstruction ? `用户补充请求：${options.userInstruction}` : undefined,
      "",
      JSON.stringify(
        {
          sessionId,
          patientId: options.patientId,
          features: options.features,
        },
        null,
        2,
      ),
    ].filter(Boolean).join("\n"),
    stopWhen: isStepCount(10),
    temperature: 0,
    onToolExecutionStart({ toolCall }) {
      trace.push({
        event: "tool_start",
        toolName: toolCall.toolName,
        toolCallId: toolCall.toolCallId,
        input: toolCall.input,
      });
    },
    onToolExecutionEnd({ toolCall, toolOutput, toolExecutionMs }) {
      trace.push({
        event: "tool_end",
        toolName: toolCall.toolName,
        toolCallId: toolCall.toolCallId,
        toolExecutionMs,
        output: toolOutput,
      });
    },
  });

  return {
    text: result.text,
    finishReason: result.finishReason,
    steps: result.steps.length,
    toolCalls: result.toolCalls.map((call) => ({
      toolName: call.toolName,
      toolCallId: call.toolCallId,
      input: call.input,
    })),
    toolResults: result.toolResults.map((toolResult) => ({
      toolName: toolResult.toolName,
      toolCallId: toolResult.toolCallId,
      output: toolResult.output,
    })),
    trace,
  };
}
