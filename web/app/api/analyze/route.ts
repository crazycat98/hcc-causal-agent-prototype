import { NextResponse } from "next/server";
import { z } from "zod";
import { partialHccFeatureSchema } from "../../../../agent/src/features.js";
import { runHccAgent } from "../../../../agent/src/runner.js";
import {
  REPORT_DISCLAIMER,
  SYNTHETIC_DATA_NOTICE,
} from "../../../../agent/src/safety.js";

export const runtime = "nodejs";

const analyzeRequestSchema = z
  .object({
    sessionId: z.string().min(1),
    patientId: z.string().min(1).optional(),
    features: partialHccFeatureSchema,
    userInstruction: z.string().max(1200).optional(),
  })
  .strict();

function toolOutput<T = unknown>(
  toolResults: Array<{ toolName: string; output: unknown }>,
  toolName: string,
): T | undefined {
  return toolResults.find((result) => result.toolName === toolName)?.output as
    | T
    | undefined;
}

export async function POST(request: Request) {
  try {
    const body = analyzeRequestSchema.parse(await request.json());

    // 安全边界：演示用合成数据，非真实患者数据；非临床诊断依据。
    // The web layer only displays Agent/Tool outputs; it never computes
    // prediction probabilities, SHAP values, citations, or memory trends itself.
    const agentResult = await runHccAgent({
      sessionId: body.sessionId,
      patientId: body.patientId,
      features: body.features,
      userInstruction: body.userInstruction,
      memoryDir: process.env.AGENT_MEMORY_DIR,
    });

    return NextResponse.json({
      safetyNotice: SYNTHETIC_DATA_NOTICE,
      disclaimer: REPORT_DISCLAIMER,
      text: agentResult.text,
      finishReason: agentResult.finishReason,
      steps: agentResult.steps,
      toolCalls: agentResult.toolCalls,
      trace: agentResult.trace,
      analysis: {
        completeness: toolOutput(
          agentResult.toolResults,
          "checkFeatureCompleteness",
        ),
        history: toolOutput(agentResult.toolResults, "getPatientHistory"),
        prediction: toolOutput(agentResult.toolResults, "predictHccGrade"),
        explanation: toolOutput(
          agentResult.toolResults,
          "explainPredictionWithShap",
        ),
        evidence: toolOutput(agentResult.toolResults, "retrieveMedicalEvidence"),
        memory: toolOutput(agentResult.toolResults, "saveCaseMemory"),
      },
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown Agent execution error";
    const status = message.includes("fetch failed") ? 502 : 400;

    return NextResponse.json(
      {
        safetyNotice: SYNTHETIC_DATA_NOTICE,
        disclaimer: REPORT_DISCLAIMER,
        error: message,
        hint:
          status === 502
            ? "ML 服务不可用。请先在另一个终端运行 npm.cmd run m1:serve。"
            : "请求数据未通过 schema 校验，请检查字段范围和类型。",
      },
      { status },
    );
  }
}
