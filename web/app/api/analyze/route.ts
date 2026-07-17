import { NextResponse } from "next/server";
import { z } from "zod";
import { runHccDeepReasonWorkflow } from "../../../../agent/src/deepreason/index.js";
import { partialHccFeatureSchema } from "../../../../agent/src/features.js";
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
    approvedBy: z.string().min(1).max(120).optional(),
  })
  .strict();

export async function POST(request: Request) {
  try {
    const body = analyzeRequestSchema.parse(await request.json());

    // 安全边界：演示用合成数据，非真实患者数据；非临床诊断依据。
    // The web layer only displays Agent/Tool outputs; it never computes
    // prediction probabilities, SHAP values, citations, or memory trends itself.
    const workflowResult = await runHccDeepReasonWorkflow({
      sessionId: body.sessionId,
      patientId: body.patientId,
      features: body.features,
      userInstruction: body.userInstruction,
      approvedBy: body.approvedBy,
      memoryDir: process.env.AGENT_MEMORY_DIR,
    });

    return NextResponse.json(workflowResult);
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
