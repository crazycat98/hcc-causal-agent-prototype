import type {
  ExplanationServiceResponse,
  PatientHistoryOutput,
  PredictionServiceResponse,
  RetrievalToolOutput,
  SaveCaseMemoryOutput,
} from "./predictionTypes.js";
import { REPORT_DISCLAIMER, SYNTHETIC_DATA_NOTICE } from "./safety.js";

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatSignedPercentPoint(value: number): string {
  const sign = value >= 0 ? "+" : "-";
  return `${sign}${Math.abs(value * 100).toFixed(1)} 个百分点`;
}

function formatShapSection(explanation?: ExplanationServiceResponse): string[] {
  if (!explanation) {
    return [
      "## SHAP 解释",
      "- M3 SHAP Tool 尚未返回结果，因此不输出特征贡献解释。",
    ];
  }

  const featureLines = explanation.shap.top_features.map((item, index) => {
    const signedValue =
      item.shap_value >= 0
        ? `+${item.shap_value.toFixed(4)}`
        : item.shap_value.toFixed(4);
    const direction =
      item.direction === "pushes_toward_high_grade"
        ? "推向高分级预测"
        : "推向低/中分级预测";
    const trust =
      item.trust_level === "high_trust_causal_candidate"
        ? "高可信解释线索"
        : "仅统计相关，谨慎解读";
    return `${index + 1}. ${item.feature}: SHAP=${signedValue}, ${direction}, ${trust}`;
  });

  return [
    "## SHAP Top 特征解释",
    `- method: ${explanation.shap.method}`,
    `- target_class: ${explanation.shap.target_class}`,
    `- base_value: ${explanation.shap.base_value.toFixed(4)}`,
    ...featureLines.map((line) => `- ${line}`),
    "",
    "## 因果-SHAP 一致性校验",
    `- summary: ${explanation.shap.consistency_summary}`,
    `- high_trust_features: ${explanation.shap.high_trust_features.join(", ") || "无"}`,
    `- statistical_only_features: ${explanation.shap.statistical_only_features.join(", ") || "无"}`,
    `- caveat: ${explanation.shap.caveat}`,
  ];
}

function formatEvidenceSection(evidence?: RetrievalToolOutput): string[] {
  if (!evidence) {
    return [
      "## 医学依据检索",
      "- 现有资料不足，无法给出解释。",
    ];
  }

  if (!evidence.evidenceSufficient || evidence.results.length === 0) {
    return [
      "## 医学依据检索",
      `- confidence: ${evidence.confidence}`,
      "- 现有资料不足，无法给出解释。",
    ];
  }

  const citations = evidence.results.map((hit) => {
    return `- [${hit.id}] ${hit.title}: ${hit.paragraph} 来源：${hit.source.label} (${hit.source.url})`;
  });

  return [
    "## 医学依据检索",
    `- method: ${evidence.retrievalMethod}`,
    `- confidence: ${evidence.confidence}`,
    "- 以下背景说明仅基于检索返回段落，不扩展生成未检索到的医学结论。",
    ...citations,
  ];
}

function formatMemorySection(
  history?: PatientHistoryOutput,
  saved?: SaveCaseMemoryOutput,
): string[] {
  if (!history && !saved) {
    return [
      "## 会话与病例记忆",
      "- 未返回记忆 Tool 输出。",
    ];
  }

  const lines = [
    "## 会话与病例记忆",
    `- historical_record_count_before_analysis: ${history?.recordCount ?? 0}`,
    `- case_memory_saved: ${saved?.saved ? "yes" : "no"}`,
    `- record_count_after_analysis: ${saved?.recordCount ?? history?.recordCount ?? 0}`,
  ];

  if (saved?.comparison.hasPrevious) {
    lines.push(`- revisit_trend: ${saved.comparison.summary}`);
    if (typeof saved.comparison.probabilityDelta === "number") {
      lines.push(
        `- probability_delta: ${formatSignedPercentPoint(saved.comparison.probabilityDelta)}`,
      );
    }
    if (saved.comparison.changedFeatures.length > 0) {
      lines.push(
        `- changed_features_top: ${saved.comparison.changedFeatures
          .map((item) => `${item.feature}(${item.previous} -> ${item.current})`)
          .join(", ")}`,
      );
    }
  } else {
    lines.push(`- revisit_trend: ${saved?.comparison.summary ?? "未发现历史记录。"}`);
  }

  return lines;
}

function formatConfidenceSection(
  predictionResult: PredictionServiceResponse,
  evidence?: RetrievalToolOutput,
): string[] {
  const lines = ["## 置信度与澄清状态"];

  if (predictionResult.prediction.uncertain_probability_band) {
    lines.push(
      "- prediction_confidence: 预测概率处于 0.4-0.6 临界区间，建议结合其他检查综合判断。",
    );
  } else {
    lines.push("- prediction_confidence: 预测概率未落入预设临界区间。");
  }

  if (evidence && !evidence.evidenceSufficient) {
    lines.push(
      "- retrieval_confidence: 相关医学资料检索证据不足，该部分解释仅供参考。",
    );
  } else if (evidence) {
    lines.push(`- retrieval_confidence: ${evidence.confidence}`);
  } else {
    lines.push("- retrieval_confidence: 未执行检索。");
  }

  return lines;
}

export function formatPredictionReport(
  predictionResult: PredictionServiceResponse,
  explanation?: ExplanationServiceResponse,
  evidence?: RetrievalToolOutput,
  history?: PatientHistoryOutput,
  savedMemory?: SaveCaseMemoryOutput,
): string {
  const auc =
    typeof predictionResult.model.cv_auc_mean === "number"
      ? predictionResult.model.cv_auc_mean.toFixed(3)
      : "unknown";
  const uncertainty = predictionResult.prediction.uncertain_probability_band
    ? "预测概率位于 0.4-0.6 临界区间，报告需标注不确定。"
    : "预测概率未落入预设临界区间。";

  return [
    "# 合成病例分级预测报告",
    "",
    `安全声明：${SYNTHETIC_DATA_NOTICE}`,
    "",
    "## Tool 输出摘要",
    `- patient_id: ${predictionResult.patient_id ?? "未提供"}`,
    `- model: ${predictionResult.model.type}`,
    `- feature_selection_method: ${predictionResult.model.feature_selection_method ?? "unknown"}`,
    `- cv_auc_mean: ${auc}`,
    "",
    "## 预测结果",
    `- label: ${predictionResult.prediction.label}`,
    `- probability_high_grade: ${formatPercent(predictionResult.prediction.probability_high_grade)}`,
    `- probability_low_or_intermediate: ${formatPercent(predictionResult.prediction.probability_low_or_intermediate)}`,
    `- uncertainty_note: ${uncertainty}`,
    "",
    ...formatConfidenceSection(predictionResult, evidence),
    "",
    ...formatMemorySection(history, savedMemory),
    "",
    ...formatShapSection(explanation),
    "",
    ...formatEvidenceSection(evidence),
    "",
    "## M5 边界",
    "- 会话记忆和病例记忆均为本地 JSON 原型存储，仅保存演示用合成数据。",
    "- 复诊趋势比较的是本原型前后两次合成分析记录，不代表真实病情变化。",
    "- 检索段落用于背景解释，不构成临床诊断或治疗建议。",
    "",
    REPORT_DISCLAIMER,
  ].join("\n");
}

