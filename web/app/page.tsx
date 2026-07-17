"use client";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ClipboardList,
  Database,
  FileText,
  History,
  Loader2,
  Play,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { FormEvent, useMemo, useState } from "react";

const SAFETY_NOTICE = "演示用合成数据，非真实患者数据；非临床诊断依据。";

type FeatureKey =
  | "tumor_size_cm"
  | "afp_ng_ml"
  | "alt_u_l"
  | "ast_u_l"
  | "bilirubin_umol_l"
  | "albumin_g_l"
  | "platelet_10e9_l"
  | "portal_vein_invasion"
  | "radiomics_entropy"
  | "radiomics_glcm_contrast";

type FeatureSpec = {
  key: FeatureKey;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
};

const FEATURES: FeatureSpec[] = [
  {
    key: "tumor_size_cm",
    label: "肿瘤直径",
    unit: "cm",
    min: 0.5,
    max: 20,
    step: 0.1,
  },
  {
    key: "afp_ng_ml",
    label: "AFP",
    unit: "ng/mL",
    min: 0.5,
    max: 50000,
    step: 1,
  },
  { key: "alt_u_l", label: "ALT", unit: "U/L", min: 5, max: 500, step: 1 },
  { key: "ast_u_l", label: "AST", unit: "U/L", min: 5, max: 500, step: 1 },
  {
    key: "bilirubin_umol_l",
    label: "总胆红素",
    unit: "umol/L",
    min: 2,
    max: 150,
    step: 0.1,
  },
  {
    key: "albumin_g_l",
    label: "白蛋白",
    unit: "g/L",
    min: 15,
    max: 55,
    step: 0.1,
  },
  {
    key: "platelet_10e9_l",
    label: "血小板",
    unit: "10e9/L",
    min: 20,
    max: 500,
    step: 1,
  },
  {
    key: "portal_vein_invasion",
    label: "门静脉侵犯",
    unit: "0/1",
    min: 0,
    max: 1,
    step: 1,
  },
  {
    key: "radiomics_entropy",
    label: "影像组学熵",
    unit: "score",
    min: 2,
    max: 8,
    step: 0.1,
  },
  {
    key: "radiomics_glcm_contrast",
    label: "GLCM 对比度",
    unit: "score",
    min: 10,
    max: 250,
    step: 0.1,
  },
];

const SAMPLE_HIGH: Record<FeatureKey, number> = {
  tumor_size_cm: 6.2,
  afp_ng_ml: 420,
  alt_u_l: 61,
  ast_u_l: 72,
  bilirubin_umol_l: 24,
  albumin_g_l: 36,
  platelet_10e9_l: 128,
  portal_vein_invasion: 1,
  radiomics_entropy: 5.4,
  radiomics_glcm_contrast: 112,
};

const SAMPLE_BORDERLINE: Record<FeatureKey, number> = {
  tumor_size_cm: 3.4,
  afp_ng_ml: 95,
  alt_u_l: 42,
  ast_u_l: 48,
  bilirubin_umol_l: 18,
  albumin_g_l: 39,
  platelet_10e9_l: 168,
  portal_vein_invasion: 0,
  radiomics_entropy: 4.5,
  radiomics_glcm_contrast: 74,
};

type FeatureValues = Partial<Record<FeatureKey, string>>;

type Completeness = {
  complete: boolean;
  missingFeatures: string[];
  receivedFeatures: string[];
  requiredFeatures: string[];
};

type Prediction = {
  prediction: {
    label: string;
    probability_high_grade: number;
    probability_low_or_intermediate: number;
    uncertain_probability_band: boolean;
  };
  model: {
    type: string;
    cv_auc_mean?: number | null;
    feature_selection_method?: string | null;
  };
  features_used: string[];
};

type ShapFeature = {
  feature: string;
  value: number;
  shap_value: number;
  abs_shap_value: number;
  direction: string;
  trust_level: string;
  consistency_note: string;
};

type Explanation = {
  shap: {
    method: string;
    base_value: number;
    top_features: ShapFeature[];
    high_trust_features: string[];
    statistical_only_features: string[];
    consistency_summary: string;
    caveat: string;
  };
};

type EvidenceHit = {
  id: string;
  title: string;
  paragraph: string;
  source: { label: string; url: string };
  scores: { final: number; rerank: number; bm25: number; embedding: number };
};

type Evidence = {
  confidence: "high" | "medium" | "low";
  evidenceSufficient: boolean;
  retrievalMethod: string;
  results: EvidenceHit[];
};

type MemoryOutput = {
  saved: boolean;
  recordCount: number;
  comparison: {
    hasPrevious: boolean;
    summary: string;
    probabilityDelta?: number;
    labelChanged?: boolean;
    changedFeatures: Array<{
      feature: string;
      previous: number;
      current: number;
      delta: number;
    }>;
  };
};

type DeepReasonTraceItem = {
  nodeId: string;
  agentName: string;
  handlerName: string;
  status: string;
  durationMs: number;
  outputSummary: string;
};

type DeepReasonClaimEvidence = {
  claimId: string;
  supportStatus: "supported" | "partially_supported" | "unsupported";
  evidenceIds: string[];
};

type DeepReasonGateDecision = {
  status: "allow" | "limited" | "interrupt" | "deny";
  permittedClaimIds: string[];
  deniedClaimIds: string[];
  evidenceGaps: string[];
  reasons: string[];
};

type DeepReasonMemoryProposal = {
  proposalId: string;
  status: "pending_approval" | "approved_applied" | "rejected";
  applied: boolean;
};

type DeepReasonOutput = {
  workflowTrace: DeepReasonTraceItem[];
  agentTrace: DeepReasonTraceItem[];
  claimEvidenceMap: DeepReasonClaimEvidence[];
  gateDecision?: DeepReasonGateDecision;
  retryCount: number;
  memoryProposal?: DeepReasonMemoryProposal;
  verificationResult?: {
    passed: boolean;
    status: string;
    failures: string[];
  };
};

type AnalyzeResponse = {
  safetyNotice: string;
  disclaimer: string;
  text?: string;
  finishReason?: string;
  steps?: number;
  error?: string;
  hint?: string;
  toolCalls?: Array<{ toolName: string; toolCallId: string }>;
  trace?: Array<Record<string, unknown>>;
  analysis?: {
    completeness?: Completeness;
    prediction?: Prediction;
    explanation?: Explanation;
    evidence?: Evidence;
    memory?: MemoryOutput;
  };
  deepreason?: DeepReasonOutput;
};

function initialValues(): FeatureValues {
  return Object.fromEntries(
    Object.entries(SAMPLE_HIGH).map(([key, value]) => [key, String(value)]),
  ) as FeatureValues;
}

function valuesFromSample(sample: Partial<Record<FeatureKey, number>>) {
  return Object.fromEntries(
    Object.entries(sample).map(([key, value]) => [key, String(value)]),
  ) as FeatureValues;
}

function compactFeatures(values: FeatureValues) {
  const features: Record<string, number> = {};
  for (const feature of FEATURES) {
    const raw = values[feature.key];
    if (raw === undefined || raw.trim() === "") {
      continue;
    }
    features[feature.key] =
      feature.key === "portal_vein_invasion"
        ? Number.parseInt(raw, 10)
        : Number.parseFloat(raw);
  }
  return features;
}

function percent(value?: number) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "NA";
}

function signedPercentPoint(value?: number) {
  if (typeof value !== "number") {
    return "NA";
  }
  const sign = value >= 0 ? "+" : "-";
  return `${sign}${Math.abs(value * 100).toFixed(1)} 个百分点`;
}

function trustLabel(value: string) {
  return value === "high_trust_causal_candidate"
    ? "高可信解释"
    : "仅统计相关";
}

function directionLabel(value: string) {
  return value === "pushes_toward_high_grade" ? "推向高分级" : "推向低/中分级";
}

function samplePartial() {
  return valuesFromSample({
    tumor_size_cm: SAMPLE_HIGH.tumor_size_cm,
    afp_ng_ml: SAMPLE_HIGH.afp_ng_ml,
  });
}

export default function Page() {
  const [sessionId, setSessionId] = useState(
    () => `web-session-${Math.random().toString(16).slice(2, 10)}`,
  );
  const [patientId, setPatientId] = useState("synthetic-demo-001");
  const [values, setValues] = useState<FeatureValues>(() => initialValues());
  const [userInstruction, setUserInstruction] = useState(
    "请生成结构化报告，并保留引用、免责声明和复诊对比。",
  );
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const completedCount = useMemo(
    () => Object.keys(compactFeatures(values)).length,
    [values],
  );

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    setResult(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          sessionId,
          patientId: patientId.trim() || undefined,
          features: compactFeatures(values),
          userInstruction: userInstruction.trim() || undefined,
        }),
      });
      const payload = (await response.json()) as AnalyzeResponse;
      setResult(payload);
    } catch (error) {
      setResult({
        safetyNotice: SAFETY_NOTICE,
        disclaimer:
          "免责声明：本报告仅用于科研学习与工程演示，不作为任何临床诊断、治疗或病理分级依据。",
        error: error instanceof Error ? error.message : "请求失败",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  function updateValue(key: FeatureKey, nextValue: string) {
    setValues((current) => ({ ...current, [key]: nextValue }));
  }

  function resetSession() {
    setSessionId(`web-session-${Math.random().toString(16).slice(2, 10)}`);
    setResult(null);
  }

  const prediction = result?.analysis?.prediction;
  const explanation = result?.analysis?.explanation;
  const evidence = result?.analysis?.evidence;
  const memory = result?.analysis?.memory;
  const completeness = result?.analysis?.completeness;
  const deepreason = result?.deepreason;
  const gateDecision = deepreason?.gateDecision;
  const missingFeatures = completeness?.missingFeatures ?? [];

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-teal-700">
              <ShieldCheck className="h-4 w-4" aria-hidden />
              <span>{SAFETY_NOTICE}</span>
            </div>
            <h1 className="mt-2 text-2xl font-semibold text-slate-950">
              HCC Causal Agent Prototype
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="rounded-md border border-amber-300 bg-amber-50 px-3 py-1.5 text-amber-900">
              不作为临床诊断依据
            </span>
            <span className="rounded-md border border-teal-300 bg-teal-50 px-3 py-1.5 text-teal-900">
              Tool-bound prediction
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[440px_minmax(0,1fr)]">
        <form
          onSubmit={submit}
          className="rounded-md border border-slate-200 bg-white p-4 shadow-panel"
        >
          <div className="flex items-center justify-between gap-3 border-b border-slate-200 pb-3">
            <div className="flex items-center gap-2">
              <ClipboardList className="h-5 w-5 text-teal-700" aria-hidden />
              <h2 className="text-base font-semibold">合成特征输入</h2>
            </div>
            <span className="text-sm text-slate-600">{completedCount}/10</span>
          </div>

          <div className="mt-4 grid gap-3">
            <label className="grid gap-1 text-sm">
              <span className="font-medium text-slate-700">session_id</span>
              <div className="flex gap-2">
                <input
                  className="min-w-0 flex-1 rounded-md border border-slate-300 px-3 py-2 outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
                  value={sessionId}
                  onChange={(event) => setSessionId(event.target.value)}
                />
                <button
                  type="button"
                  title="新建会话"
                  onClick={resetSession}
                  className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-slate-300 text-slate-700 hover:bg-slate-100"
                >
                  <RefreshCw className="h-4 w-4" aria-hidden />
                </button>
              </div>
            </label>

            <label className="grid gap-1 text-sm">
              <span className="font-medium text-slate-700">patient_id</span>
              <input
                className="rounded-md border border-slate-300 px-3 py-2 outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
                value={patientId}
                onChange={(event) => setPatientId(event.target.value)}
              />
            </label>

            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => setValues(valuesFromSample(SAMPLE_HIGH))}
                className="inline-flex items-center justify-center gap-1 rounded-md border border-teal-300 bg-teal-50 px-2 py-2 text-sm font-medium text-teal-900 hover:bg-teal-100"
              >
                <Sparkles className="h-4 w-4" aria-hidden />
                高风险样例
              </button>
              <button
                type="button"
                onClick={() => setValues(valuesFromSample(SAMPLE_BORDERLINE))}
                className="inline-flex items-center justify-center gap-1 rounded-md border border-amber-300 bg-amber-50 px-2 py-2 text-sm font-medium text-amber-900 hover:bg-amber-100"
              >
                <Activity className="h-4 w-4" aria-hidden />
                临界样例
              </button>
              <button
                type="button"
                onClick={() => setValues(samplePartial())}
                className="inline-flex items-center justify-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-2 text-sm font-medium text-slate-800 hover:bg-slate-100"
              >
                <AlertTriangle className="h-4 w-4" aria-hidden />
                缺失样例
              </button>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {FEATURES.map((feature) => (
                <label key={feature.key} className="grid gap-1 text-sm">
                  <span className="flex items-center justify-between gap-2">
                    <span className="font-medium text-slate-700">
                      {feature.label}
                    </span>
                    <span className="text-xs text-slate-500">
                      {feature.unit}
                    </span>
                  </span>
                  {feature.key === "portal_vein_invasion" ? (
                    <select
                      value={values[feature.key] ?? ""}
                      onChange={(event) =>
                        updateValue(feature.key, event.target.value)
                      }
                      className="h-10 rounded-md border border-slate-300 px-3 outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
                    >
                      <option value="">未填</option>
                      <option value="0">0</option>
                      <option value="1">1</option>
                    </select>
                  ) : (
                    <input
                      type="number"
                      min={feature.min}
                      max={feature.max}
                      step={feature.step}
                      value={values[feature.key] ?? ""}
                      onChange={(event) =>
                        updateValue(feature.key, event.target.value)
                      }
                      className="h-10 rounded-md border border-slate-300 px-3 outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
                    />
                  )}
                </label>
              ))}
            </div>

            <label className="grid gap-1 text-sm">
              <span className="font-medium text-slate-700">自然语言补充</span>
              <textarea
                rows={3}
                value={userInstruction}
                onChange={(event) => setUserInstruction(event.target.value)}
                className="resize-none rounded-md border border-slate-300 px-3 py-2 outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
              />
            </label>

            <button
              type="submit"
              disabled={isSubmitting}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {isSubmitting ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              ) : (
                <Play className="h-4 w-4" aria-hidden />
              )}
              运行 Agent
            </button>
          </div>
        </form>

        <section className="grid gap-5">
          {result?.error ? (
            <div className="rounded-md border border-rose-300 bg-rose-50 p-4 text-rose-950">
              <div className="flex items-center gap-2 font-semibold">
                <AlertTriangle className="h-5 w-5" aria-hidden />
                Agent 执行失败
              </div>
              <p className="mt-2 text-sm">{result.error}</p>
              {result.hint ? <p className="mt-1 text-sm">{result.hint}</p> : null}
            </div>
          ) : null}

          <div className="grid gap-3 md:grid-cols-3">
            <MetricPanel
              icon={<BarChart3 className="h-5 w-5" aria-hidden />}
              label="高分级概率"
              value={percent(prediction?.prediction.probability_high_grade)}
              tone={
                prediction?.prediction.uncertain_probability_band
                  ? "amber"
                  : "teal"
              }
            />
            <MetricPanel
              icon={<Activity className="h-5 w-5" aria-hidden />}
              label="预测标签"
              value={prediction?.prediction.label ?? "等待 Tool 输出"}
              tone="slate"
            />
            <MetricPanel
              icon={<Database className="h-5 w-5" aria-hidden />}
              label="病例记忆"
              value={
                memory
                  ? `${memory.saved ? "已保存" : "未保存"} / ${memory.recordCount}`
                  : "等待 Tool 输出"
              }
              tone="teal"
            />
          </div>

          <div className="grid gap-3 md:grid-cols-4">
            <MetricPanel
              icon={<ShieldCheck className="h-5 w-5" aria-hidden />}
              label="DeepReason Gate"
              value={gateDecision?.status ?? "等待 Gate"}
              tone={
                gateDecision?.status === "allow"
                  ? "teal"
                  : gateDecision?.status === "limited"
                    ? "amber"
                    : "slate"
              }
            />
            <MetricPanel
              icon={<ClipboardList className="h-5 w-5" aria-hidden />}
              label="Claim-Evidence"
              value={
                deepreason
                  ? `${deepreason.claimEvidenceMap.length} 条 Claim`
                  : "等待核验"
              }
              tone="slate"
            />
            <MetricPanel
              icon={<RefreshCw className="h-5 w-5" aria-hidden />}
              label="证据重试"
              value={
                typeof deepreason?.retryCount === "number"
                  ? String(deepreason.retryCount)
                  : "等待"
              }
              tone={deepreason?.retryCount ? "amber" : "teal"}
            />
            <MetricPanel
              icon={<Database className="h-5 w-5" aria-hidden />}
              label="Memory Proposal"
              value={deepreason?.memoryProposal?.status ?? "等待提案"}
              tone={
                deepreason?.memoryProposal?.status === "rejected"
                  ? "amber"
                  : "teal"
              }
            />
          </div>

          {missingFeatures.length > 0 ? (
            <div className="rounded-md border border-amber-300 bg-amber-50 p-4">
              <div className="flex items-center gap-2 font-semibold text-amber-950">
                <AlertTriangle className="h-5 w-5" aria-hidden />
                需要补充的合成特征
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {missingFeatures.map((feature) => (
                  <span
                    key={feature}
                    className="rounded-md border border-amber-300 bg-white px-2.5 py-1 text-sm text-amber-950"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          <div className="grid gap-5 xl:grid-cols-[minmax(0,1.1fr)_minmax(340px,0.9fr)]">
            <div className="rounded-md border border-slate-200 bg-white p-4 shadow-panel">
              <div className="flex items-center justify-between gap-3 border-b border-slate-200 pb-3">
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-teal-700" aria-hidden />
                  <h2 className="text-base font-semibold">结构化报告</h2>
                </div>
                {result?.finishReason ? (
                  <span className="text-sm text-slate-600">
                    steps: {result.steps}
                  </span>
                ) : null}
              </div>
              <pre className="mt-4 max-h-[720px] overflow-auto rounded-md bg-slate-950 p-4 text-sm leading-6 text-slate-50">
                {result?.text ??
                  "运行后显示 Agent 生成的结构化报告。报告末尾保留免责声明。"}
              </pre>
            </div>

            <div className="grid gap-5">
              <PanelTitle icon={<BarChart3 className="h-5 w-5" aria-hidden />}>
                SHAP Top 特征
              </PanelTitle>
              <div className="grid gap-3">
                {explanation?.shap.top_features.length ? (
                  explanation.shap.top_features.map((feature) => (
                    <div
                      key={feature.feature}
                      className="rounded-md border border-slate-200 bg-white p-3 shadow-panel"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="font-semibold text-slate-950">
                            {feature.feature}
                          </div>
                          <div className="mt-1 text-sm text-slate-600">
                            {directionLabel(feature.direction)}
                          </div>
                        </div>
                        <span
                          className={`rounded-md border px-2 py-1 text-xs font-medium ${
                            feature.trust_level ===
                            "high_trust_causal_candidate"
                              ? "border-teal-300 bg-teal-50 text-teal-900"
                              : "border-amber-300 bg-amber-50 text-amber-900"
                          }`}
                        >
                          {trustLabel(feature.trust_level)}
                        </span>
                      </div>
                      <div className="mt-3 h-2 overflow-hidden rounded-md bg-slate-100">
                        <div
                          className={`h-full ${
                            feature.shap_value >= 0
                              ? "bg-teal-600"
                              : "bg-rose-500"
                          }`}
                          style={{
                            width: `${Math.min(
                              100,
                              Math.max(8, feature.abs_shap_value * 420),
                            )}%`,
                          }}
                        />
                      </div>
                      <div className="mt-2 flex items-center justify-between text-sm">
                        <span>value: {feature.value}</span>
                        <span>SHAP: {feature.shap_value.toFixed(4)}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <EmptyState text="等待 SHAP Tool 输出。" />
                )}
              </div>

              <PanelTitle icon={<Search className="h-5 w-5" aria-hidden />}>
                检索引用
              </PanelTitle>
              <div className="grid gap-3">
                {evidence?.results.length ? (
                  evidence.results.map((hit) => (
                    <article
                      key={hit.id}
                      className="rounded-md border border-slate-200 bg-white p-3 shadow-panel"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <h3 className="font-semibold text-slate-950">
                          [{hit.id}] {hit.title}
                        </h3>
                        <span className="text-xs text-slate-500">
                          {hit.scores.final.toFixed(3)}
                        </span>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-slate-700">
                        {hit.paragraph}
                      </p>
                      <a
                        href={hit.source.url}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-2 inline-flex text-sm font-medium text-teal-700 hover:text-teal-900"
                      >
                        {hit.source.label}
                      </a>
                    </article>
                  ))
                ) : (
                  <EmptyState
                    text={
                      evidence && !evidence.evidenceSufficient
                        ? "现有资料不足，无法给出解释。"
                        : "等待检索 Tool 输出。"
                    }
                  />
                )}
              </div>

              <div className="rounded-md border border-slate-200 bg-white p-4 shadow-panel">
                <div className="flex items-center gap-2 font-semibold">
                  <History className="h-5 w-5 text-teal-700" aria-hidden />
                  复诊趋势
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-700">
                  {memory?.comparison.summary ?? "等待病例记忆 Tool 输出。"}
                </p>
                {memory?.comparison.hasPrevious ? (
                  <div className="mt-3 grid gap-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-600">概率变化</span>
                      <span className="font-medium">
                        {signedPercentPoint(memory.comparison.probabilityDelta)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-600">标签变化</span>
                      <span className="font-medium">
                        {memory.comparison.labelChanged ? "是" : "否"}
                      </span>
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="rounded-md border border-slate-200 bg-white p-4 shadow-panel">
                <div className="flex items-center gap-2 font-semibold">
                  <CheckCircle2 className="h-5 w-5 text-teal-700" aria-hidden />
                  Tool Trace
                </div>
                <div className="mt-3 grid gap-2">
                  {result?.toolCalls?.length ? (
                    result.toolCalls.map((call, index) => (
                      <div
                        key={`${call.toolCallId}-${index}`}
                        className="flex items-center justify-between rounded-md border border-slate-200 px-3 py-2 text-sm"
                      >
                        <span>{call.toolName}</span>
                        <span className="text-slate-500">#{index + 1}</span>
                      </div>
                    ))
                  ) : (
                    <EmptyState text="等待 Agent 调用 Tool。" />
                  )}
                </div>
              </div>

              <div className="rounded-md border border-slate-200 bg-white p-4 shadow-panel">
                <div className="flex items-center gap-2 font-semibold">
                  <ShieldCheck className="h-5 w-5 text-teal-700" aria-hidden />
                  Gate 决策
                </div>
                {gateDecision ? (
                  <div className="mt-3 grid gap-2 text-sm">
                    <div className="flex items-center justify-between rounded-md border border-slate-200 px-3 py-2">
                      <span className="text-slate-600">status</span>
                      <span className="font-semibold">{gateDecision.status}</span>
                    </div>
                    <div className="flex items-center justify-between rounded-md border border-slate-200 px-3 py-2">
                      <span className="text-slate-600">permitted</span>
                      <span className="font-semibold">
                        {gateDecision.permittedClaimIds.length}
                      </span>
                    </div>
                    <div className="flex items-center justify-between rounded-md border border-slate-200 px-3 py-2">
                      <span className="text-slate-600">denied</span>
                      <span className="font-semibold">
                        {gateDecision.deniedClaimIds.length}
                      </span>
                    </div>
                    <p className="rounded-md bg-slate-50 p-3 leading-6 text-slate-700">
                      {gateDecision.reasons[0] ?? "等待 Gate 输出。"}
                    </p>
                  </div>
                ) : (
                  <EmptyState text="等待 DeepReason Gate 输出。" />
                )}
              </div>

              <div className="rounded-md border border-slate-200 bg-white p-4 shadow-panel">
                <div className="flex items-center gap-2 font-semibold">
                  <ClipboardList className="h-5 w-5 text-teal-700" aria-hidden />
                  Workflow Trace
                </div>
                <div className="mt-3 grid gap-2">
                  {deepreason?.workflowTrace.length ? (
                    deepreason.workflowTrace.map((item, index) => (
                      <div
                        key={`${item.nodeId}-${index}`}
                        className="rounded-md border border-slate-200 px-3 py-2 text-sm"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="font-medium">{item.nodeId}</span>
                          <span className="text-slate-500">
                            {item.durationMs.toFixed(1)}ms
                          </span>
                        </div>
                        <div className="mt-1 text-xs text-slate-500">
                          {item.agentName} · {item.status}
                        </div>
                      </div>
                    ))
                  ) : (
                    <EmptyState text="等待 DeepReason 工作流轨迹。" />
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

function MetricPanel({
  icon,
  label,
  value,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  tone: "teal" | "amber" | "slate";
}) {
  const toneClass =
    tone === "teal"
      ? "border-teal-200 bg-teal-50 text-teal-950"
      : tone === "amber"
        ? "border-amber-200 bg-amber-50 text-amber-950"
        : "border-slate-200 bg-white text-slate-950";

  return (
    <div className={`rounded-md border p-4 shadow-panel ${toneClass}`}>
      <div className="flex items-center gap-2 text-sm font-medium opacity-85">
        {icon}
        {label}
      </div>
      <div className="mt-2 break-words text-xl font-semibold">{value}</div>
    </div>
  );
}

function PanelTitle({
  icon,
  children,
}: {
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 rounded-md border border-slate-200 bg-white px-4 py-3 font-semibold shadow-panel">
      <span className="text-teal-700">{icon}</span>
      {children}
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-md border border-dashed border-slate-300 bg-white p-4 text-sm text-slate-600">
      {text}
    </div>
  );
}
