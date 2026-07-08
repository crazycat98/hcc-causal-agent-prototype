import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  knowledgeEntrySchema,
  retrievalResultSchema,
  type KnowledgeEntry,
  type RetrievalResult,
} from "./schema.js";

const SYNTHETIC_DATA_NOTICE =
  "\u6f14\u793a\u7528\u5408\u6210\u6570\u636e\uff0c\u975e\u771f\u5b9e\u60a3\u8005\u6570\u636e\uff1b\u975e\u4e34\u5e8a\u8bca\u65ad\u4f9d\u636e\u3002";

const CITATION_INSTRUCTION =
  "\u62a5\u544a\u53ea\u80fd\u57fa\u4e8e results \u4e2d\u7684 paragraph \u4f5c\u7b54\uff0c\u5e76\u4f7f\u7528 [KB-ID] \u6807\u6ce8\u5f15\u7528\uff1b\u8bc1\u636e\u4e0d\u8db3\u65f6\u5fc5\u987b\u8bf4\u660e\u73b0\u6709\u8d44\u6599\u4e0d\u8db3\u3002";

const FEATURE_SYNONYMS: Record<string, string[]> = {
  tumor_size_cm: ["tumor size", "lesion size", "diameter", "mass size"],
  afp_ng_ml: ["AFP", "alpha-fetoprotein", "tumor marker"],
  alt_u_l: ["ALT", "alanine transaminase", "liver function"],
  ast_u_l: ["AST", "aspartate aminotransferase", "liver function"],
  bilirubin_umol_l: ["bilirubin", "jaundice", "liver function"],
  albumin_g_l: ["albumin", "liver synthesis", "liver function"],
  platelet_10e9_l: ["platelet", "platelet count", "thrombocyte"],
  portal_vein_invasion: [
    "portal vein invasion",
    "portal vein",
    "vascular invasion",
  ],
  radiomics_entropy: ["radiomics entropy", "entropy", "texture"],
  radiomics_glcm_contrast: [
    "GLCM contrast",
    "grey level co-occurrence",
    "radiomics texture",
  ],
};

let cachedKnowledgeBase: KnowledgeEntry[] | undefined;

function loadKnowledgeBase(): KnowledgeEntry[] {
  if (cachedKnowledgeBase) {
    return cachedKnowledgeBase;
  }

  const currentDir = dirname(fileURLToPath(import.meta.url));
  const filePath = resolve(currentDir, "..", "knowledge-base.json");
  const raw = JSON.parse(readFileSync(filePath, "utf-8")) as unknown;
  cachedKnowledgeBase = knowledgeEntrySchema.array().parse(raw);
  return cachedKnowledgeBase;
}

function tokenize(text: string): string[] {
  const normalized = text
    .toLowerCase()
    .replaceAll("_", " ")
    .replaceAll("-", " ");
  return normalized.match(/[a-z0-9]+|[\u4e00-\u9fff]/g) ?? [];
}

function expandQuery(query: string, featureNames: string[]): string {
  const expansions = featureNames.flatMap((name) => [
    name,
    ...(FEATURE_SYNONYMS[name] ?? []),
  ]);
  return [query, ...expansions].join(" ");
}

function termFrequency(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  return counts;
}

function bm25Scores(queryTokens: string[], documents: string[][]): number[] {
  const k1 = 1.4;
  const b = 0.75;
  const avgDocLength =
    documents.reduce((sum, doc) => sum + doc.length, 0) / documents.length;
  const docFreq = new Map<string, number>();

  for (const doc of documents) {
    for (const token of new Set(doc)) {
      docFreq.set(token, (docFreq.get(token) ?? 0) + 1);
    }
  }

  return documents.map((doc) => {
    const frequencies = termFrequency(doc);
    let score = 0;
    for (const token of queryTokens) {
      const tf = frequencies.get(token) ?? 0;
      if (tf === 0) {
        continue;
      }
      const df = docFreq.get(token) ?? 0;
      const idf = Math.log(1 + (documents.length - df + 0.5) / (df + 0.5));
      const denominator = tf + k1 * (1 - b + b * (doc.length / avgDocLength));
      score += idf * ((tf * (k1 + 1)) / denominator);
    }
    return score;
  });
}

function hashToken(token: string, dimensions: number): number {
  let hash = 2166136261;
  for (let i = 0; i < token.length; i += 1) {
    hash ^= token.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash) % dimensions;
}

function hashedVector(tokens: string[], dimensions = 128): number[] {
  const vector = Array.from({ length: dimensions }, () => 0);
  for (const token of tokens) {
    const index = hashToken(token, dimensions);
    vector[index] = (vector[index] ?? 0) + 1;
  }
  const norm =
    Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => value / norm);
}

function cosine(left: number[], right: number[]): number {
  return left.reduce(
    (sum, value, index) => sum + value * (right[index] ?? 0),
    0,
  );
}

function normalize(scores: number[]): number[] {
  const max = Math.max(...scores);
  if (max <= 0) {
    return scores.map(() => 0);
  }
  return scores.map((score) => score / max);
}

function matchedTerms(entry: KnowledgeEntry, queryTokens: Set<string>): string[] {
  return entry.terms
    .filter((term) => tokenize(term).some((token) => queryTokens.has(token)))
    .slice(0, 8);
}

function coveredFeatures(entry: KnowledgeEntry, featureNames: string[]): string[] {
  const termTokenSets = entry.terms.map((term) => new Set(tokenize(term)));
  return featureNames.filter((name) => {
    const phrases = [name, ...(FEATURE_SYNONYMS[name] ?? [])];
    return phrases.some((phrase) => {
      const phraseTokens = tokenize(phrase);
      return termTokenSets.some((termTokens) =>
        phraseTokens.every((token) => termTokens.has(token)),
      );
    });
  });
}

type ScoredEntry = {
  entry: KnowledgeEntry;
  matchedTerms: string[];
  coveredFeatures: string[];
  scores: {
    bm25: number;
    embedding: number;
    rerank: number;
    final: number;
  };
};

export type RetrievalAblationMode =
  | "full"
  | "no_bm25"
  | "no_embedding"
  | "no_rerank"
  | "no_query_expansion"
  | "no_diversity"
  | "bm25_only"
  | "embedding_only";

const RETRIEVAL_METHOD_BY_MODE: Record<RetrievalAblationMode, string> = {
  full: "bm25+local_hash_embedding+heuristic_rerank",
  no_bm25: "ablation:no_bm25+local_hash_embedding+heuristic_rerank",
  no_embedding: "ablation:bm25+no_embedding+heuristic_rerank",
  no_rerank: "ablation:bm25+local_hash_embedding+no_rerank",
  no_query_expansion: "ablation:no_query_expansion",
  no_diversity: "ablation:no_diverse_topk",
  bm25_only: "ablation:bm25_only",
  embedding_only: "ablation:local_hash_embedding_only",
};

function componentWeights(mode: RetrievalAblationMode) {
  const base = {
    bm25: 0.45,
    embedding: 0.35,
    rerank: 0.2,
  };
  const weights = { ...base };

  if (mode === "no_bm25" || mode === "embedding_only") {
    weights.bm25 = 0;
  }
  if (mode === "no_embedding" || mode === "bm25_only") {
    weights.embedding = 0;
  }
  if (
    mode === "no_rerank" ||
    mode === "bm25_only" ||
    mode === "embedding_only"
  ) {
    weights.rerank = 0;
  }

  const total = weights.bm25 + weights.embedding + weights.rerank || 1;
  return {
    bm25: weights.bm25 / total,
    embedding: weights.embedding / total,
    rerank: weights.rerank / total,
  };
}

function selectDiverseTopK(scored: ScoredEntry[], topK: number): ScoredEntry[] {
  const selected: ScoredEntry[] = [];
  const used = new Set<ScoredEntry>();
  const covered = new Set<string>();

  while (selected.length < topK && selected.length < scored.length) {
    const diverseCandidate = scored.find(
      (item) =>
        !used.has(item) &&
        item.coveredFeatures.some((feature) => !covered.has(feature)),
    );
    const candidate = diverseCandidate ?? scored.find((item) => !used.has(item));
    if (!candidate) {
      break;
    }

    selected.push(candidate);
    used.add(candidate);
    for (const feature of candidate.coveredFeatures) {
      covered.add(feature);
    }
  }

  return selected;
}

export function retrieveMedicalEvidence(options: {
  query: string;
  featureNames?: string[];
  topK?: number;
  ablationMode?: RetrievalAblationMode;
}): RetrievalResult {
  const featureNames = options.featureNames ?? [];
  const topK = options.topK ?? 5;
  const ablationMode = options.ablationMode ?? "full";
  const knowledgeBase = loadKnowledgeBase();
  const expandedQuery =
    ablationMode === "no_query_expansion"
      ? options.query
      : expandQuery(options.query, featureNames);
  const queryTokens = tokenize(expandedQuery);
  const queryTokenSet = new Set(queryTokens);
  const documentTokens = knowledgeBase.map((entry) =>
    tokenize([entry.title, entry.terms.join(" "), entry.text].join(" ")),
  );

  const bm25Raw = bm25Scores(queryTokens, documentTokens);
  const bm25 = normalize(bm25Raw);
  const queryVector = hashedVector(queryTokens);
  const embedding = documentTokens.map((tokens) =>
    cosine(queryVector, hashedVector(tokens)),
  );
  const weights = componentWeights(ablationMode);

  const scored = knowledgeBase
    .map((entry, index): ScoredEntry => {
      const matches = matchedTerms(entry, queryTokenSet);
      const features = coveredFeatures(entry, featureNames);
      const featureCoverage = features.length > 0 ? 1 : 0;
      const termCoverage = Math.min(matches.length / 4, 1);
      const traceableSource = entry.source.url.startsWith("https://") ? 1 : 0;
      const rerank =
        0.5 * termCoverage + 0.35 * featureCoverage + 0.15 * traceableSource;
      const bm25Score = bm25[index] ?? 0;
      const embeddingScore = embedding[index] ?? 0;
      const final =
        weights.bm25 * bm25Score +
        weights.embedding * embeddingScore +
        weights.rerank * rerank;

      return {
        entry,
        matchedTerms: matches,
        coveredFeatures: features,
        scores: {
          bm25: Number(bm25Score.toFixed(4)),
          embedding: Number(embeddingScore.toFixed(4)),
          rerank: Number(rerank.toFixed(4)),
          final: Number(final.toFixed(4)),
        },
      };
    })
    .sort((left, right) => right.scores.final - left.scores.final);

  const selected =
    ablationMode === "no_diversity"
      ? scored.slice(0, topK)
      : selectDiverseTopK(scored, topK);
  const results = selected.map(
    ({ entry, matchedTerms: matches, scores }) => ({
      id: entry.id,
      title: entry.title,
      paragraph: entry.text,
      source: entry.source,
      matchedTerms: matches,
      scores,
    }),
  );

  const bestScore = results[0]?.scores.final ?? 0;
  const hasAnyMatchedTerm = results.some((item) => item.matchedTerms.length > 0);
  const confidence =
    !hasAnyMatchedTerm
      ? "low"
      : bestScore >= 0.45 && results.length >= Math.min(3, topK)
      ? "high"
      : bestScore >= 0.25
        ? "medium"
        : "low";
  const evidenceSufficient = confidence !== "low";

  return retrievalResultSchema.parse({
    query: options.query,
    topK,
    evidenceSufficient,
    confidence,
    retrievalMethod: RETRIEVAL_METHOD_BY_MODE[ablationMode],
    results,
    safetyNotice: SYNTHETIC_DATA_NOTICE,
    citationInstruction: CITATION_INSTRUCTION,
  });
}
