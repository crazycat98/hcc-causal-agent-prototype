import { z } from "zod";

export const knowledgeSourceSchema = z
  .object({
    label: z.string(),
    url: z.string().url(),
    type: z.string(),
  })
  .strict();

export const knowledgeEntrySchema = z
  .object({
    id: z.string(),
    title: z.string(),
    terms: z.array(z.string()),
    text: z.string(),
    source: knowledgeSourceSchema,
  })
  .strict();

export const retrievalHitSchema = z
  .object({
    id: z.string(),
    title: z.string(),
    paragraph: z.string(),
    source: knowledgeSourceSchema,
    matchedTerms: z.array(z.string()),
    scores: z
      .object({
        bm25: z.number(),
        embedding: z.number(),
        rerank: z.number(),
        final: z.number(),
      })
      .strict(),
  })
  .strict();

export const retrievalResultSchema = z
  .object({
    query: z.string(),
    topK: z.number().int(),
    evidenceSufficient: z.boolean(),
    confidence: z.enum(["high", "medium", "low"]),
    retrievalMethod: z.literal("bm25+local_hash_embedding+heuristic_rerank"),
    results: z.array(retrievalHitSchema),
    safetyNotice: z.string(),
    citationInstruction: z.string(),
  })
  .strict();

export type KnowledgeEntry = z.infer<typeof knowledgeEntrySchema>;
export type RetrievalHit = z.infer<typeof retrievalHitSchema>;
export type RetrievalResult = z.infer<typeof retrievalResultSchema>;

