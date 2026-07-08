# Retrieval Module

M4 implements a local hybrid retrieval prototype for public, traceable medical background snippets.

Safety notice: 演示用合成数据，非真实患者数据；非临床诊断依据。

## Scope

- Knowledge base entries are short, self-written Chinese summaries.
- Sources are public pages such as NCI Dictionary, MedlinePlus, IBSI, and PyRadiomics documentation.
- The module does not store or retrieve real patient records.
- It does not use copyrighted full-text medical literature.

## Local Hybrid Retrieval

The M4 prototype combines:

- BM25 keyword scoring.
- Local hashed token-vector cosine similarity as a lightweight embedding substitute.
- Heuristic rerank based on feature/term coverage and source traceability.

This is intentionally replaceable. Later versions can swap the embedding stage for a local sentence-transformer or API embedding model, and swap heuristic rerank for a cross-encoder or rerank API.

