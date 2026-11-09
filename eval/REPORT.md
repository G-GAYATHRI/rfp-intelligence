# Retrieval Evaluation Report

## Dataset construction

- Source: the actual Bid 1 and Bid 2 PDFs and downloaded BidNet HTML pages supplied with the assignment.
- Judged pairs: **27** (Bid 1: 12; Bid 2: 15).
- Parsed source pages: **76**.
- Offline evaluation chunks: **253**, spanning 75 pages with non-empty chunks.
- Every evaluation row contains a natural-language query, bid ID, expected source file, expected page, exact target phrase, and expected chunk ID. `tests/test_eval_dataset.py` checks that every label points to a real corpus chunk containing the judged passage.

## Offline results

The environment used to prepare this deliverable did not have ChromaDB, rank-bm25, SentenceTransformers, or LangGraph installed, and package installation was blocked by unavailable network/DNS. Therefore, the measured table below is an **offline diagnostic benchmark**, not a measurement of the production embedding/reranker stack.

| Configuration | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|
| TF-IDF cosine vector-style baseline | 0.8148 | 0.6023 | 0.6721 |
| TF-IDF + BM25 with RRF | 0.7778 | 0.6299 | 0.6779 |
| Hybrid + deterministic lexical reranking | 0.7778 | 0.7050 | 0.7326 |

### Interpretation

- The hybrid improved MRR and nDCG slightly over the TF-IDF baseline, but lowered Recall@5 in this run.
- The deterministic reranker improved MRR from 0.6299 to 0.7050 and nDCG@5 from 0.6779 to 0.7326, without increasing Recall@5.
- This indicates that candidate recall and ranking quality should be tuned separately. In particular, increase/adjust the hybrid candidate pool and evaluate RRF settings before concluding that reranking improves end-to-end recall.

### Missed judged passages at rank 5

- Vector-style TF-IDF: `B1-01`, `B1-02`, `B1-03`, `B2-12`, `B2-13`.
- Hybrid: `B1-01`, `B1-02`, `B1-03`, `B2-08`, `B2-12`, `B2-13`.
- Hybrid + lexical reranker: same misses as hybrid.

## Run the production benchmark

After installing the full `requirements.txt` and indexing both bid folders into the configured Chroma collection:

```bash
python -m src.eval.run_eval
```

This runs the actual configured `vector`, `hybrid`, and `hybrid_rerank` modes using SentenceTransformer embeddings, BM25, RRF, and the configured CrossEncoder. It writes `eval/production_results.json`. Replace the offline table with those production measurements before claiming performance for the production stack.
