# RFP Intelligence Platform — RAG Search Engine & Multi-Agent System

A Python project for ingesting bid/RFP folders (PDF and HTML), indexing their passages, retrieving cited evidence, extracting structured bid fields, reconciling addenda, validating citations, and answering questions grounded in source documents.

## 1. What is included

- **Ingestion:** PDF/HTML parsing, page-aware text and table extraction, repeated-line cleanup, document classification, addendum-number detection, and source metadata.
- **Chunking:** heading-aware, page-preserving chunks (default 1,000 characters with 150-character overlap); tables are kept as separate chunks and can grow to 1,500 characters.
- **Retrieval:** Chroma cosine vector search, BM25 keyword search, Reciprocal Rank Fusion (RRF), optional query expansion, metadata filters, and CrossEncoder reranking.
- **Agents:** retrieval, field extraction, addendum reconciliation, validation/critic, bounded retry, Q&A, and LangGraph orchestration.
- **Citations:** extracted values are expected to cite source chunk IDs, file names, and page numbers.
- **Evaluation:** 27 manually judged query-to-passage pairs derived from the supplied Bid 1 and Bid 2 documents, plus a dependency-light offline benchmark and a production-stack evaluation runner.

## 2. Setup

Use Python 3.10+ (Python 3.11 is a good default). From the project root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` file by copying `.env.example`. Select one provider and configure its key and model:

```dotenv
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=your_key_here
```

For an OpenAI-compatible endpoint, set `LLM_PROVIDER=openai`, `LLM_MODEL`, `OPENAI_API_KEY`, and (if needed) `LLM_BASE_URL`. Do not commit `.env` or API keys.

The LLM is used for extraction, reconciliation, and generated answers. Retrieval indexing/evaluation does not require an LLM key.

## 3. Run the project

The two supplied bid folders are kept under `eval/corpus/` as reproducible test data. Index each bid independently:

```bash
python main.py parse eval/corpus/Bid1
python main.py index eval/corpus/Bid1
python main.py index eval/corpus/Bid2
python main.py search "What is the final submission deadline?" --bid Bid1 --top_k 5 --mode hybrid_rerank
```

The CLI currently exposes `parse`, `index`, and `search`. The FastAPI application is in `src/api/app.py`:

```bash
uvicorn src.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API interface.

## 4. Architecture

```text
PDF / HTML bid folder
        |
        v
Ingestion + parsing + metadata
        |
        v
Page/heading/table-aware chunking
        |
        +------------------------------+
        |                              |
        v                              v
SentenceTransformer embeddings       BM25 keyword index
        |                              |
        v                              |
     Chroma                            |
        |                              |
        +---------- RRF fusion <-------+
                       |
                       v
             CrossEncoder reranker
                       |
                       v
              Cited retrieval results
                       |
          +------------+-------------+
          |            |             |
       Extract      Reconcile      Q&A
          |          addenda        |
          +------------+-------------+
                       |
                 Validator/Critic
                       |
              Retry rejected fields
                       |
             Structured JSON + trace
```

### Design decisions

- **Chunking:** 1,000 characters with 150-character overlap balances local context with retrieval precision. Headings are carried as section metadata; tables are emitted separately so rows and specifications remain together.
- **Embeddings:** `BAAI/bge-small-en-v1.5` by default. Query text is prefixed with the model's retrieval instruction.
- **Keyword retrieval:** BM25 preserves exact identifiers such as bid numbers, model IDs, and SKUs.
- **Fusion:** RRF with `k=60` combines dense and keyword rankings without requiring score calibration.
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks the fused candidate pool.
- **Metadata:** every chunk records bid ID, file, document type, addendum number, page, section, and document date where available.
- **Addenda:** reconciliation loads all addendum chunks for the current bid and processes addenda in ascending addendum-number order. A change is applied only when its cited chunk is indexed as an addendum and the chunk's addendum number agrees with the change. Page number is never treated as addendum number.
- **Failure safety:** a supported source file that cannot be parsed causes ingestion to fail closed. The indexer does not replace a bid index when parsing yields no pages/chunks. The orchestrator routes failed ingestion directly to a failure record instead of extracting from stale index contents.
- **Validation:** non-null values need resolvable citations matching the cited file/page; date, numeric, and identifier tokens are checked against cited text. This is a deterministic critic, not a guarantee of semantic truth; human review remains appropriate for high-impact bid decisions.

## 5. Structured output

The extraction schema contains the 20 required fields:

`Bid Number`, `Title`, `Due Date`, `Bid Submission Type`, `Term of Bid`, `Pre Bid Meeting`, `Installation`, `Bid Bond Requirement`, `Delivery Date`, `Payment Terms`, `Any Additional Documentation Required`, `MFG for Registration`, `Contract or Cooperative to use`, `Model_no`, `Part_no`, `Product`, `contact_info`, `company_name`, `Bid Summary`, `Product Specification`.

Each field is represented with `value`, `sources`, `confidence`, and `notes`. Bid-level JSON also records `addendum_changes`, `validation`, `errors`, and `ingestion_ok`. Agent traces are written separately to `outputs/trace_<bid_id>.json` and `logs/trace_<bid_id>.json`.

## 6. Retrieval evaluation

### Dataset

`eval/qa_questions.json` contains **27 judged query-to-passage pairs**: 12 for Bid 1 and 15 for Bid 2. Each row includes a query, bid filter, expected source-file substring, expected page, exact passage phrase, and an expected chunk ID. The corpus was built from the uploaded source documents in `eval/corpus/` and contains 75 non-empty source pages and 253 evaluation chunks.

### Offline reproducible benchmark

Run:

```bash
python -m src.eval.offline_benchmark
```

This benchmark uses scikit-learn TF-IDF cosine as a lightweight vector-style baseline, a local BM25 implementation, and a deterministic lexical second-stage reranker. It is intentionally dependency-light and **is not a substitute for the production SentenceTransformer embeddings or CrossEncoder**. Results on the checked-in 27-pair dataset:

| Configuration | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|
| TF-IDF vector-style baseline (offline) | 0.8148 | 0.6023 | 0.6721 |
| TF-IDF + BM25 RRF hybrid (offline) | 0.7778 | 0.6299 | 0.6779 |
| Hybrid + deterministic lexical rerank (offline) | 0.7778 | 0.7050 | 0.7326 |

The offline reranker improves ranking quality (MRR and nDCG) in this run, but does not improve Recall@5 over the vector baseline. The hybrid's Recall@5 is lower than the vector-only baseline, showing that fusion/reranking parameters should be tuned rather than assuming hybrid always wins. Missed pair IDs are preserved in `eval/offline_results.json`.

### Production-stack evaluation

After installing `requirements.txt` and indexing the two bid folders, run:

```bash
python -m src.eval.run_eval
```

This executes the actual configured production modes: `vector`, `hybrid`, and `hybrid_rerank`, and writes `eval/production_results.json`. The offline metrics above must not be presented as production embedding/CrossEncoder metrics. Production results should be regenerated in the target runtime, because the execution environment used to prepare this project did not have ChromaDB, rank-bm25, SentenceTransformers, or LangGraph installed.

## 7. Tests and verification

```bash
python -m compileall -q config.py main.py src tests
python -m pytest -q
```

The current unit suite covers document classification, addendum numbering, reconciliation evidence requirements, page-vs-addendum numbering, citation resolution, due-date evidence, LLM provider configuration, and safe indexer behavior when parsing returns no pages.

## 8. Known limitations / next validation steps

- OCR is not implemented. A scanned PDF with no extractable text is reported and fails closed rather than being indexed as an empty document.
- The production retrieval benchmark still needs to be run after installing the full requirements in the deployment/demo environment; see Section 6.
- The offline reranker is a deterministic diagnostic baseline, not a neural cross-encoder.
- The HTML BidNet pages can vary in what the portal makes available; the parser indexes only the content present in the downloaded HTML file.
- Review generated extraction JSON and citations against source documents before relying on them for an actual bid submission.
