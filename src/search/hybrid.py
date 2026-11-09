import json
import re
from functools import lru_cache
from typing import Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config
from . import store
from .query import expand_query

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Keeps IDs whole (JA-207652, WD22TB4) AND adds their parts."""
    toks: list[str] = []
    for t in TOKEN_RE.findall(text.lower()):
        toks.append(t)
        parts = re.split(r"[-_./]", t)
        if len(parts) > 1:
            toks.extend(parts)
    return toks


def build_where(bid_id=None, doc_type=None, addendum_number=None) -> Optional[dict]:
    conds = []
    if bid_id:
        conds.append({"bid_id": bid_id})
    if doc_type:
        conds.append({"doc_type": doc_type})
    if addendum_number is not None:
        conds.append({"addendum_number": addendum_number})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


# ---------- keyword (BM25) ----------
_bm25_cache: dict = {}


def _bm25_index(where):
    key = (json.dumps(where, sort_keys=True, default=str), store.VERSION)
    if key not in _bm25_cache:
        if len(_bm25_cache) > 20:
            _bm25_cache.clear()
        chunks = store.get_chunks(where)
        kept, corpus = [], []
        for chunk in chunks:
            tokens = tokenize(chunk.get("section", "") + " " + chunk.get("text", ""))
            if tokens:
                kept.append(chunk)
                corpus.append(tokens)
        _bm25_cache[key] = (kept, BM25Okapi(corpus) if corpus else None)
    return _bm25_cache[key]


def keyword_search(query: str, top_k: int, where=None) -> list[dict]:
    chunks, bm25 = _bm25_index(where)
    if bm25 is None:
        return []
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in order if scores[i] > 0]


# ---------- fusion ----------
def rrf(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) over all lists."""
    fused: dict[str, float] = {}
    items: dict[str, dict] = {}
    for lst in result_lists:
        for rank, r in enumerate(lst, start=1):
            cid = r["chunk_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            items.setdefault(cid, r)
    return [{**items[c], "score": fused[c]}
            for c in sorted(fused, key=fused.get, reverse=True)]


# ---------- re-ranking ----------
@lru_cache(maxsize=1)
def _cross() -> CrossEncoder:
    return CrossEncoder(config.RERANK_MODEL)


def rerank(query: str, cands: list[dict], top_k: int) -> list[dict]:
    if not cands:
        return []
    scores = _cross().predict(
        [(query, f"{c['section']}\n{c['text']}") for c in cands])
    ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
    return [{**c, "score": float(s)} for c, s in ranked[:top_k]]


# ---------- public API ----------
def _cite(r: dict) -> dict:
    return {
        "bid_id": r["bid_id"], "file": r["file_name"], "page": r["page_number"],
        "doc_type": r["doc_type"],
        "addendum_number": r["addendum_number"] or None,
        "section": r.get("section", ""), "text": r["text"],
        "doc_date": r.get("doc_date") or None,
        "score": round(float(r["score"]), 4), "chunk_id": r["chunk_id"],
    }


def search(query: str, top_k: int = 5, mode: str = "hybrid_rerank",
           bid_id=None, doc_type=None, addendum_number=None,
           expand: bool = True, pool: int = 20) -> list[dict]:
    """mode: 'vector' | 'hybrid' | 'hybrid_rerank'"""
    if not query or not str(query).strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if pool < top_k:
        pool = top_k
    where = build_where(bid_id, doc_type, addendum_number)
    q = expand_query(query) if expand else query

    if mode == "vector":
        results = store.vector_search(q, top_k, where)
    elif mode in ("hybrid", "hybrid_rerank"):
        fused = rrf([store.vector_search(q, pool, where),
                     keyword_search(q, pool, where)])
        results = (rerank(query, fused[:pool], top_k)
                   if mode == "hybrid_rerank" else fused[:top_k])
    else:
        raise ValueError(f"unknown mode: {mode}")
    return [_cite(r) for r in results]