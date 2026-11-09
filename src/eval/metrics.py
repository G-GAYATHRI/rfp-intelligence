"""Recall@k, MRR and nDCG for judged query-to-passage pairs."""
from __future__ import annotations
import json
from pathlib import Path

def load_pairs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))

def matches(result: dict, pair: dict) -> bool:
    file_need=(pair.get("expect_file_contains") or "").lower()
    if file_need and file_need not in (result.get("file") or "").lower(): return False
    expected_page=pair.get("expect_page")
    if expected_page is not None and result.get("page") is not None and int(result["page"])!=int(expected_page): return False
    text=(result.get("text") or "").lower()
    snippets=pair.get("expect_text_contains") or []
    return all(snippet.lower() in text for snippet in snippets)

def first_hit(results: list[dict], pair: dict) -> int | None:
    for rank,result in enumerate(results,start=1):
        if matches(result,pair): return rank
    return None

def summarize(pairs: list[dict], mode: str, k: int, search_fn) -> dict:
    hits=[]; reciprocal=[]; ndcg=[]; misses=[]
    for pair in pairs:
        results=search_fn(pair["query"],top_k=k,mode=mode,bid_id=pair.get("bid_id"),expand=True)
        rank=first_hit(results,pair)
        hits.append(1.0 if rank and rank<=k else 0.0)
        reciprocal.append(1.0/rank if rank else 0.0)
        ndcg.append(1.0/(1.0 if rank==1 else __import__("math").log2(rank+1)) if rank else 0.0)
        if not rank or rank>k: misses.append(pair["id"])
    n=max(len(pairs),1)
    return {"mode":mode,"k":k,"n":len(pairs),"recall_at_k":round(sum(hits)/n,4),"mrr":round(sum(reciprocal)/n,4),"ndcg_at_k":round(sum(ndcg)/n,4),"misses":misses}

def run_evaluation(pairs: list[dict], modes: list[str], k: int=5) -> list[dict]:
    from src.search.hybrid import search
    return [summarize(pairs,mode,k,search) for mode in modes]
