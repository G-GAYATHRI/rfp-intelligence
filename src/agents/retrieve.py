"""Retrieval agent: expand a field question, filter to one bid, return cited passages."""

from __future__ import annotations

from src.agents.schema import FIELD_QUERIES


def retrieve(bid_id: str, fields: list[str], feedback: dict | None = None,
             per_query: int = 4, limit: int = 10) -> list[dict]:
    from src.search.hybrid import search

    hits: dict[str, dict] = {}
    feedback = feedback or {}
    for name in fields:
        query = FIELD_QUERIES.get(name, name)
        if feedback.get(name):
            query = f"{query} {name}"
        # Let failures reach the orchestrator so they are recorded in the trace;
        # silently converting an index/model failure into "not found" is misleading.
        found = search(query, top_k=per_query, mode="hybrid", bid_id=bid_id)
        for hit in found:
            hits.setdefault(hit["chunk_id"], hit)
    if any(name in {"Due Date", "Bid Bond Requirement", "Payment Terms", "Pre Bid Meeting"} for name in fields):
        for hit in search(
            "addendum amendment change revised date",
            top_k=per_query, mode="hybrid", bid_id=bid_id, doc_type="addendum",
        ):
            hits.setdefault(hit["chunk_id"], hit)
    return list(hits.values())[:limit]


def format_evidence(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            f"[{hit['chunk_id']}] file={hit['file']} page={hit['page']} "
            f"doc_type={hit['doc_type']} addendum={hit.get('addendum_number')} "
            f"section={hit.get('section')}\n{hit['text'][:900]}"
        )
    return "\n\n".join(blocks)
