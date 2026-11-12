"""Question answering grounded in retrieved passages."""

from __future__ import annotations

from src.agents.extract import bind_field
from src.agents.llm import LLMCallError, LLMConfigError, chat_json
from src.agents.retrieve import format_evidence

SYSTEM = """You answer questions about bid documents.
Use only the evidence passages. If they do not contain the answer, set answer to "Not found in documents".
Every answer other than "Not found in documents" must include source_chunk_ids copied from the evidence.
Return JSON only: {"answer": "", "confidence": 0.0, "notes": "", "source_chunk_ids": []}
"""


def infer_bid_id(question: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    q = question.lower()
    if any(phrase in q for phrase in ("both bids", "compare", "each bid", "all bids")):
        return None
    from src.search.store import list_bids
    matches = [bid for bid in list_bids() if bid.lower() in q]
    return matches[0] if len(matches) == 1 else None


def answer_question(question: str, bid_id: str | None = None, top_k: int = 6) -> dict:
    from src.search.hybrid import search

    scoped = infer_bid_id(question, bid_id)
    try:
        hits = search(question, top_k=top_k, mode="hybrid_rerank", bid_id=scoped)
    except Exception as exc:
        return {
            "question": question,
            "answer": "Not found in documents",
            "sources": [],
            "confidence": 0.0,
            "notes": f"Retrieval error: {exc}",
            "bid_id": scoped,
        }
    if not hits:
        return {
            "question": question,
            "answer": "Not found in documents",
            "sources": [],
            "confidence": 0.0,
            "notes": "No matching passages",
            "bid_id": scoped,
        }
    evidence = {hit["chunk_id"]: hit for hit in hits}
    try:
        payload, usage = chat_json(SYSTEM, f"Question: {question}\n\nEvidence:\n{format_evidence(hits)}")
    except (LLMCallError, LLMConfigError) as exc:
        return {
            "question": question,
            "answer": "Not found in documents",
            "sources": [],
            "confidence": 0.0,
            "notes": f"Extraction error: {exc}",
            "bid_id": scoped,
            "evidence": hits,
        }
    bound = bind_field(
        {
            "value": payload.get("answer"),
            "confidence": payload.get("confidence"),
            "notes": payload.get("notes"),
            "source_chunk_ids": payload.get("source_chunk_ids"),
        },
        evidence,
    )
    answer = bound["value"] or "Not found in documents"
    if answer != "Not found in documents" and not bound["sources"]:
        answer = "Not found in documents"
        bound["notes"] = "Rejected: answer had no citation."
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"file": s["file"], "page": s["page"], "chunk_id": s["chunk_id"]}
            for s in bound["sources"]
        ],
        "confidence": bound["confidence"],
        "notes": bound["notes"],
        "bid_id": scoped,
        "usage": usage,
        "evidence": hits,
    }
