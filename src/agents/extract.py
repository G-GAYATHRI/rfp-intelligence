from __future__ import annotations

from src.agents.llm import LLMCallError, chat_json
from src.agents.retrieve import format_evidence
from src.agents.schema import blank_field

SYSTEM = """You extract fields from RFP evidence passages.
Rules:
- Use only the passages provided. Do not use outside knowledge.
- If a passage does not state the fact, set value to null and notes to "Not found in documents".
- Every non-null value must list one or more source_chunk_ids copied exactly from the evidence labels.
- Prefer the latest addendum over the base RFP when both state the same field.
- For Pre Bid Meeting, use "None" only when the evidence says there is no meeting.
- For Bid Bond Requirement, use "Not required" only when the evidence says no bond or no bid security.
- Return JSON only, with this shape:
{"fields": {"Field Name": {"value": null, "confidence": 0.0, "notes": "", "source_chunk_ids": []}}}
"""


def bind_field(payload, evidence: dict, field_name: str | None = None) -> dict:
    if not isinstance(payload, dict):
        return blank_field()
    value = payload.get("value")
    if isinstance(value, str):
        lowered = value.strip().lower()
        drop = {"", "null", "not found", "not found in documents"}
        if field_name != "Pre Bid Meeting":
            drop.add("none")
        if lowered in drop:
            value = None
    notes = str(payload.get("notes") or "").strip()
    try:
        confidence = float(payload.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    sources = []
    for cid in payload.get("source_chunk_ids") or []:
        hit = evidence.get(cid)
        if hit:
            sources.append({"file": hit["file"], "page": hit["page"], "chunk_id": cid})
    if value is not None and not sources:
        return blank_field("Rejected: value had no citation in the retrieved evidence.")
    if value is None and not notes:
        notes = "Not found in documents"
    return {"value": value, "sources": sources, "confidence": confidence, "notes": notes}


def extract_fields(field_names: list[str], hits: list[dict], feedback: dict | None = None) -> tuple[dict, dict]:
    evidence = {hit["chunk_id"]: hit for hit in hits}
    if not hits:
        return {name: blank_field() for name in field_names}, {}
    feedback = feedback or {}
    corrections = {
        name: feedback[name] for name in field_names if feedback.get(name)
    }
    user = (
        "Extract these fields: "
        + ", ".join(field_names)
        + ".\n"
    )
    if corrections:
        user += f"A previous draft was rejected. Fix only these issues: {corrections}\n"
    user += "Evidence:\n" + format_evidence(hits)
    try:
        payload, usage = chat_json(SYSTEM, user)
    except LLMCallError as exc:
        note = f"Extraction error: {exc}"
        return {name: blank_field(note) for name in field_names}, {}
    raw_fields = payload.get("fields") or {}
    return {
        name: bind_field(raw_fields.get(name), evidence, name) for name in field_names
    }, usage


SUMMARY_SYSTEM = """Write a Bid Summary of 3 to 6 sentences using only the evidence and the extracted fields.
Do not add facts that are not supported. Return JSON only:
{"fields": {"Bid Summary": {"value": "", "confidence": 0.0, "notes": "", "source_chunk_ids": []}}}
"""


def extract_summary(fields: dict, hits: list[dict]) -> tuple[dict, dict]:
    evidence = {hit["chunk_id"]: hit for hit in hits}
    if not hits:
        return blank_field(), {}
    brief = {
        name: item.get("value")
        for name, item in fields.items()
        if item.get("value") is not None and name != "Bid Summary"
    }
    user = f"Extracted fields:\n{brief}\n\nEvidence:\n{format_evidence(hits[:8])}"
    try:
        payload, usage = chat_json(SUMMARY_SYSTEM, user)
    except LLMCallError as exc:
        return blank_field(f"Extraction error: {exc}"), {}
    raw = (payload.get("fields") or {}).get("Bid Summary")
    return bind_field(raw, evidence, "Bid Summary"), usage
