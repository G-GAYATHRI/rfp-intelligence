from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import config
from src.agents.extract import extract_fields, extract_summary
from src.agents.llm import LLMConfigError
from src.agents.reconcile import apply_addendum_changes
from src.agents.retrieve import format_evidence, retrieve
from src.agents.schema import FIELDS, GROUPS, blank_field
from src.agents.validate import validate_fields, validation_counts
from src.search.indexer import index_bid_folder

MAX_RETRIES = 2

RECONCILE_SYSTEM = """You compare a base RFP with its addenda.
Return only real changes stated in an addendum passage. Do not invent changes.
source_chunk_ids must be addendum passages, copied exactly.
If nothing changed, return {"changes": []}.
JSON shape:
{"changes": [{"field": "", "old_value": null, "new_value": null, "addendum_number": 1,
"confidence": 0.0, "notes": "", "source_chunk_ids": []}]}
"""


class GraphState(TypedDict, total=False):
    folder: str
    bid_id: str
    fields: dict
    evidence: dict
    feedback: dict
    rejected: list
    attempt: int
    addendum_changes: list
    validation: dict
    trace: list
    errors: list
    output_path: str
    ingestion_ok: bool


def _step(agent: str, started: float, payload: dict, usage: dict | None = None, error: str | None = None) -> dict:
    return {
        "agent": agent,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "tokens": usage or {},
        "error": error,
        **payload,
    }


def _evidence_map(hits: list[dict]) -> dict:
    return {hit["chunk_id"]: hit for hit in hits}


def ingest_node(state: GraphState) -> dict:
    started = time.perf_counter()
    folder = Path(state["folder"])
    try:
        stats = index_bid_folder(folder)
        error = None
    except Exception as exc:
        stats = {}
        error = str(exc)
    errors = list(state.get("errors") or [])
    if error:
        errors.append(error)
    trace = list(state.get("trace") or [])
    trace.append(_step("ingestion", started, {"output": stats}, error=error))
    return {
        "bid_id": folder.name,
        "fields": {name: blank_field("Not extracted") for name in FIELDS},
        "evidence": {},
        "feedback": {},
        "rejected": [],
        "attempt": 0,
        "addendum_changes": [],
        "trace": trace,
        "errors": errors,
        "ingestion_ok": error is None,
    }


def _extract_prepared(names: list[str], hits: list[dict], feedback: dict):
    try:
        fields, usage = extract_fields(names, hits, feedback)
        return fields, usage, None
    except LLMConfigError as exc:
        return {name: blank_field(str(exc)) for name in names}, {}, str(exc)
    except Exception as exc:
        note = f"Extraction error: {exc}"
        return {name: blank_field(note) for name in names}, {}, str(exc)


def extract_node(state: GraphState) -> dict:
    started = time.perf_counter()
    bid_id = state["bid_id"]
    feedback = state.get("feedback") or {}
    rejected = state.get("rejected") or []
    groups = {"retry": rejected} if rejected else GROUPS
    merged = dict(state.get("fields") or {})
    evidence = dict(state.get("evidence") or {})
    usage = {"input_tokens": 0, "output_tokens": 0}
    errors = list(state.get("errors") or [])
    prepared = []
    for names in groups.values():
        try:
            hits = retrieve(bid_id, names, feedback)
        except Exception as exc:
            hits = []
            errors.append(f"Retrieval failed for {names}: {exc}")
        evidence.update(_evidence_map(hits))
        prepared.append((names, hits))

    with ThreadPoolExecutor(max_workers=min(4, len(prepared) or 1)) as pool:
        futures = [pool.submit(_extract_prepared, names, hits, feedback) for names, hits in prepared]
        for future in futures:
            fields, group_usage, error = future.result()
            merged.update(fields)
            for key in usage:
                usage[key] += int(group_usage.get(key) or 0)
            if error and error not in errors:
                errors.append(error)

    if not rejected:
        try:
            summary_hits = retrieve(bid_id, ["Bid Summary"])
        except Exception as exc:
            summary_hits = []
            errors.append(f"Summary retrieval failed: {exc}")
        try:
            summary, summary_usage = extract_summary(merged, summary_hits)
        except LLMConfigError as exc:
            summary, summary_usage = blank_field(str(exc)), {}
            if str(exc) not in errors:
                errors.append(str(exc))
        merged["Bid Summary"] = summary
        evidence.update(_evidence_map(summary_hits))
        for key in usage:
            usage[key] += int(summary_usage.get(key) or 0)

    trace = list(state.get("trace") or [])
    trace.append(_step(
        "extraction",
        started,
        {"input": {"groups": list(groups), "attempt": state.get("attempt", 0)},
         "output": {name: merged[name]["value"] for name in merged}},
        usage,
    ))
    return {"fields": merged, "evidence": evidence, "trace": trace, "errors": errors}


def reconcile_node(state: GraphState) -> dict:
    started = time.perf_counter()
    from src.agents.llm import LLMCallError, chat_json
    from src.search import store
    bid_id = state["bid_id"]
    fields = dict(state.get("fields") or {})
    evidence = dict(state.get("evidence") or {})
    changes = []
    errors = list(state.get("errors") or [])
    usage = {"input_tokens": 0, "output_tokens": 0}
    raw = store.get_chunks({"$and": [{"bid_id": bid_id}, {"doc_type": "addendum"}]})
    addenda = {}
    for item in raw:
        number = item.get("addendum_number")
        if number in (None, 0, "0", ""): continue
        hit = {"chunk_id": item["chunk_id"], "file": item.get("file_name"),
               "page": item.get("page_number"), "doc_type": item.get("doc_type"),
               "addendum_number": int(number), "section": item.get("section", ""),
               "text": item.get("text", "")}
        evidence[hit["chunk_id"]] = hit
        addenda.setdefault(int(number), []).append(hit)
    for number in sorted(addenda):
        hits = addenda[number]
        user = (f"Current field values:\n{json.dumps({k:v.get('value') for k,v in fields.items()}, default=str)}"
                f"\n\nReview ALL evidence from Addendum {number}. Only report explicit changes."
                f"\n\nAddendum evidence:\n{format_evidence(hits)}")
        try:
            payload, one_usage = chat_json(RECONCILE_SYSTEM, user)
            for key in usage: usage[key] += int(one_usage.get(key) or 0)
            for item in payload.get("changes") or []:
                item["addendum_number"] = number
            fields, one_changes = apply_addendum_changes(fields, payload, evidence)
            changes.extend(one_changes)
        except (LLMCallError, LLMConfigError) as exc:
            if str(exc) not in errors: errors.append(str(exc))
            break
        except Exception as exc:
            if str(exc) not in errors: errors.append(f"Addendum {number} reconciliation failed: {exc}")
            break
    trace = list(state.get("trace") or [])
    trace.append(_step("addendum_reconciliation", started,
        {"input": {"addenda_processed": sorted(addenda)}, "output": changes}, usage,
        error=(errors[-1] if len(errors)>(len(state.get("errors") or [])) else None)))
    return {"fields": fields, "addendum_changes": changes, "evidence": evidence,
            "trace": trace, "errors": errors}


def validate_node(state: GraphState) -> dict:
    started = time.perf_counter()
    report, rejected = validate_fields(state.get("fields") or {}, state.get("evidence") or {})
    counts = validation_counts(state.get("fields") or {}, rejected)
    trace = list(state.get("trace") or [])
    trace.append(_step("validator", started, {"output": {"rejected": rejected, "counts": counts}}))
    return {"rejected": rejected, "feedback": report, "validation": counts, "trace": trace}


def route_after_validate(state: GraphState) -> str:
    if state.get("rejected") and int(state.get("attempt") or 0) < MAX_RETRIES:
        config_error = any(("LLM_API_KEY" in err or "_API_KEY" in err or "LLM_PROVIDER" in err)
                           for err in (state.get("errors") or []))
        if config_error:
            return "finalize"
        return "retry"
    return "finalize"


def retry_node(state: GraphState) -> dict:
    started = time.perf_counter()
    attempt = int(state.get("attempt") or 0) + 1
    trace = list(state.get("trace") or [])
    trace.append(_step("orchestrator", started, {
        "output": {"retry": attempt, "fields": state.get("rejected") or []},
    }))
    return {"attempt": attempt, "trace": trace}


def finalize_node(state: GraphState) -> dict:
    started = time.perf_counter()
    bid_id = state.get("bid_id") or Path(state["folder"]).name
    fields = state.get("fields") or {name: blank_field() for name in FIELDS}
    for name in FIELDS:
        fields.setdefault(name, blank_field())
    rejected = state.get("rejected") or []
    record = {
        "bid_id": bid_id,
        "fields": fields,
        "addendum_changes": state.get("addendum_changes") or [],
        "validation": validation_counts(fields, rejected),
        "errors": state.get("errors") or [],
        "ingestion_ok": bool(state.get("ingestion_ok", True)),
    }
    path = config.OUTPUT_DIR / f"{bid_id}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    trace = list(state.get("trace") or [])
    trace.append(_step("report", started, {"output": {"path": str(path), "validation": record["validation"]}}))
    trace_path = config.OUTPUT_DIR / f"trace_{bid_id}.json"
    trace_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    (config.LOG_DIR / f"trace_{bid_id}.json").write_text(
        json.dumps(trace, indent=2, default=str), encoding="utf-8")
    return {"output_path": str(path), "validation": record["validation"], "trace": trace, "fields": fields}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("extract", extract_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", lambda state: "extract" if state.get("ingestion_ok") else "finalize",
                                {"extract": "extract", "finalize": "finalize"})
    graph.add_edge("extract", "reconcile")
    graph.add_edge("reconcile", "validate")
    graph.add_conditional_edges("validate", route_after_validate, {"retry": "retry", "finalize": "finalize"})
    graph.add_edge("retry", "extract")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_extraction(folder: str | Path) -> dict:
    folder = Path(folder)
    app = build_graph()
    final = app.invoke({"folder": str(folder), "trace": [], "errors": []})
    record_path = Path(final["output_path"])
    return json.loads(record_path.read_text(encoding="utf-8"))
