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


CANDIDATE_SYSTEM = """You identify which structured RFP fields may have been changed by an addendum.

Compare the addendum passages with the list of available field names.
Return only fields for which the addendum contains explicit evidence of a possible
change, clarification, replacement, extension, removal, or newly stated requirement.

IMPORTANT:
- Do not compare the addendum only with the current extracted field values.
- Current extracted values may already reflect the addendum.
- A field can still be a candidate even when its current value matches the addendum.
- Do not invent changes. Include a field only when the addendum evidence is relevant.
- Return valid JSON only.

JSON shape:
{"candidate_fields": ["Due Date", "Submission Instructions"]}
"""


RECONCILE_SYSTEM = """You compare original/base RFP passages with addendum passages.

Your task is to identify explicit changes made by the addendum to the original RFP.

Rules:
1. Use the BASE RFP passages to establish the old/original value.
2. Use the ADDENDUM passages to establish the new/revised value.
3. Do not use the current extracted field value as a substitute for the original value.
   The current extracted value may already reflect a previous addendum.
4. Report a change even if the current extracted value already equals the new value.
5. Return only changes explicitly supported by both source documents.
6. Do not report a change when the addendum merely repeats the original value.
7. source_chunk_ids must contain exact chunk IDs from ADDENDUM passages.
8. base_source_chunk_ids must contain exact chunk IDs from BASE RFP passages.
9. Do not invent dates, values, addendum numbers, or source IDs.
10. If no supported changes exist, return {"changes": []}.

Return valid JSON only, in this shape:
{
  "changes": [
    {
      "field": "Due Date",
      "old_value": "original value from base RFP",
      "new_value": "revised value from addendum",
      "addendum_number": 1,
      "confidence": 0.0,
      "notes": "Brief evidence-based explanation",
      "base_source_chunk_ids": ["exact base RFP chunk ID"],
      "source_chunk_ids": ["exact addendum chunk ID"]
    }
  ]
}
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


def _step(
    agent: str,
    started: float,
    payload: dict,
    usage: dict | None = None,
    error: str | None = None,
) -> dict:
    return {
        "agent": agent,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "tokens": usage or {},
        "error": error,
        **payload,
    }


def _evidence_map(hits: list[dict]) -> dict:
    return {
        hit["chunk_id"]: hit
        for hit in hits
        if isinstance(hit, dict) and hit.get("chunk_id")
    }


def _normalise_field_names(candidate_payload: dict) -> list[str]:
    """Return only candidate names that exist in the project schema."""
    candidates = (
        candidate_payload.get("candidate_fields")
        or candidate_payload.get("fields")
        or []
    )

    if isinstance(candidates, dict):
        candidates = list(candidates.keys())

    if not isinstance(candidates, list):
        return []

    valid = set(FIELDS)
    result = []
    for name in candidates:
        if isinstance(name, str) and name in valid and name not in result:
            result.append(name)
    return result


def _merge_changes(existing: list, new_changes: list) -> list:
    """Preserve prior verified changes and avoid duplicate records on retries."""
    merged = []
    seen = set()

    for item in list(existing or []) + list(new_changes or []):
        if not isinstance(item, dict):
            continue

        key = (
            item.get("field"),
            item.get("addendum_number"),
            str(item.get("old_value")),
            str(item.get("new_value")),
        )
        if key in seen:
            continue

        seen.add(key)
        merged.append(item)

    return merged


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
    if error and error not in errors:
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
            message = f"Retrieval failed for {names}: {exc}"
            if message not in errors:
                errors.append(message)

        evidence.update(_evidence_map(hits))
        prepared.append((names, hits))

    with ThreadPoolExecutor(max_workers=min(4, len(prepared) or 1)) as pool:
        futures = [
            pool.submit(_extract_prepared, names, hits, feedback)
            for names, hits in prepared
        ]

        for future in futures:
            fields, group_usage, error = future.result()
            merged.update(fields)

            for key in usage:
                usage[key] += int((group_usage or {}).get(key) or 0)

            if error and error not in errors:
                errors.append(error)

    if not rejected:
        try:
            summary_hits = retrieve(bid_id, ["Bid Summary"])
        except Exception as exc:
            summary_hits = []
            message = f"Summary retrieval failed: {exc}"
            if message not in errors:
                errors.append(message)

        try:
            summary, summary_usage = extract_summary(merged, summary_hits)
        except Exception as exc:
            summary = blank_field(f"Summary extraction error: {exc}")
            summary_usage = {}
            message = f"Summary extraction failed: {exc}"
            if message not in errors:
                errors.append(message)

        merged["Bid Summary"] = summary
        evidence.update(_evidence_map(summary_hits))

        for key in usage:
            usage[key] += int((summary_usage or {}).get(key) or 0)

    trace = list(state.get("trace") or [])
    trace.append(
        _step(
            "extraction",
            started,
            {
                "input": {
                    "groups": list(groups),
                    "attempt": state.get("attempt", 0),
                },
                "output": {
                    name: value.get("value") if isinstance(value, dict) else value
                    for name, value in merged.items()
                },
            },
            usage,
        )
    )

    return {
        "fields": merged,
        "evidence": evidence,
        "trace": trace,
        "errors": errors,
    }


def reconcile_node(state: GraphState) -> dict:
    started = time.perf_counter()

    from src.agents.llm import LLMCallError, chat_json
    from src.search import store

    bid_id = state["bid_id"]
    fields = dict(state.get("fields") or {})
    evidence = dict(state.get("evidence") or {})

    # Important: do not discard changes accepted during an earlier retry.
    changes = list(state.get("addendum_changes") or [])
    errors = list(state.get("errors") or [])
    usage = {"input_tokens": 0, "output_tokens": 0}

    raw = store.get_chunks(
        {"$and": [{"bid_id": bid_id}, {"doc_type": "addendum"}]}
    )

    addenda: dict[int, list[dict]] = {}
    for item in raw:
        number = item.get("addendum_number")
        if number in (None, 0, "0", ""):
            continue

        try:
            number = int(number)
        except (TypeError, ValueError):
            continue

        hit = {
            "chunk_id": item["chunk_id"],
            "file": item.get("file_name"),
            "page": item.get("page_number"),
            "doc_type": item.get("doc_type"),
            "addendum_number": number,
            "section": item.get("section", ""),
            "text": item.get("text", ""),
        }
        evidence[hit["chunk_id"]] = hit
        addenda.setdefault(number, []).append(hit)

    print(f"\n[RECONCILE DEBUG] Bid: {bid_id}")
    print(f"[RECONCILE DEBUG] Addendum numbers found: {sorted(addenda)}")

    for number in sorted(addenda):
        addendum_hits = addenda[number]
        addendum_text = format_evidence(addendum_hits)

        # Stage 1: ask the model which schema fields the addendum may affect.
        candidate_user = (
            f"Available field names:\n{json.dumps(FIELDS, ensure_ascii=False)}"
            f"\n\nCurrent extracted values (context only; these may already include "
            f"this addendum):\n"
            f"{json.dumps({k: v.get('value') if isinstance(v, dict) else v for k, v in fields.items()}, default=str)}"
            f"\n\nAddendum {number} evidence:\n{addendum_text}"
        )

        try:
            candidate_payload, candidate_usage = chat_json(
                CANDIDATE_SYSTEM, candidate_user
            )
            for key in usage:
                usage[key] += int((candidate_usage or {}).get(key) or 0)

            candidate_names = _normalise_field_names(candidate_payload)

            print(f"\n[RECONCILE DEBUG] Addendum {number}")
            print("[RECONCILE DEBUG] Candidate fields:", candidate_names)

            if not candidate_names:
                print(
                    f"[RECONCILE DEBUG] No candidate fields identified "
                    f"for Addendum {number}."
                )
                continue

            # Stage 2: retrieve evidence for those fields, then retain only
            # passages explicitly tagged as the original/base RFP.
            base_hits = retrieve(bid_id, candidate_names, state.get("feedback") or {})
            base_hits = [
                hit for hit in base_hits
                if str(hit.get("doc_type", "")).lower() in {"rfp", "base_rfp"}
            ]

            print(
                f"[RECONCILE DEBUG] Addendum {number}: retrieved "
                f"{len(base_hits)} base-RFP chunks."
            )

            evidence.update(_evidence_map(base_hits))

            if not base_hits:
                message = (
                    f"Addendum {number}: no base-RFP evidence retrieved for "
                    f"candidate fields {candidate_names}; reconciliation skipped."
                )
                print("[RECONCILE DEBUG]", message)
                if message not in errors:
                    errors.append(message)
                continue

            # Stage 3: compare original RFP passages against addendum passages.
            reconcile_user = (
                f"Candidate fields:\n{json.dumps(candidate_names, ensure_ascii=False)}"
                f"\n\nCurrent extracted values (context only; do not treat these as "
                f"the original values):\n"
                f"{json.dumps({k: v.get('value') if isinstance(v, dict) else v for k, v in fields.items()}, default=str)}"
                f"\n\nBASE RFP evidence:\n{format_evidence(base_hits)}"
                f"\n\nADDENDUM {number} evidence:\n{addendum_text}"
            )

            payload, reconcile_usage = chat_json(
                RECONCILE_SYSTEM, reconcile_user
            )
            for key in usage:
                usage[key] += int((reconcile_usage or {}).get(key) or 0)

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Addendum {number} reconciliation returned a non-object payload."
                )

            for change in payload.get("changes") or []:
                if isinstance(change, dict):
                    change["addendum_number"] = number

            print(
                f"[RECONCILE DEBUG] Verification result for Addendum {number}:"
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

            fields, accepted_changes = apply_addendum_changes(
                fields, payload, evidence
            )
            changes = _merge_changes(changes, accepted_changes)

            print(
                f"[RECONCILE DEBUG] Addendum {number}: "
                f"{len(accepted_changes)} validated change(s) accepted."
            )

        except (LLMCallError, LLMConfigError) as exc:
            message = f"Addendum {number} LLM reconciliation error: {exc}"
            if message not in errors:
                errors.append(message)
            print("[RECONCILE DEBUG]", message)
            break
        except Exception as exc:
            message = f"Addendum {number} reconciliation failed: {exc}"
            if message not in errors:
                errors.append(message)
            print("[RECONCILE DEBUG]", message)
            break

    trace = list(state.get("trace") or [])
    new_error = (
        errors[-1]
        if len(errors) > len(state.get("errors") or [])
        else None
    )
    trace.append(
        _step(
            "addendum_reconciliation",
            started,
            {
                "input": {"addenda_processed": sorted(addenda)},
                "output": changes,
            },
            usage,
            error=new_error,
        )
    )

    return {
        "fields": fields,
        "addendum_changes": changes,
        "evidence": evidence,
        "trace": trace,
        "errors": errors,
    }


def validate_node(state: GraphState) -> dict:
    started = time.perf_counter()

    report, rejected = validate_fields(
        state.get("fields") or {},
        state.get("evidence") or {},
    )
    counts = validation_counts(state.get("fields") or {}, rejected)

    trace = list(state.get("trace") or [])
    trace.append(
        _step(
            "validator",
            started,
            {"output": {"rejected": rejected, "counts": counts}},
        )
    )

    return {
        "rejected": rejected,
        "feedback": report,
        "validation": counts,
        "trace": trace,
    }


def route_after_validate(state: GraphState) -> str:
    if (
        state.get("rejected")
        and int(state.get("attempt") or 0) < MAX_RETRIES
    ):
        config_error = any(
            (
                "LLM_API_KEY" in err
                or "_API_KEY" in err
                or "LLM_PROVIDER" in err
            )
            for err in (state.get("errors") or [])
        )
        if config_error:
            return "finalize"
        return "retry"

    return "finalize"


def retry_node(state: GraphState) -> dict:
    started = time.perf_counter()
    attempt = int(state.get("attempt") or 0) + 1

    trace = list(state.get("trace") or [])
    trace.append(
        _step(
            "orchestrator",
            started,
            {
                "output": {
                    "retry": attempt,
                    "fields": state.get("rejected") or [],
                }
            },
        )
    )

    return {"attempt": attempt, "trace": trace}


def finalize_node(state: GraphState) -> dict:
    started = time.perf_counter()

    bid_id = state.get("bid_id") or Path(state["folder"]).name
    fields = state.get("fields") or {
        name: blank_field() for name in FIELDS
    }

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

    output_dir = Path(config.OUTPUT_DIR)
    log_dir = Path(config.LOG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{bid_id}.json"
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    trace = list(state.get("trace") or [])
    trace.append(
        _step(
            "report",
            started,
            {
                "output": {
                    "path": str(path),
                    "validation": record["validation"],
                }
            },
        )
    )

    trace_path = output_dir / f"trace_{bid_id}.json"
    trace_json = json.dumps(trace, indent=2, ensure_ascii=False, default=str)
    trace_path.write_text(trace_json, encoding="utf-8")
    (log_dir / f"trace_{bid_id}.json").write_text(
        trace_json,
        encoding="utf-8",
    )

    return {
        "output_path": str(path),
        "validation": record["validation"],
        "trace": trace,
        "fields": fields,
    }


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("extract", extract_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "ingest")

    graph.add_conditional_edges(
        "ingest",
        lambda state: (
            "extract" if state.get("ingestion_ok") else "finalize"
        ),
        {
            "extract": "extract",
            "finalize": "finalize",
        },
    )

    graph.add_edge("extract", "reconcile")
    graph.add_edge("reconcile", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "retry": "retry",
            "finalize": "finalize",
        },
    )

    graph.add_edge("retry", "extract")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_extraction(folder: str | Path) -> dict:
    folder = Path(folder)

    app = build_graph()
    final = app.invoke(
        {
            "folder": str(folder),
            "trace": [],
            "errors": [],
        }
    )

    record_path = Path(final["output_path"])
    return json.loads(record_path.read_text(encoding="utf-8"))
