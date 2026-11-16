
from __future__ import annotations

import re


def _source(hit: dict) -> dict:
    return {
        "file": hit.get("file"),
        "page": hit.get("page"),
        "chunk_id": hit.get("chunk_id"),
    }


def _is_addendum(hit: dict) -> bool:
    return (
        str(hit.get("doc_type", "")).lower() == "addendum"
        and hit.get("addendum_number") is not None
    )


def _is_base_rfp(hit: dict) -> bool:
    return str(hit.get("doc_type", "")).lower() == "rfp"


def _clean_value(value):
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "null",
        "none",
        "not stated",
        "n/a",
    }:
        return None
    return value


def _normalise_for_match(value) -> str:
    """Normalise text for conservative source-value verification."""
    if value is None:
        return ""

    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _value_supported(value, hits: list[dict]) -> bool:
    """
    Verify that the complete proposed value occurs in a source passage.
    Uses consecutive token matching to prevent partial numeric matches
    such as 10 matching 100.
    """
    value_tokens = re.findall(r"[a-z0-9]+", str(value).casefold())

    if not value_tokens:
        return False

    for hit in hits:
        text_tokens = re.findall(
            r"[a-z0-9]+",
            str(hit.get("text", "")).casefold(),
        )

        window_size = len(value_tokens)

        for index in range(len(text_tokens) - window_size + 1):
            if text_tokens[index:index + window_size] == value_tokens:
                return True

    return False


def apply_addendum_changes(
    fields: dict,
    payload: dict,
    evidence: dict,
) -> tuple[dict, list]:
    updated = {
        name: dict(field)
        for name, field in fields.items()
    }

    changes = []
    items = list(payload.get("changes") or [])

    def sort_key(item):
        try:
            return int(item.get("addendum_number"))
        except (TypeError, ValueError):
            return 10**9

    for item in sorted(items, key=sort_key):
        if not isinstance(item, dict):
            continue

        name = item.get("field")

        if name not in updated:
            continue

        # Validate the cited addendum passages.
        valid_addendum_hits = []

        for cid in item.get("source_chunk_ids") or []:
            hit = evidence.get(cid)

            if not hit or not _is_addendum(hit):
                continue

            try:
                actual_no = int(hit.get("addendum_number"))
            except (TypeError, ValueError):
                continue

            declared = item.get("addendum_number")

            if declared is not None:
                try:
                    if int(declared) != actual_no:
                        continue
                except (TypeError, ValueError):
                    continue

            valid_addendum_hits.append(hit)

        if not valid_addendum_hits:
            continue

        addendum_numbers = {
            int(hit["addendum_number"])
            for hit in valid_addendum_hits
        }

        if len(addendum_numbers) != 1:
            continue

        add_no = next(iter(addendum_numbers))

        # Validate the cited original RFP passages.
        valid_base_hits = []

        for cid in item.get("base_source_chunk_ids") or []:
            hit = evidence.get(cid)

            if hit and _is_base_rfp(hit):
                valid_base_hits.append(hit)

        if not valid_base_hits:
            continue

        # Read and validate old and new values.
        old_value = _clean_value(item.get("old_value"))
        new_value = _clean_value(item.get("new_value"))

        if old_value is None or new_value is None:
            continue

        # The existing field must contain either the original value
        # or the new value if extraction already applied the addendum.
        current_value = _clean_value(updated[name].get("value"))

        if current_value is None:
            continue

        if _normalise_for_match(current_value) not in {
            _normalise_for_match(old_value),
            _normalise_for_match(new_value),
        }:
            continue

        # Verify that old_value is supported by the original RFP.
        if not _value_supported(old_value, valid_base_hits):
            continue

        # Verify that new_value is supported by the cited addendum.
        if not _value_supported(new_value, valid_addendum_hits):
            continue

        # Do not record a change if the old and new values are equivalent.
        if (
            _normalise_for_match(old_value)
            == _normalise_for_match(new_value)
        ):
            continue

        try:
            confidence = float(
                item.get("confidence")
                if item.get("confidence") is not None
                else 0.7
            )
        except (TypeError, ValueError):
            confidence = 0.7

        confidence = max(0.0, min(1.0, confidence))

        addendum_sources = [
            _source(hit)
            for hit in valid_addendum_hits
        ]

        base_sources = [
            _source(hit)
            for hit in valid_base_hits
        ]

        note = (
            item.get("notes")
            or f"Updated by Addendum {add_no}; previous value: {old_value}"
        )

        # Update the field even if extraction already found new_value.
        updated[name] = {
            "value": new_value,
            "sources": addendum_sources,
            "confidence": confidence,
            "notes": note,
        }

        changes.append({
            "field": name,
            "old_value": old_value,
            "new_value": new_value,
            "addendum_number": add_no,
            "old_sources": [
                {
                    "file": source["file"],
                    "page": source["page"],
                    "chunk_id": source["chunk_id"],
                }
                for source in base_sources
            ],
            "sources": [
                {
                    "file": source["file"],
                    "page": source["page"],
                    "chunk_id": source["chunk_id"],
                }
                for source in addendum_sources
            ],
            "confidence": confidence,
            "notes": note,
        })

    return updated, changes
