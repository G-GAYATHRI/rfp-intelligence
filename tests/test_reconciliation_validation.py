from src.agents.reconcile import apply_addendum_changes
from src.agents.validate import validate_fields


def test_addendum_requires_real_addendum_metadata_and_never_uses_page_as_number():
    fields = {"Due Date": {"value": "June 27, 2024", "sources": [], "confidence": 0.8, "notes": "base"}}
    evidence = {
        "base": {"chunk_id": "base", "file": "RFP.pdf", "page": 3, "doc_type": "rfp", "text": "June 27, 2024"},
        "add": {"chunk_id": "add", "file": "Addendum 2.pdf", "page": 1, "doc_type": "addendum", "addendum_number": 2, "text": "July 9, 2024"},
    }
    payload = {"changes": [
        {"field": "Due Date", "new_value": "July 9, 2024", "source_chunk_ids": ["base"], "confidence": .9},
    ]}
    updated, changes = apply_addendum_changes(fields, payload, evidence)
    assert updated["Due Date"]["value"] == "June 27, 2024"
    assert changes == []


def test_addendum_change_uses_indexed_number_not_page_number():
    fields = {"Due Date": {"value": "June 27, 2024", "sources": [], "confidence": .8, "notes": "base"}}
    evidence = {"a": {"chunk_id": "a", "file": "Addendum No. 2.pdf", "page": 5,
                      "doc_type": "addendum", "addendum_number": 2, "text": "July 9, 2024"}}
    payload = {"changes": [{"field": "Due Date", "new_value": "July 9, 2024", "addendum_number": 2,
                            "source_chunk_ids": ["a"]}]}
    updated, changes = apply_addendum_changes(fields, payload, evidence)
    assert updated["Due Date"]["value"] == "July 9, 2024"
    assert changes[0]["addendum_number"] == 2


def test_validator_rejects_missing_or_mismatched_citation():
    fields = {"Due Date": {"value": "July 9, 2024", "sources": [
        {"file": "wrong.pdf", "page": 1, "chunk_id": "c1"}], "notes": ""}}
    evidence = {"c1": {"file": "right.pdf", "page": 1, "text": "July 9, 2024"}}
    report, rejected = validate_fields(fields, evidence)
    assert "Due Date" in rejected
    assert "Citation does not resolve" in report["Due Date"][0]


def test_validator_accepts_exactly_supported_cited_due_date():
    fields = {"Due Date": {"value": "July 9, 2024", "sources": [
        {"file": "Addendum 2.pdf", "page": 1, "chunk_id": "c1"}], "notes": ""}}
    evidence = {"c1": {"file": "Addendum 2.pdf", "page": 1,
                       "text": "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST."}}
    report, rejected = validate_fields(fields, evidence)
    assert report == {}
    assert rejected == []
