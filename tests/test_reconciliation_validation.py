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
    fields = {
        "Due Date": {
            "value": "June 27, 2024",
            "sources": [],
            "confidence": 0.8,
            "notes": "base",
        }
    }

    evidence = {
        "base": {
            "chunk_id": "base",
            "file": "RFP.pdf",
            "page": 3,
            "doc_type": "rfp",
            "text": "The proposal due date is June 27, 2024.",
        },
        "add": {
            "chunk_id": "add",
            "file": "Addendum No. 2.pdf",
            "page": 5,
            "doc_type": "addendum",
            "addendum_number": 2,
            "text": "The new proposal due date is July 9, 2024.",
        },
    }

    payload = {
        "changes": [
            {
                "field": "Due Date",
                "old_value": "June 27, 2024",
                "new_value": "July 9, 2024",
                "addendum_number": 2,
                "base_source_chunk_ids": ["base"],
                "source_chunk_ids": ["add"],
                "confidence": 0.9,
            }
        ]
    }

    updated, changes = apply_addendum_changes(fields, payload, evidence)

    assert updated["Due Date"]["value"] == "July 9, 2024"
    assert len(changes) == 1
    assert changes[0]["addendum_number"] == 2
    assert changes[0]["old_value"] == "June 27, 2024"



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


def test_reconciliation_rejects_old_value_that_differs_from_current_field():
    fields = {
        "Due Date": {
            "value": "June 27, 2024",
            "sources": [],
            "confidence": 0.8,
            "notes": "base",
        }
    }

    evidence = {
        "base": {
            "doc_type": "rfp",
            "text": "The proposal due date is June 26, 2024.",
        },
        "add": {
            "doc_type": "addendum",
            "addendum_number": 2,
            "text": "The new proposal due date is July 9, 2024.",
        },
    }

    payload = {
        "changes": [{
            "field": "Due Date",
            "old_value": "June 26, 2024",
            "new_value": "July 9, 2024",
            "addendum_number": 2,
            "base_source_chunk_ids": ["base"],
            "source_chunk_ids": ["add"],
        }]
    }

    updated, changes = apply_addendum_changes(fields, payload, evidence)

    assert updated["Due Date"]["value"] == "June 27, 2024"
    assert changes == []


def test_reconciliation_rejects_partial_numeric_value_match():
    fields = {
        "Quantity": {
            "value": "5",
            "sources": [],
            "confidence": 0.8,
            "notes": "base",
        }
    }

    evidence = {
        "base": {
            "doc_type": "rfp",
            "text": "The required quantity is 5 units.",
        },
        "add": {
            "doc_type": "addendum",
            "addendum_number": 1,
            "text": "The required quantity is 100 units.",
        },
    }

    payload = {
        "changes": [{
            "field": "Quantity",
            "old_value": "5",
            "new_value": "10",
            "addendum_number": 1,
            "base_source_chunk_ids": ["base"],
            "source_chunk_ids": ["add"],
        }]
    }

    updated, changes = apply_addendum_changes(fields, payload, evidence)

    assert updated["Quantity"]["value"] == "5"
    assert changes == []
