"""Structured bid record shared by every agent."""

from __future__ import annotations

FIELDS = [
    "Bid Number",
    "Title",
    "Due Date",
    "Bid Submission Type",
    "Term of Bid",
    "Pre Bid Meeting",
    "Installation",
    "Bid Bond Requirement",
    "Delivery Date",
    "Payment Terms",
    "Any Additional Documentation Required",
    "MFG for Registration",
    "Contract or Cooperative to use",
    "Model_no",
    "Part_no",
    "Product",
    "contact_info",
    "company_name",
    "Bid Summary",
    "Product Specification",
]

GROUPS = {
    "identity": [
        "Bid Number", "Title", "company_name", "contact_info", "Bid Submission Type",
    ],
    "schedule": [
        "Due Date", "Term of Bid", "Pre Bid Meeting", "Delivery Date",
    ],
    "commercial": [
        "Bid Bond Requirement", "Payment Terms",
        "Any Additional Documentation Required", "MFG for Registration",
        "Contract or Cooperative to use",
    ],
    "product": [
        "Model_no", "Part_no", "Product", "Installation", "Product Specification",
    ],
}

FIELD_QUERIES = {
    "Bid Number": "solicitation number RFP number bid number PORFP number",
    "Title": "title of the bid request for proposal",
    "Due Date": "submission deadline due date closing date time",
    "Bid Submission Type": "how to submit the bid electronic portal sealed paper email",
    "Term of Bid": "contract term duration renewal option years",
    "Pre Bid Meeting": "pre-bid meeting pre-proposal conference date location mandatory",
    "Installation": "installation deployment imaging white glove setup required",
    "Bid Bond Requirement": "bid bond bid security surety guarantee amount",
    "Delivery Date": "delivery date delivery window days after award",
    "Payment Terms": "payment terms net 30 invoice invoicing",
    "Any Additional Documentation Required": "affidavit certificate insurance W-9 forms required documents",
    "MFG for Registration": "manufacturer authorization registered reseller OEM",
    "Contract or Cooperative to use": "cooperative contract state contract vehicle",
    "Model_no": "model number models requested",
    "Part_no": "part number SKU manufacturer part",
    "Product": "product name quantity devices requested",
    "contact_info": "procurement contact buyer email phone",
    "company_name": "issuing agency organization school district department",
    "Product Specification": "processor CPU memory RAM storage display warranty specifications",
    "Bid Summary": "scope of work purpose of the solicitation summary",
}


def blank_field(note: str = "Not found in documents") -> dict:
    return {"value": None, "sources": [], "confidence": 0.0, "notes": note}


def empty_record(bid_id: str) -> dict:
    return {
        "bid_id": bid_id,
        "fields": {name: blank_field() for name in FIELDS},
        "addendum_changes": [],
        "validation": {"passed": 0, "failed": 0, "not_found": len(FIELDS)},
    }
