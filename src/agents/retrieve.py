
from __future__ import annotations

from src.agents.schema import FIELD_QUERIES


DUE_DATE_QUERIES = [
    "PROPOSAL DUE DATE and TIME",
    "proposal submission deadline bid response due date",
]

DOCUMENTATION_QUERIES = [
    "Form 1295 Certificate of Interested Parties submit with proposal",
    "M/WBE Compliance Guidelines Forms complete sign date submit",
    "M/WBE certificates Affirmative Action EEO Supplier Diversity Plan",
    "W-9 completed signed submit with proposal",
    "product literature pertinent documentation proposed products goods services",
    "credit application required to conduct business submit with offer",
    "Punch Out Process optional attachment",
    "insurance form within 10 days after award post-award",
]


def retrieve(
    bid_id: str,
    fields: list[str],
    feedback: dict | None = None,
    per_query: int = 4,
    limit: int = 10,
) -> list[dict]:
    from src.search.hybrid import search

    hits: dict[str, dict] = {}
    documentation_batches: list[list[dict]] = []
    feedback = feedback or {}

    documentation_field = "Any Additional Documentation Required"
    needs_documentation = documentation_field in fields

    for name in fields:
        query = FIELD_QUERIES.get(name, name)

        if feedback.get(name):
            query = f"{query} {name}"

        queries = (
            DUE_DATE_QUERIES + [query]
            if name == "Due Date"
            else [query]
        )

        # Run normal field-specific searches.
        for search_query in queries:
            top_k = max(per_query, 6) if name == "Due Date" else per_query

            found = search(
                search_query,
                top_k=top_k,
                mode="hybrid",
                bid_id=bid_id,
            )

            if name == documentation_field:
                documentation_batches.append(found)
            else:
                for hit in found:
                    hits.setdefault(hit["chunk_id"], hit)

        # Run targeted searches for each documentation category.
        if name == documentation_field:
            for search_query in DOCUMENTATION_QUERIES:
                found = search(
                    search_query,
                    top_k=per_query,
                    mode="hybrid",
                    bid_id=bid_id,
                )

                documentation_batches.append(found)

    # Preserve the existing addendum retrieval behavior.
    if any(
        name in {
            "Due Date",
            "Bid Bond Requirement",
            "Payment Terms",
            "Pre Bid Meeting",
        }
        for name in fields
    ):
        for hit in search(
            "addendum amendment change revised date",
            top_k=per_query,
            mode="hybrid",
            bid_id=bid_id,
            doc_type="addendum",
        ):
            hits.setdefault(hit["chunk_id"], hit)

    if needs_documentation:
        # Interleave results from different documentation queries.
        # This prevents the first categories from occupying all slots.
        doc_results = []
        seen_doc_ids = set()

        max_batch_size = max(
            (len(batch) for batch in documentation_batches),
            default=0,
        )

        for index in range(max_batch_size):
            for batch in documentation_batches:
                if index >= len(batch):
                    continue

                hit = batch[index]
                chunk_id = hit["chunk_id"]

                if chunk_id not in seen_doc_ids:
                    seen_doc_ids.add(chunk_id)
                    doc_results.append(hit)

        # Documentation requests get a local budget of 12 chunks
        # and a total group limit of 20.
        effective_limit = max(limit, 20)
        documentation_budget = min(
            effective_limit,
            max(per_query, 12),
            len(doc_results),
        )

        other_results = [
            hit
            for chunk_id, hit in hits.items()
            if chunk_id not in seen_doc_ids
        ]

        return (
            doc_results[:documentation_budget]
            + other_results[:max(0, effective_limit - documentation_budget)]
        )

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
