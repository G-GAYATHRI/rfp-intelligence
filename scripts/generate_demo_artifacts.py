"""Generate shareable extraction and cited Q&A demo artifacts.

Run from the repository root:
    python -m scripts.generate_demo_artifacts
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import config
from src.agents.llm import _require_config
from src.agents.orchestrator import run_extraction
from src.agents.qa import answer_question

BID_IDS = ("Bid1", "Bid2")
SAMPLE_QUESTION_IDS = (
    "B1-01", "B1-03", "B1-05", "B1-07", "B1-12",
    "B2-01", "B2-02", "B2-05", "B2-09", "B2-12",
)


def main() -> None:
    # Fail early instead of silently writing empty extraction records.
    _require_config()

    generated_dir = config.ROOT / "examples" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    extraction_summary = {}
    for bid_id in BID_IDS:
        folder = config.DATA_DIR / bid_id
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing {bid_id} source folder: {folder}"
            )

        record = run_extraction(folder)
        runtime_json = config.OUTPUT_DIR / f"{bid_id}.json"
        if not runtime_json.is_file():
            raise RuntimeError(
                f"Extraction did not create the expected file: {runtime_json}"
            )

        published_json = generated_dir / f"{bid_id}.json"
        shutil.copy2(runtime_json, published_json)

        extraction_summary[bid_id] = {
            "path": str(published_json.relative_to(config.ROOT)),
            "validation": record.get("validation", {}),
            "error_count": len(record.get("errors") or []),
            "ingestion_ok": record.get("ingestion_ok", False),
        }

    questions_path = config.ROOT / "eval" / "qa_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in questions}
    missing = [qid for qid in SAMPLE_QUESTION_IDS if qid not in by_id]
    if missing:
        raise ValueError(f"Missing sample question IDs: {missing}")

    qa_items = []
    for question_id in SAMPLE_QUESTION_IDS:
        item = by_id[question_id]
        result = answer_question(
            item["query"],
            bid_id=item["bid_id"],
            top_k=10,
        )
        sources = result.get("sources") or []
        has_traceable_citation = (
            result.get("answer") != "Not found in documents"
            and bool(sources)
            and all(
                source.get("file")
                and source.get("page") is not None
                and source.get("chunk_id")
                for source in sources
            )
        )
        qa_items.append({
            "id": question_id,
            "bid_id": item["bid_id"],
            "question": item["query"],
            "answer": result.get("answer", "Not found in documents"),
            "confidence": result.get("confidence", 0.0),
            "notes": result.get("notes", ""),
            "citation_status": "cited" if has_traceable_citation else "missing_or_unresolved",
            "sources": sources,
        })

    cited_count = sum(
        item["citation_status"] == "cited" for item in qa_items
    )
    qa_log = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts.generate_demo_artifacts",
        "retrieval_mode": "hybrid_rerank",
        "question_count": len(qa_items),
        "cited_answer_count": cited_count,
        "items": qa_items,
    }
    qa_path = generated_dir / "sample_qa.json"
    qa_path.write_text(
        json.dumps(qa_log, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "generated_at": qa_log["generated_at"],
        "extractions": extraction_summary,
        "sample_qa": {
            "path": str(qa_path.relative_to(config.ROOT)),
            "question_count": len(qa_items),
            "cited_answer_count": cited_count,
        },
    }
    (generated_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    if cited_count != len(SAMPLE_QUESTION_IDS):
        print(
            "WARNING: Not all sample answers have a resolvable citation. "
            "Review sample_qa.json before presenting it as a fully cited demo."
        )


if __name__ == "__main__":
    main()
