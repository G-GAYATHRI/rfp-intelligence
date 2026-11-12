import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.search.store import get_chunks


INPUT_PATH = ROOT / "examples" / "generated" / "sample_qa.json"
OUTPUT_PATH = ROOT / "examples" / "generated" / "citation_validation.json"


def validate_citations() -> dict:
    # Step 1: Load generated Q&A
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        qa_data = json.load(file)

    # Step 2: Retrieve all indexed chunks
    indexed_chunks = get_chunks()

    # Create lookup for fast chunk ID verification
    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in indexed_chunks
    }

    results = []
    total_citations = 0
    valid_citations = 0
    invalid_citations = 0

    # Step 3: Validate every citation
    for item in qa_data.get("items", []):
        question_id = item.get("id")
        bid_id = item.get("bid_id")

        for source in item.get("sources", []):
            total_citations += 1
            issues = []

            chunk_id = source.get("chunk_id")
            indexed = chunk_lookup.get(chunk_id)

            if not chunk_id:
                issues.append("Missing chunk ID")

            elif indexed is None:
                issues.append("Chunk ID not found in ChromaDB")

            else:
                # Verify bid ID
                if indexed.get("bid_id") != bid_id:
                    issues.append("Bid ID mismatch")

                # Verify filename
                if indexed.get("file_name") != source.get("file"):
                    issues.append("Filename mismatch")

                # Verify page number
                if indexed.get("page_number") != source.get("page"):
                    issues.append("Page number mismatch")

            is_valid = len(issues) == 0

            if is_valid:
                valid_citations += 1
            else:
                invalid_citations += 1

            results.append({
                "question_id": question_id,
                "bid_id": bid_id,
                "chunk_id": chunk_id,
                "file": source.get("file"),
                "page": source.get("page"),
                "valid": is_valid,
                "issues": issues,
            })

    # Step 4: Generate validation summary
    report = {
        "input_file": str(INPUT_PATH.relative_to(ROOT)),
        "indexed_chunk_count": len(indexed_chunks),
        "question_count": len(qa_data.get("items", [])),
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "overall_status": (
            "PASS"
            if total_citations > 0 and invalid_citations == 0
            else "FAIL"
        ),
        "results": results,
    }

    # Step 5: Save report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    report = validate_citations()

    print("\n========== CITATION VALIDATION ==========")
    print(f"Indexed chunks : {report['indexed_chunk_count']}")
    print(f"Questions      : {report['question_count']}")
    print(f"Total citations: {report['total_citations']}")
    print(f"Valid citations: {report['valid_citations']}")
    print(f"Invalid citations: {report['invalid_citations']}")
    print(f"Overall status : {report['overall_status']}")

    for result in report["results"]:
        if not result["valid"]:
            print(f"\nFAILED: {result['question_id']}")
            print(f"Chunk ID: {result['chunk_id']}")
            for issue in result["issues"]:
                print(f"  - {issue}")

    print(f"\nReport saved to: {OUTPUT_PATH}")
