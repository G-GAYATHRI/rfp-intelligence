import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_dataset_has_15_plus_grounded_pairs_for_both_bids():
    pairs = json.loads((ROOT / "eval/qa_questions.json").read_text(encoding="utf-8"))
    chunks = {row["chunk_id"]: row for row in (
        json.loads(line) for line in (ROOT / "eval/corpus_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    )}
    assert len(pairs) >= 15
    assert sum(p["bid_id"] == "Bid1" for p in pairs) >= 5
    assert sum(p["bid_id"] == "Bid2" for p in pairs) >= 5
    for pair in pairs:
        assert pair["expected_chunk_id"] in chunks
        passage = chunks[pair["expected_chunk_id"]]
        assert passage["bid_id"] == pair["bid_id"]
        assert all(snippet.lower() in passage["text"].lower() for snippet in pair["expect_text_contains"])
