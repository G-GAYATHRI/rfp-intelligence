import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main():
    ap = argparse.ArgumentParser("rfp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse")
    p.add_argument("folder")

    p = sub.add_parser("index")
    p.add_argument("folder")

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--bid", default=None)
    p.add_argument("--doc_type", default=None)
    p.add_argument("--addendum", type=int, default=None)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--mode", default="hybrid_rerank",
                   choices=["vector", "hybrid", "hybrid_rerank"])
    p.add_argument("--no_expand", action="store_true")

    args = ap.parse_args()

    if args.cmd == "parse":
        from src.ingestion.parsers import parse_bid_folder
        for pg in parse_bid_folder(Path(args.folder)):
            print(f"[{pg.file_name}] {pg.doc_type} add#={pg.addendum_number} "
                  f"p{pg.page_number} chars={len(pg.text)} tables={len(pg.tables)}")

    elif args.cmd == "index":
        from src.search.indexer import index_bid_folder
        print(json.dumps(index_bid_folder(Path(args.folder)), indent=2))

    elif args.cmd == "search":
        from src.search.hybrid import search
        results = search(args.query, args.top_k, args.mode, args.bid,
                         args.doc_type, args.addendum, not args.no_expand)
        if not results:
            print("No results.")
        for r in results:
            print(f"\n{r['score']:.3f} | {r['file']} p{r['page']} "
                  f"[{r['doc_type']}] | {r['section']}\n{r['text'][:300]}")


if __name__ == "__main__":
    main()