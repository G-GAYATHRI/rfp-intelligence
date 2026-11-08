import logging, sys
from pathlib import Path
from src.ingestion.parsers import parse_bid_folder

logging.basicConfig(level=logging.INFO)
pages = parse_bid_folder(Path(sys.argv[1]))
for p in pages:
    print(f"[{p.file_name}] type={p.doc_type} add#={p.addendum_number} "
          f"page={p.page_number} chars={len(p.text)} tables={len(p.tables)}")