import logging
from pathlib import Path
from src.ingestion.parsers import parse_bid_folder
from .chunking import chunk_pages
log=logging.getLogger("indexer")

def index_bid_folder(folder: Path) -> dict:
    folder=Path(folder)
    if not folder.is_dir(): raise FileNotFoundError(f"Bid folder not found: {folder}")
    pages=parse_bid_folder(folder)
    if not pages: raise RuntimeError(f"No readable pages found in {folder}; existing index was not changed")
    chunks=chunk_pages(pages)
    if not chunks: raise RuntimeError(f"No searchable chunks produced from {folder}; existing index was not changed")
    from . import store
    n=store.replace_bid(folder.name,chunks)
    if n<=0: raise RuntimeError(f"Index store wrote zero chunks for {folder.name}; existing index was not changed")
    log.info("Indexed %s: %d pages -> %d chunks",folder.name,len(pages),n)
    return {"bid_id":folder.name,"pages":len(pages),"chunks":n}
