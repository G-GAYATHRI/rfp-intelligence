import logging
from pathlib import Path

from src.ingestion.parsers import parse_bid_folder
from .chunking import chunk_pages
from . import store

log = logging.getLogger("indexer")


def index_bid_folder(folder: Path) -> dict:
    pages = parse_bid_folder(folder)
    chunks = chunk_pages(pages)
    n = store.replace_bid(folder.name, chunks)
    log.info("Indexed %s: %d pages -> %d chunks", folder.name, len(pages), n)
    return {"bid_id": folder.name, "pages": len(pages), "chunks": n}