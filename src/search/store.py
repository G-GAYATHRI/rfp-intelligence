import logging
from typing import Optional

import chromadb

import config
from .chunking import Chunk
from .embeddings import embed_documents, embed_query

log = logging.getLogger("store")

_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
_col = _client.get_or_create_collection(
    "rfp_chunks", metadata={"hnsw:space": "cosine"})


VERSION = 0   # bumps on every index change; invalidates the BM25 cache
_UPSERT_BATCH = 64


def _meta(chunk: Chunk) -> dict:
    return {
        "bid_id": chunk.bid_id,
        "file_name": chunk.file_name,
        "doc_type": chunk.doc_type,
        "addendum_number": chunk.addendum_number or 0,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "is_table": chunk.is_table,
        "doc_date": chunk.doc_date or "",
    }


def replace_bid(bid_id: str, chunks: list[Chunk]) -> int:
    """Upsert the new chunks first, then drop stale ids.

    A failure while embedding or writing leaves the previous bid index in place.
    An empty chunk list does not delete the existing bid.
    """
    global VERSION
    unique: dict[str, Chunk] = {}
    for chunk in chunks:
        unique[chunk.chunk_id] = chunk
    chunks = list(unique.values())
    if not chunks:
        log.warning("No chunks for %s; existing index left unchanged", bid_id)
        return 0

    existing = _col.get(where={"bid_id": bid_id}, include=[])["ids"]
    embedded = embed_documents([c.embed_text for c in chunks])
    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start:start + _UPSERT_BATCH]
        vectors = embedded[start:start + _UPSERT_BATCH]
        _col.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors,
            metadatas=[_meta(c) for c in batch],
        )
    stale = [cid for cid in existing if cid not in unique]
    if stale:
        _col.delete(ids=stale)
    VERSION += 1
    return len(chunks)


def list_bids() -> list[str]:
    metas = _col.get(include=["metadatas"]).get("metadatas") or []
    return sorted({m["bid_id"] for m in metas if m and m.get("bid_id")})


def _matching_ids(where: Optional[dict]) -> list[str]:
    return _col.get(where=where, include=[]).get("ids") or []


def vector_search(query: str, top_k: int = 5,
                  where: Optional[dict] = None) -> list[dict]:
    if top_k < 1 or _col.count() == 0:
        return []
    available = len(_matching_ids(where))
    if available == 0:
        return []
    res = _col.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, available),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = res.get("ids") or [[]]
    if not ids or not ids[0]:
        return []
    return [{
        "chunk_id": cid, "text": doc, "score": 1 - dist, **meta,
    } for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0],
        res["metadatas"][0], res["distances"][0])]

def get_chunks(where: Optional[dict] = None) -> list[dict]:
    res = _col.get(where=where, include=["documents", "metadatas"])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    return [{"chunk_id": i, "text": d or "", **(m or {})}
            for i, d, m in zip(ids, docs, metas)]

