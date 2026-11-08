import chromadb
from typing import Optional

import config
from .chunking import Chunk
from .embeddings import embed_documents, embed_query

_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
_col = _client.get_or_create_collection(
    "rfp_chunks", metadata={"hnsw:space": "cosine"})


def replace_bid(bid_id: str, chunks: list[Chunk]) -> int:
    _col.delete(where={"bid_id": bid_id})
    if not chunks:
        return 0
    _col.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed_documents([c.embed_text for c in chunks]),
        metadatas=[{
            "bid_id": c.bid_id, "file_name": c.file_name,
            "doc_type": c.doc_type,
            "addendum_number": c.addendum_number or 0,   
            "page_number": c.page_number, "section": c.section,
            "is_table": c.is_table,
        } for c in chunks],
    )
    return len(chunks)


def list_bids() -> list[str]:
    metas = _col.get(include=["metadatas"])["metadatas"]
    return sorted({m["bid_id"] for m in metas})


def vector_search(query: str, top_k: int = 5,
                  where: Optional[dict] = None) -> list[dict]:
    res = _col.query(query_embeddings=[embed_query(query)], n_results=top_k,
                     where=where, include=["documents", "metadatas", "distances"])
    return [{
        "chunk_id": cid, "text": doc, "score": 1 - dist, **meta,
    } for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0],
        res["metadatas"][0], res["distances"][0])]