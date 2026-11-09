"""REST API for indexing, search, extraction, and questions."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.search.hybrid import search
from src.search.indexer import index_bid_folder

app = FastAPI(title="RFP Intelligence", version="1.0")


class IndexRequest(BaseModel):
    folder: str


class AskRequest(BaseModel):
    question: str
    bid_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
def index(body: IndexRequest):
    folder = Path(body.folder)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Bid folder not found: {folder}")
    try:
        return index_bid_folder(folder)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/search")
def search_endpoint(
    q: str,
    bid_id: str | None = None,
    doc_type: str | None = None,
    addendum: int | None = None,
    top_k: int = 5,
    mode: str = "hybrid",
    expand: bool = True,
):
    if mode not in {"vector", "hybrid", "hybrid_rerank"}:
        raise HTTPException(status_code=400, detail="mode must be vector, hybrid, or hybrid_rerank")
    try:
        results = search(q, top_k, mode, bid_id, doc_type, addendum, expand)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"query": q, "mode": mode, "results": results}


@app.post("/ask")
def ask(body: AskRequest):
    from src.agents.qa import answer_question
    return answer_question(body.question, body.bid_id, body.top_k)


@app.post("/extract")
def extract(body: IndexRequest):
    folder = Path(body.folder)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Bid folder not found: {folder}")
    from src.agents.orchestrator import run_extraction
    try:
        return run_extraction(folder)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
