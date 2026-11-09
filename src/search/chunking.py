import hashlib
import re
from typing import Optional
from pydantic import BaseModel

from src.ingestion.models import PageContent

MAX_CHARS = 1000
OVERLAP = 150
TABLE_MAX_CHARS = 1500

HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[\.\)]?\s+[A-Z].{2,90}"   # 1. Scope / 2.3 Bid Bond
    r"|[A-Z][A-Z0-9 \-&/,\.]{4,90}):?$"           # ALL CAPS HEADING
)


class Chunk(BaseModel):
    chunk_id: str
    bid_id: str
    file_name: str
    doc_type: str
    addendum_number: Optional[int] = None
    page_number: int
    section: str = ""
    is_table: bool = False
    doc_date: Optional[str] = None
    text: str

    @property
    def embed_text(self) -> str:
        label = self.doc_type + (f" {self.addendum_number}" if self.addendum_number else "")
        return f"{label} | {self.section}\n{self.text}"


def _make_id(page: PageContent, idx: int, text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{page.bid_id}:{page.file_name}:p{page.page_number}:{idx}:{h}"


def _tail(text: str, n: int) -> str:
    """Last ~n chars, starting at a word boundary (used as overlap)."""
    if len(text) <= n:
        return text
    t = text[-n:]
    return t[t.find(" ") + 1:] if " " in t else t


def _split_table(md: str) -> list[str]:
    lines = md.splitlines()
    if len(md) <= TABLE_MAX_CHARS or len(lines) <= 3:
        return [md]
    header = "\n".join(lines[:2])
    pieces, cur = [], header
    for row in lines[2:]:
        if len(cur) + len(row) > TABLE_MAX_CHARS and cur != header:
            pieces.append(cur)
            cur = header
        cur += "\n" + row
    if cur != header:
        pieces.append(cur)
    return pieces


def chunk_pages(pages: list[PageContent]) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_section = ""
    last_file = None

    for page in pages:
        if page.file_name != last_file:      
            current_section, last_file = "", page.file_name
        idx = 0

        def emit(text: str, section: str, is_table: bool = False):
            nonlocal idx
            text = text.strip()
            if len(text) < 20:
                return
            chunks.append(Chunk(
                chunk_id=_make_id(page, idx, text),
                bid_id=page.bid_id, file_name=page.file_name,
                doc_type=page.doc_type, addendum_number=page.addendum_number,
                page_number=page.page_number, section=section,
                is_table=is_table, doc_date=page.doc_date, text=text))
            idx += 1

        buf = ""
        for line in page.text.splitlines():
            line = line.strip()
            if not line:
                continue
            if HEADING_RE.match(line) and len(line) < 90:
                emit(buf, current_section)
                buf = ""
                current_section = line
                continue
            if len(buf) + len(line) + 1 > MAX_CHARS and buf:
                emit(buf, current_section)
                buf = _tail(buf, OVERLAP)
            buf = f"{buf} {line}".strip()
        emit(buf, current_section)

        
        for tbl in page.tables:
            for piece in _split_table(tbl):
                emit(piece, current_section or "Table", is_table=True)

    return chunks