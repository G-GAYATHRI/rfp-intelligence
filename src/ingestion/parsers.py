import logging
from pathlib import Path
import fitz  
import pdfplumber
from bs4 import BeautifulSoup

from .models import PageContent
from .classify import detect_doc_type, detect_addendum_number
from .cleaning import fix_text, remove_repeated_lines

log = logging.getLogger("ingestion")

def _table_to_md(rows) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r]
    if not rows:
        return ""
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)

def parse_pdf(path: Path, bid_id: str) -> list[PageContent]:
    doc_type = detect_doc_type(path.name)
    add_no = detect_addendum_number(path.name)
    raw_pages, tables_per_page = [], []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                raw_pages.append(page.extract_text() or "")
                tbls = [_table_to_md(t) for t in page.extract_tables()]
                tables_per_page.append([t for t in tbls if t])
    except Exception as e:
        log.error("Unreadable PDF %s: %s", path.name, e)
        return []

    cleaned = remove_repeated_lines(raw_pages)
    pages = []
    for i, (txt, tbls) in enumerate(zip(cleaned, tables_per_page), start=1):
        txt = fix_text(txt)
        if not txt and not tbls:
            log.warning("Empty/scanned page %s p%d (OCR needed)", path.name, i)
            continue
        pages.append(PageContent(
            bid_id=bid_id, file_name=path.name, doc_type=doc_type,
            addendum_number=add_no, page_number=i, text=txt, tables=tbls))
    return pages

def parse_html(path: Path, bid_id: str) -> list[PageContent]:
    try:
        soup = BeautifulSoup(path.read_text(errors="ignore"), "lxml")
    except Exception as e:
        log.error("Unreadable HTML %s: %s", path.name, e)
        return []
    for t in soup(["script", "style", "nav", "footer", "noscript"]):
        t.decompose()
    tables = []
    for tbl in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                for tr in tbl.find_all("tr")]
        md = _table_to_md(rows)
        if md:
            tables.append(md)
        tbl.decompose()
    text = fix_text(soup.get_text("\n"))
    return [PageContent(bid_id=bid_id, file_name=path.name, doc_type="bid_page",
                        page_number=1, text=text, tables=tables)]

def parse_bid_folder(folder: Path) -> list[PageContent]:
    bid_id = folder.name
    out = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() == ".pdf":
            out += parse_pdf(f, bid_id)
        elif f.suffix.lower() in (".html", ".htm"):
            out += parse_html(f, bid_id)
        else:
            log.warning("Skipping unsupported file %s", f.name)
    return out