import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from .classify import detect_addendum_number, detect_doc_type
from .cleaning import find_doc_date, fix_text, remove_repeated_lines
from .models import PageContent

log = logging.getLogger("ingestion")

SUPPORTED = {".pdf", ".html", ".htm"}

def _table_to_md(rows) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r]
    if not rows:
        return ""
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)

def _table_line_keys(tables: list[str]) -> set[str]:
    keys = set()
    for md in tables:
        for row in md.splitlines():
            if not row.startswith("|"):
                continue
            cells = [re.sub(r"\s+", " ", c).strip() for c in row.strip("|").split("|")]
            if not cells or set("".join(cells)) <= set("-: "):
                continue
            joined = re.sub(r"\s+", " ", " ".join(c for c in cells if c)).strip().lower()
            if len(joined) >= 12:
                keys.add(joined)
    return keys


def _drop_duplicated_table_lines(text: str, tables: list[str]) -> str:
    """Remove page lines that repeat an extracted table row."""
    keys = _table_line_keys(tables)
    if not keys:
        return text
    kept = []
    for line in text.splitlines():
        norm = re.sub(r"\s+", " ", line).strip().lower()
        if norm and norm in keys:
            continue
        kept.append(line)
    return "\n".join(kept)


def parse_pdf(path: Path, bid_id: str, file_name: str | None = None) -> list[PageContent]:
    """Fast page-preserving PDF extraction using PyMuPDF and native table detection."""
    file_name = file_name or path.name
    doc_type = detect_doc_type(file_name)
    add_no = detect_addendum_number(file_name)
    raw_pages, tables_per_page = [], []
    try:
        import fitz  # PyMuPDF
        with fitz.open(path) as pdf:
            for page in pdf:
                raw_pages.append(page.get_text("text", sort=True) or "")
                page_tables = []
                try:
                    finder = page.find_tables()
                    for table in finder.tables:
                        md = _table_to_md(table.extract() or [])
                        if md:
                            page_tables.append(md)
                except Exception as table_error:
                    log.warning("Table extraction skipped for %s p%d: %s", file_name, page.number + 1, table_error)
                tables_per_page.append(page_tables)
    except Exception as exc:
        log.error("Unreadable PDF %s: %s", file_name, exc)
        return []

    # Some vendors use generic filenames; identify addenda from the document heading too.
    if doc_type != "addendum":
        heading = "\n".join(raw_pages[:2])
        match = re.search(r"addendum\s*(?:no\.?\s*)?[#_\-]?\s*(\d+)", heading, re.I)
        if match:
            doc_type = "addendum"
            add_no = int(match.group(1))
    cleaned = remove_repeated_lines(raw_pages)
    doc_date = find_doc_date(cleaned[0] if cleaned else "")
    pages = []
    for i, (txt, tbls) in enumerate(zip(cleaned, tables_per_page), start=1):
        txt = _drop_duplicated_table_lines(fix_text(txt), tbls)
        if not doc_date:
            doc_date = find_doc_date(txt)
        if not txt and not tbls:
            log.warning("Empty/scanned page %s p%d (OCR needed)", file_name, i)
            continue
        pages.append(PageContent(
            bid_id=bid_id, file_name=file_name, doc_type=doc_type,
            addendum_number=add_no, page_number=i, text=txt, tables=tbls,
            doc_date=doc_date))
    return pages

COOKIE_RE = re.compile(r"cookie|onetrust|consent|gdpr", re.I)

def parse_html(path: Path, bid_id: str, file_name: str | None = None) -> list[PageContent]:
    file_name = file_name or path.name
    try:
        soup = BeautifulSoup(path.read_bytes(), "lxml")
    except Exception as e:
        log.error("Unreadable HTML %s: %s", file_name, e)
        return []
    for t in soup(["script", "style", "noscript", "svg", "iframe"]):
        t.decompose()
    for el in soup.find_all(True):
        if getattr(el, "decomposed", False):
            continue
        ident = (el.get("id") or "") + " " + " ".join(el.get("class") or [])
        # only small cookie/consent boxes; never a big wrapper holding real content
        if COOKIE_RE.search(ident) and len(el.get_text(strip=True)) < 3000:
            el.decompose()
    tables = []
    for tbl in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                for tr in tbl.find_all("tr")]
        md = _table_to_md(rows)
        if md:
            tables.append(md)
        tbl.decompose()
    text = _drop_duplicated_table_lines(fix_text(soup.get_text("\n")), tables)
    return [PageContent(
        bid_id=bid_id, file_name=file_name, doc_type="bid_page",
        page_number=1, text=text, tables=tables, doc_date=find_doc_date(text))]


def _rel_name(folder: Path, path: Path) -> str:
    rel = path.relative_to(folder).as_posix()
    return path.name if "/" not in rel else rel


def parse_bid_folder(folder: Path) -> list[PageContent]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Bid folder not found: {folder}")
    bid_id = folder.name
    out = []
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )
    for f in files:
        name = _rel_name(folder, f)
        suffix = f.suffix.lower()
        if suffix == ".pdf":
            parsed = parse_pdf(f, bid_id, name)
        elif suffix in (".html", ".htm"):
            parsed = parse_html(f, bid_id, name)
        else:
            log.warning("Skipping unsupported file %s", name)
            continue
        if not parsed:
            raise RuntimeError(
                f"Supported source file could not be parsed: {name}. "
                "The bid was not indexed to avoid silently using an incomplete document set. "
                "If this is a scanned PDF, OCR is required."
            )
        out.extend(parsed)
    if not out:
        raise RuntimeError(f"No readable supported documents found in {folder}")
    return out