import re
from collections import Counter

DATE_RE = re.compile(
    r"(?:issue date|date issued|dated|addendum date|bid end date|due date)"
    r"\s*[:\-]?\s*("
    r"[A-Z][a-z]+ \d{1,2},?\s+\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r")",
    re.I,
)


def find_doc_date(text: str) -> str | None:
    match = DATE_RE.search((text or "")[:4000])
    return match.group(1).strip() if match else None

def fix_text(text: str) -> str:
    """Normalize whitespace while keeping headings and paragraphs on their own lines.

    Wrapped lines are joined only when the next line continues a sentence in
    lowercase. A line that starts with a capital, a number, or a heading stays
    intact so later chunking can still see section boundaries.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<![.!?:;\n])\n(?=[a-z])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_repeated_lines(pages: list[str], min_ratio: float = 0.6) -> list[str]:
    if len(pages) < 3:
        return pages
    counts = Counter()
    for p in pages:
        lines = {l.strip() for l in p.splitlines() if l.strip()}
        counts.update(lines)
    threshold = max(3, int(len(pages) * min_ratio))
    repeated = {l for l, c in counts.items() if c >= threshold and len(l) < 120}
    return [
        "\n".join(l for l in p.splitlines() if l.strip() not in repeated)
        for p in pages
    ]