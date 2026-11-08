import re
from collections import Counter

def fix_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     
    text = re.sub(r"(?<![\.\:\n])\n(?!\n)", " ", text)  
    text = re.sub(r"[ \t]+", " ", text)
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