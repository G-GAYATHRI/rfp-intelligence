import re
from typing import Optional

def detect_doc_type(file_name: str) -> str:
    n = file_name.lower()
    if n.endswith((".html", ".htm")):
        return "bid_page"
    if "addendum" in n:
        return "addendum"
    if "affidavit" in n:
        return "affidavit"
    if "spec" in n:
        return "specs"
    return "rfp"

def detect_addendum_number(file_name: str) -> Optional[int]:
    m = re.search(r"addendum\s*[#_\-]?\s*(\d+)", file_name, re.I)
    return int(m.group(1)) if m else None