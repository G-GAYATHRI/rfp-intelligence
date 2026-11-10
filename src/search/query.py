SYNONYMS: dict[str, list[str]] = {
    "deadline": ["due date", "submission deadline", "closing date", "proposal due"],
    "due date": ["submission deadline", "closing date", "deadline", "new due date"],
    "closing": ["due date", "deadline", "submission"],
    "bid bond": ["bid security", "bid guarantee", "surety", "bond"],
    "bond": ["bid security", "surety", "performance bond"],
    "warranty": ["extended warranty", "warranty period", "years"],
    "affidavit": ["affidavits", "certification", "contract affidavit", "forms"],
    "pre-bid": ["pre-proposal meeting", "pre-bid conference", "mandatory attendance"],
    "pre bid": ["pre-proposal meeting", "pre-bid conference", "mandatory attendance"],
    "payment": ["payment terms", "net 30", "invoice", "invoicing"],
    "term": ["contract term", "length of contract", "renewal", "extensions"],
    "delivery": ["delivery date", "delivered within", "days after award"],
    "contact": ["buyer", "procurement officer", "email", "phone"],
    "submit": ["submission", "electronic", "portal", "sealed"],
    "installation": ["install", "deployment", "imaging", "setup"],
    "insurance": ["certificate of insurance", "liability coverage"],
    "model": ["model number", "part number", "sku"],
    "specification": ["specs", "processor", "ram", "memory", "storage", "display"],
    "addendum": ["amendment", "modification", "revised", "changes"],
}


def expand_query(query: str) -> str:
    q = query.lower()
    extra: list[str] = []
    for key, syns in SYNONYMS.items():
        if key in q:
            extra += [s for s in syns if s not in q and s not in extra]
    return query if not extra else f"{query} ({', '.join(extra)})"