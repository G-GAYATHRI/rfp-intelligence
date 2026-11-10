from __future__ import annotations

def _source(hit: dict) -> dict:
    return {"file": hit.get("file"), "page": hit.get("page"), "chunk_id": hit.get("chunk_id")}

def _is_addendum(hit: dict) -> bool:
    return str(hit.get("doc_type", "")).lower() == "addendum" and hit.get("addendum_number") is not None

def _clean_value(value):
    if isinstance(value,str) and value.strip().lower() in {"", "null", "none", "not stated", "n/a"}: return None
    return value

def apply_addendum_changes(fields: dict, payload: dict, evidence: dict) -> tuple[dict,list]:
    updated={name:dict(field) for name,field in fields.items()}
    changes=[]
    items=list(payload.get("changes") or [])
    def sort_key(item):
        try: return int(item.get("addendum_number"))
        except (TypeError,ValueError): return 10**9
    for item in sorted(items,key=sort_key):
        name=item.get("field")
        if name not in updated: continue
        valid=[]
        for cid in item.get("source_chunk_ids") or []:
            hit=evidence.get(cid)
            if not hit or not _is_addendum(hit): continue
            try: actual_no=int(hit.get("addendum_number"))
            except (TypeError,ValueError): continue
            declared=item.get("addendum_number")
            if declared is not None:
                try:
                    if int(declared)!=actual_no: continue
                except (TypeError,ValueError): continue
            valid.append((actual_no,_source(hit)))
        if not valid: continue
        numbers={n for n,_ in valid}
        if len(numbers)!=1: continue
        add_no=next(iter(numbers))
        new_value=_clean_value(item.get("new_value")); old_value=updated[name].get("value")
        if new_value is None or new_value==old_value: continue
        try: confidence=float(item.get("confidence") if item.get("confidence") is not None else 0.7)
        except (TypeError,ValueError): confidence=0.7
        sources=[s for _,s in valid]
        note=item.get("notes") or f"Updated by Addendum {add_no}; previous value: {old_value}"
        updated[name]={"value":new_value,"sources":sources,"confidence":max(0.0,min(1.0,confidence)),"notes":note}
        changes.append({"field":name,"old_value":old_value,"new_value":new_value,"addendum_number":add_no,"sources":[{"file":s["file"],"page":s["page"]} for s in sources],"notes":note})
    return updated,changes
