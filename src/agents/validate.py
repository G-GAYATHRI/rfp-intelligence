"""Evidence-aware critic: verifies citations and exact high-risk values."""
from __future__ import annotations
import re
STOP={"the","and","for","with","that","this","from","are","was","were","bid","not","any","all","per","its","into","over","than","then","the"}
DATE_RE=re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",re.I)
NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z])")
ID_RE=re.compile(r"\b(?=[A-Z0-9-_/]*[A-Z])(?=[A-Z0-9-_/]*\d)[A-Z0-9]+(?:[-_/][A-Z0-9]+)*\b",re.I)

def validation_counts(fields,rejected):
 r=set(rejected); return {"passed":sum(1 for n,f in fields.items() if n not in r and f.get("value") is not None),"failed":sum(1 for n in fields if n in r),"not_found":sum(1 for n,f in fields.items() if n not in r and f.get("value") is None)}

def _cited_hits(field,evidence):
 out=[]
 for src in field.get("sources") or []:
  cid=src.get("chunk_id")
  if not cid or cid not in evidence: continue
  hit=evidence[cid]
  if src.get("file") and src["file"]!=hit.get("file"): continue
  if src.get("page") is not None and str(src["page"])!=str(hit.get("page")): continue
  out.append(hit)
 return out

def _tokens(value): return [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*",str(value)) if len(t)>=3 and t.lower() not in STOP]

def _supported(value,text):
 toks=_tokens(value)
 if not toks:return True
 low=text.lower(); hits=sum(1 for t in toks if t.lower() in low)
 required=1 if len(toks)<4 else max(2,int(len(toks)*0.34))
 return hits>=required

def check_field(name,field,evidence):
 reasons=[]; value=field.get("value"); notes=str(field.get("notes") or "").strip()
 if notes.lower().startswith(("extraction error","rejected","retrieval error")): return [notes]
 if value is None:
  if not notes: reasons.append("Missing value has no reason")
  return reasons
 sources=field.get("sources") or []
 if not sources: return ["Value has no citation"]
 hits=_cited_hits(field,evidence)
 if not hits: return ["Citation does not resolve to a matching evidence chunk"]
 text="\n".join(h.get("text","") for h in hits)
 if name.lower()=="due date" and not DATE_RE.search(str(value)): reasons.append("Due Date has no recognizable date")
 if not _supported(value,text): reasons.append("Cited passage does not support the value")
 # Dates, quantities, percentages and identifiers are high-risk: any such token in the
 # proposed value must occur verbatim in its cited passage (normalizing commas/case).
 norm=lambda s: re.sub(r"[\s,]","",str(s)).lower()
 for label,pattern in (("date",DATE_RE),("numeric value",NUM_RE),("identifier",ID_RE)):
  vals=pattern.findall(str(value))
  for token in vals:
   if norm(token) not in norm(text): reasons.append(f"{label.title()} '{token}' not found verbatim in cited passage")
 if name=="Bid Summary":
  sentences=[s for s in re.split(r"[.!?]+",str(value)) if s.strip()]
  if not 3<=len(sentences)<=6: reasons.append("Bid Summary must be 3 to 6 sentences")
  if not _supported(value,text): reasons.append("Bid Summary is not sufficiently supported by its cited passages")
 return list(dict.fromkeys(reasons))

def validate_fields(fields,evidence):
 report={}; rejected=[]
 for name,field in fields.items():
  reasons=check_field(name,field,evidence)
  if reasons: report[name]=reasons; rejected.append(name)
 return report,rejected
