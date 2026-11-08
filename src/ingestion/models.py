from pydantic import BaseModel
from typing import Optional

class PageContent(BaseModel):
    bid_id: str
    file_name: str
    doc_type: str                      
    addendum_number: Optional[int] = None
    page_number: int
    text: str
    tables: list[str] = []             
    doc_date: Optional[str] = None