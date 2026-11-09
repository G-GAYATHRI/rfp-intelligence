from typing import Optional

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    bid_id: str
    file_name: str
    doc_type: str
    addendum_number: Optional[int] = None
    page_number: int
    text: str
    tables: list[str] = Field(default_factory=list)
    doc_date: Optional[str] = None