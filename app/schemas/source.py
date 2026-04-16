from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class SourceType(str, Enum):
    WEB_PAGE = "web_page"
    NEWS = "news"
    BLOG = "blog"
    PAPER = "paper"
    PDF = "pdf"
    BOOK = "book"
    DATASET = "dataset"
    OTHER = "other"


class SourceProvider(str, Enum):
    WEB_SEARCH = "web_search"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    MANUAL = "manual"


class SourceBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    url: Optional[HttpUrl] = None
    source_type: SourceType = SourceType.WEB_PAGE
    provider: Optional[SourceProvider] = None
    authors: list[str] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    summary: Optional[str] = Field(default=None, max_length=2000)
    full_text: Optional[str] = Field(
        default=None,
        description="Cleaned fetched text for the source.",
    )


class SourceCreate(SourceBase):
    external_id: Optional[str] = Field(
        default=None,
        description="Provider-specific ID such as arXiv ID, DOI, or Semantic Scholar paper ID.",
    )
    doi: Optional[str] = None
    raw_text: Optional[str] = None


class Source(SourceBase):
    id: str = Field(..., description="Stable internal source ID, for example S1.")
    external_id: Optional[str] = None
    doi: Optional[str] = None
    credibility_score: Optional[float] = Field(default=None, ge=0, le=1)
    relevance_score: Optional[float] = Field(default=None, ge=0, le=1)
    token_count: Optional[int] = Field(default=None, ge=0)


class SearchResult(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    url: HttpUrl
    snippet: Optional[str] = Field(default=None, max_length=1500)
    source_type: SourceType = SourceType.WEB_PAGE
    provider: SourceProvider = SourceProvider.WEB_SEARCH
    published_at: Optional[datetime] = None
    relevance_score: Optional[float] = Field(default=None, ge=0, le=1)
