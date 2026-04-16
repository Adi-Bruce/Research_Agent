from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class EvidenceType(str, Enum):
    CLAIM = "claim"
    STATISTIC = "statistic"
    QUOTE = "quote"
    DEFINITION = "definition"
    FINDING = "finding"
    COUNTERPOINT = "counterpoint"


class EvidenceStrength(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Citation(BaseModel):
    source_id: str = Field(..., description="Matches Source.id.")
    title: Optional[str] = None
    url: Optional[HttpUrl] = None
    authors: list[str] = Field(default_factory=list)
    published_year: Optional[int] = Field(default=None, ge=1000, le=3000)
    locator: Optional[str] = Field(
        default=None,
        description="Page number, section, paragraph, timestamp, or quoted span location.",
    )


class Evidence(BaseModel):
    id: str = Field(..., description="Stable internal evidence ID, for example E1.")
    source_id: str = Field(..., description="Matches Source.id.")
    text: str = Field(..., min_length=1, max_length=5000)
    gist: Optional[str] = Field(
        default=None,
        max_length=1200,
        description="Short topic-focused summary of what this source says.",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Most relevant points extracted from the source.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Limits, uncertainty, or reasons to treat the source carefully.",
    )
    evidence_type: EvidenceType = EvidenceType.CLAIM
    strength: EvidenceStrength = EvidenceStrength.MEDIUM
    relevance_score: Optional[float] = Field(default=None, ge=0, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    citation: Optional[Citation] = None
    tags: list[str] = Field(default_factory=list)


class EvidenceCreate(BaseModel):
    source_id: str
    text: str = Field(..., min_length=1, max_length=5000)
    evidence_type: EvidenceType = EvidenceType.CLAIM
    locator: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class EvidenceCluster(BaseModel):
    id: str = Field(..., description="Stable internal cluster ID.")
    theme: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(..., min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
