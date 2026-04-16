from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.evidence import Citation, Evidence, EvidenceCluster
from app.schemas.source import Source


class ReportDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ReportSection(BaseModel):
    heading: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class KeyFinding(BaseModel):
    finding: str = Field(..., min_length=1, max_length=1000)
    explanation: Optional[str] = Field(default=None, max_length=3000)
    confidence: float = Field(..., ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)


class ResearchReport(BaseModel):
    id: str
    topic: str = Field(..., min_length=1, max_length=500)
    title: str = Field(..., min_length=1, max_length=300)
    executive_summary: str = Field(..., min_length=1, max_length=5000)
    depth: ReportDepth = ReportDepth.STANDARD
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    key_findings: list[KeyFinding] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    evidence_clusters: list[EvidenceCluster] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class ReportPreview(BaseModel):
    id: str
    topic: str
    title: str
    executive_summary: str
    generated_at: datetime
    source_count: int = Field(..., ge=0)
    citation_count: int = Field(..., ge=0)
