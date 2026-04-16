from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.report import ReportDepth, ResearchReport


class ResearchStatus(str, Enum):
    QUEUED = "queued"
    SEARCHING = "searching"
    READING = "reading"
    EXTRACTING = "extracting"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    depth: ReportDepth = ReportDepth.STANDARD
    max_sources: int = Field(default=10, ge=1, le=50)
    include_papers: bool = True
    include_web: bool = True
    date_range: Optional[str] = Field(
        default=None,
        description='Optional natural range such as "past year" or "2020-2025".',
    )
    language: str = Field(default="en", min_length=2, max_length=10)
    required_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value: str) -> str:
        return " ".join(value.strip().split())

    @field_validator("include_web")
    @classmethod
    def at_least_one_source_type(cls, include_web: bool, info) -> bool:
        include_papers = info.data.get("include_papers", True)
        if not include_web and not include_papers:
            raise ValueError("At least one of include_web or include_papers must be true.")
        return include_web


class ResearchJob(BaseModel):
    id: str
    status: ResearchStatus = ResearchStatus.QUEUED
    request: ResearchRequest
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    progress: float = Field(default=0, ge=0, le=1)
    current_step: Optional[str] = None
    error: Optional[str] = None


class ResearchResponse(BaseModel):
    job_id: str
    status: ResearchStatus
    message: str


class ResearchJobResult(BaseModel):
    job: ResearchJob
    report: Optional[ResearchReport] = None