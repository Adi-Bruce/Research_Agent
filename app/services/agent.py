import asyncio
import hashlib
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from app.schemas.report import ResearchReport
from app.schemas.research import (
    ResearchJob,
    ResearchJobResult,
    ResearchRequest,
    ResearchStatus,
)
from app.schemas.source import Source
from app.services.extraction.extractor import ExtractionError, extract_evidence
from app.services.processing.dedupe import dedupe_sources
from app.services.retrieval.fetch import FetchError, fetch_source_async
from app.services.retrieval.paper_search import PaperSearchError, search_papers_async
from app.services.retrieval.web_search import WebSearchError, search_web_async
from app.services.synthesis.synthesizer import SynthesisError, synthesize_report


class ResearchAgentError(RuntimeError):
    """Raised when the research pipeline cannot produce a report."""


class ResearchAgent:
    async def run(self, request: ResearchRequest) -> ResearchJobResult:
        job = self._new_job(request)

        try:
            job = self._update_job(
                job,
                status=ResearchStatus.SEARCHING,
                progress=0.1,
                current_step="Searching web and paper providers",
            )
            sources = await self._search_sources(request)
            sources = self._filter_sources(sources, request)
            if not sources:
                raise ResearchAgentError("No sources were found for the research topic.")

            job = self._update_job(
                job,
                status=ResearchStatus.READING,
                progress=0.35,
                current_step="Fetching source text",
            )
            fetched_sources = await self._fetch_sources(sources[: request.max_sources])
            if not fetched_sources:
                raise ResearchAgentError("No source text could be fetched.")

            job = self._update_job(
                job,
                status=ResearchStatus.EXTRACTING,
                progress=0.65,
                current_step="Extracting evidence",
            )
            evidence = self._extract_evidence(request.topic, fetched_sources)
            if not evidence:
                raise ResearchAgentError("No evidence could be extracted from fetched sources.")

            job = self._update_job(
                job,
                status=ResearchStatus.SYNTHESIZING,
                progress=0.85,
                current_step="Synthesizing final report",
            )
            report = synthesize_report(
                topic=request.topic,
                evidence=evidence,
                sources=fetched_sources,
                depth=request.depth,
            )

            job = self._update_job(
                job,
                status=ResearchStatus.COMPLETED,
                progress=1,
                current_step="Completed",
                completed_at=datetime.utcnow(),
            )
            return ResearchJobResult(job=job, report=report)

        except (ResearchAgentError, SynthesisError) as exc:
            job = self._update_job(
                job,
                status=ResearchStatus.FAILED,
                progress=job.progress,
                current_step="Failed",
                error=str(exc),
                completed_at=datetime.utcnow(),
            )
            return ResearchJobResult(job=job, report=None)

    def run_sync(self, request: ResearchRequest) -> ResearchJobResult:
        return asyncio.run(self.run(request))

    async def _search_sources(self, request: ResearchRequest) -> list[Source]:
        tasks = []
        if request.include_web:
            tasks.append(search_web_async(request.topic, request.max_sources))
        if request.include_papers:
            tasks.append(search_papers_async(request.topic, request.max_sources))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        sources: list[Source] = []
        errors: list[str] = []

        for result in results:
            if isinstance(result, (WebSearchError, PaperSearchError)):
                errors.append(str(result))
                continue
            if isinstance(result, Exception):
                errors.append(f"{type(result).__name__}: {result}")
                continue
            sources.extend(result)

        deduped = dedupe_sources(sources)
        if not deduped and errors:
            raise ResearchAgentError("Search failed: " + "; ".join(errors))
        return deduped

    async def _fetch_sources(self, sources: list[Source]) -> list[Source]:
        results = await asyncio.gather(
            *(fetch_source_async(source) for source in sources if source.url),
            return_exceptions=True,
        )

        fetched: list[Source] = []
        for result in results:
            if isinstance(result, FetchError):
                continue
            if isinstance(result, Exception):
                continue
            if result.full_text:
                fetched.append(result)
        return fetched

    def _extract_evidence(self, topic: str, sources: list[Source]):
        evidence = []
        for source in sources:
            try:
                evidence.append(extract_evidence(topic, source))
            except ExtractionError:
                continue
        return evidence

    def _filter_sources(
        self, sources: list[Source], request: ResearchRequest
    ) -> list[Source]:
        filtered = []
        required_domains = {domain.lower() for domain in request.required_domains}
        blocked_domains = {domain.lower() for domain in request.blocked_domains}

        for source in sources:
            domain = self._domain(source)
            if blocked_domains and domain in blocked_domains:
                continue
            if required_domains and domain not in required_domains:
                continue
            filtered.append(source)

        return filtered[: request.max_sources]

    def _domain(self, source: Source) -> str:
        if not source.url:
            return ""
        hostname = urlparse(str(source.url)).hostname or ""
        return hostname.removeprefix("www.").lower()

    def _new_job(self, request: ResearchRequest) -> ResearchJob:
        now = datetime.utcnow()
        return ResearchJob(
            id=_job_id(request.topic, now),
            status=ResearchStatus.QUEUED,
            request=request,
            created_at=now,
            updated_at=now,
            progress=0,
        )

    def _update_job(
        self,
        job: ResearchJob,
        *,
        status: ResearchStatus,
        progress: float,
        current_step: Optional[str],
        error: Optional[str] = None,
        completed_at: Optional[datetime] = None,
    ) -> ResearchJob:
        return job.model_copy(
            update={
                "status": status,
                "updated_at": datetime.utcnow(),
                "completed_at": completed_at,
                "progress": progress,
                "current_step": current_step,
                "error": error,
            }
        )


async def run_research(request: ResearchRequest) -> ResearchJobResult:
    return await ResearchAgent().run(request)


def run_research_sync(request: ResearchRequest) -> ResearchJobResult:
    return ResearchAgent().run_sync(request)


async def research_topic(
    topic: str,
    *,
    max_sources: int = 10,
    include_web: bool = True,
    include_papers: bool = True,
) -> ResearchReport:
    result = await run_research(
        ResearchRequest(
            topic=topic,
            max_sources=max_sources,
            include_web=include_web,
            include_papers=include_papers,
        )
    )
    if result.report is None:
        raise ResearchAgentError(result.job.error or "Research failed.")
    return result.report


def _job_id(topic: str, created_at: datetime) -> str:
    digest = hashlib.sha1(f"{topic}:{created_at.isoformat()}".encode("utf-8")).hexdigest()
    return f"J-{digest[:10]}"
