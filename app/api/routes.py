from fastapi import APIRouter, HTTPException, status

from app.schemas.research import ResearchJobResult, ResearchRequest
from app.services.agent import ResearchAgent


router = APIRouter(prefix="/research", tags=["research"])


@router.post(
    "",
    response_model=ResearchJobResult,
    status_code=status.HTTP_200_OK,
    summary="Run a research job",
)
async def run_research(request: ResearchRequest) -> ResearchJobResult:
    result = await ResearchAgent().run(request)
    if result.report is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.job.error or "Research job failed.",
        )
    return result


@router.get("/health", summary="Research API health check")
async def research_health() -> dict[str, str]:
    return {"status": "ok"}
