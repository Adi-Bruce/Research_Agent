from fastapi import FastAPI

from app.api.routes import router as research_router


app = FastAPI(
    title="Research Agent API",
    description="Searches the web and papers, extracts evidence, and synthesizes cited reports.",
    version="0.1.0",
)

app.include_router(research_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
