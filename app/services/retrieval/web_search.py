import asyncio
import hashlib
import json
import os
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import ValidationError

from app.schemas.source import Source, SourceProvider, SourceType


BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_MAX_RESULTS = 10


class WebSearchProvider(str, Enum):
    DUCKDUCKGO = "duckduckgo"
    BRAVE = "brave"


class WebSearchError(RuntimeError):
    """Raised when the configured web search provider cannot return results."""


class WebSearchClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = BRAVE_SEARCH_URL,
        provider: Optional[WebSearchProvider | str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        self.endpoint = endpoint
        self.provider = _normalize_provider(
            provider or os.getenv("WEB_SEARCH_PROVIDER") or WebSearchProvider.DUCKDUCKGO
        )
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Source]:
        normalized_query = _normalize_query(query)
        result_count = _normalize_max_results(max_results)
        if self.provider == WebSearchProvider.BRAVE:
            payload = self._call_brave(normalized_query, result_count)
            return normalize_search_results(payload, result_count)
        return self._call_duckduckgo(normalized_query, result_count)

    async def search_async(
        self, query: str, max_results: int = DEFAULT_MAX_RESULTS
    ) -> list[Source]:
        return await asyncio.to_thread(self.search, query, max_results)

    def _call_brave(self, query: str, max_results: int) -> dict[str, Any]:
        if not self.api_key:
            raise WebSearchError(
                "BRAVE_SEARCH_API_KEY is required to call the web search provider."
            )

        params = urlencode(
            {
                "q": query,
                "count": max_results,
                "result_filter": "web",
                "safesearch": "moderate",
            }
        )
        request = Request(
            f"{self.endpoint}?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "X-Subscription-Token": self.api_key,
                "User-Agent": "research-agent/0.1",
            },
            method="GET",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise WebSearchError(
                f"Web search provider returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise WebSearchError(f"Web search provider request failed: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise WebSearchError("Web search provider returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise WebSearchError("Web search provider returned an unexpected payload.")
        return data

    def _call_duckduckgo(self, query: str, max_results: int) -> list[Source]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise WebSearchError(
                "DuckDuckGo search requires the ddgs package. Install it with "
                "`conda run -n thenv python -m pip install ddgs`."
            ) from exc

        try:
            with DDGS(timeout=self.timeout_seconds) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:
            raise WebSearchError(f"DuckDuckGo search failed: {exc}") from exc

        return normalize_duckduckgo_results(results, max_results)


def search_web(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Source]:
    return WebSearchClient().search(query=query, max_results=max_results)


async def search_web_async(
    query: str, max_results: int = DEFAULT_MAX_RESULTS
) -> list[Source]:
    return await WebSearchClient().search_async(query=query, max_results=max_results)


def normalize_search_results(
    payload: dict[str, Any], max_results: int = DEFAULT_MAX_RESULTS
) -> list[Source]:
    results = payload.get("web", {}).get("results", [])
    if not isinstance(results, list):
        return []

    sources: list[Source] = []
    seen_urls: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue

        source = _source_from_result(item)
        if source is None:
            continue

        url_key = str(source.url) if source.url else source.id
        if url_key in seen_urls:
            continue

        seen_urls.add(url_key)
        sources.append(source)
        if len(sources) >= max_results:
            break

    return sources


def normalize_duckduckgo_results(
    results: list[dict[str, Any]], max_results: int = DEFAULT_MAX_RESULTS
) -> list[Source]:
    sources: list[Source] = []
    seen_urls: set[str] = set()

    for item in results:
        if not isinstance(item, dict):
            continue

        source = _source_from_duckduckgo_result(item)
        if source is None:
            continue

        url_key = str(source.url) if source.url else source.id
        if url_key in seen_urls:
            continue

        seen_urls.add(url_key)
        sources.append(source)
        if len(sources) >= max_results:
            break

    return sources


def _source_from_result(result: dict[str, Any]) -> Optional[Source]:
    title = _clean_text(result.get("title"))
    url = _clean_text(result.get("url"))
    if not title or not url:
        return None

    published_at = _parse_datetime(result.get("age"))
    description = _clean_text(result.get("description"))

    try:
        return Source(
            id=_source_id(url),
            title=title,
            url=url,
            source_type=_infer_source_type(result),
            provider=SourceProvider.WEB_SEARCH,
            authors=[],
            published_at=published_at,
            accessed_at=datetime.utcnow(),
            summary=description,
            external_id=None,
            doi=None,
            credibility_score=None,
            relevance_score=None,
            token_count=None,
        )
    except ValidationError:
        return None


def _source_from_duckduckgo_result(result: dict[str, Any]) -> Optional[Source]:
    title = _clean_text(result.get("title"))
    url = _clean_text(result.get("href") or result.get("url"))
    if not title or not url:
        return None

    description = _clean_text(result.get("body") or result.get("description"))

    try:
        return Source(
            id=_source_id(url),
            title=title,
            url=url,
            source_type=_infer_source_type({"url": url}),
            provider=SourceProvider.WEB_SEARCH,
            authors=[],
            published_at=None,
            accessed_at=datetime.utcnow(),
            summary=description,
            external_id=None,
            doi=None,
            credibility_score=None,
            relevance_score=None,
            token_count=None,
        )
    except ValidationError:
        return None


def _infer_source_type(result: dict[str, Any]) -> SourceType:
    url = str(result.get("url", "")).lower()
    profile = result.get("profile")
    profile_name = ""
    if isinstance(profile, dict):
        profile_name = str(profile.get("name", "")).lower()

    if url.endswith(".pdf"):
        return SourceType.PDF
    if "news" in profile_name:
        return SourceType.NEWS
    if "blog" in url:
        return SourceType.BLOG
    return SourceType.WEB_PAGE


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None

    candidate = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def _source_id(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"S-{digest}"


def _normalize_query(query: str) -> str:
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise ValueError("query must not be empty")
    return normalized


def _normalize_max_results(max_results: int) -> int:
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    return min(max_results, 20)


def _normalize_provider(value: WebSearchProvider | str) -> WebSearchProvider:
    if isinstance(value, WebSearchProvider):
        return value

    normalized = value.strip().lower()
    if normalized in {"duckduckgo", "ddg", "ddgs"}:
        return WebSearchProvider.DUCKDUCKGO
    if normalized == "brave":
        return WebSearchProvider.BRAVE
    raise ValueError(f"Unsupported web search provider: {value}")


def _clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None
