import asyncio
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import ValidationError

from app.schemas.source import Source, SourceProvider, SourceType


SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL = "https://export.arxiv.org/api/query"
OPENALEX_URL = "https://api.openalex.org/works"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_RESULTS = 10
DEFAULT_PAPER_PROVIDERS = (
    "openalex",
    "arxiv",
)


class PaperProvider(str, Enum):
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ARXIV = "arxiv"
    OPENALEX = "openalex"


class PaperSearchError(RuntimeError):
    """Raised when paper search providers cannot return results."""


class PaperSearchClient:
    def __init__(
        self,
        semantic_scholar_api_key: Optional[str] = None,
        openalex_email: Optional[str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.semantic_scholar_api_key = semantic_scholar_api_key or os.getenv(
            "SEMANTIC_SCHOLAR_API_KEY"
        )
        self.openalex_email = openalex_email or os.getenv("OPENALEX_EMAIL")
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        providers: Optional[list[PaperProvider]] = None,
    ) -> list[Source]:
        normalized_query = _normalize_query(query)
        result_count = _normalize_max_results(max_results)
        active_providers = providers or _default_paper_providers()

        sources: list[Source] = []
        errors: list[str] = []
        for provider in active_providers:
            try:
                sources.extend(self._search_provider(provider, normalized_query, result_count))
            except PaperSearchError as exc:
                errors.append(f"{provider.value}: {exc}")

        deduped = _dedupe_sources(sources)
        if deduped:
            return deduped[:result_count]

        if errors:
            raise PaperSearchError("All paper providers failed: " + "; ".join(errors))
        return []

    async def search_async(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        providers: Optional[list[PaperProvider]] = None,
    ) -> list[Source]:
        return await asyncio.to_thread(self.search, query, max_results, providers)

    def _search_provider(
        self, provider: PaperProvider, query: str, max_results: int
    ) -> list[Source]:
        if provider == PaperProvider.SEMANTIC_SCHOLAR:
            payload = self._call_semantic_scholar(query, max_results)
            return normalize_semantic_scholar_results(payload, max_results)
        if provider == PaperProvider.ARXIV:
            payload = self._call_arxiv(query, max_results)
            return normalize_arxiv_results(payload, max_results)
        if provider == PaperProvider.OPENALEX:
            payload = self._call_openalex(query, max_results)
            return normalize_openalex_results(payload, max_results)
        raise PaperSearchError(f"Unsupported paper provider: {provider}")

    def _call_semantic_scholar(self, query: str, max_results: int) -> dict[str, Any]:
        params = urlencode(
            {
                "query": query,
                "limit": max_results,
                "fields": (
                    "title,url,abstract,authors,year,publicationDate,"
                    "externalIds,citationCount"
                ),
            }
        )
        headers = {"Accept": "application/json", "User-Agent": "research-agent/0.1"}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key

        return _request_json(f"{SEMANTIC_SCHOLAR_URL}?{params}", headers, self.timeout_seconds)

    def _call_arxiv(self, query: str, max_results: int) -> str:
        params = urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            quote_via=quote,
        )
        return _request_text(
            f"{ARXIV_URL}?{params}",
            {"Accept": "application/atom+xml", "User-Agent": "research-agent/0.1"},
            self.timeout_seconds,
        )

    def _call_openalex(self, query: str, max_results: int) -> dict[str, Any]:
        params = urlencode(
            _without_none(
                {
                "search": query,
                "per-page": max_results,
                "select": (
                    "id,display_name,doi,authorships,publication_year,"
                    "publication_date,abstract_inverted_index,cited_by_count,"
                    "primary_location"
                ),
                "mailto": self.openalex_email,
                }
            )
        )
        return _request_json(
            f"{OPENALEX_URL}?{params}",
            {"Accept": "application/json", "User-Agent": "research-agent/0.1"},
            self.timeout_seconds,
        )


def search_papers(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    providers: Optional[list[PaperProvider]] = None,
) -> list[Source]:
    return PaperSearchClient().search(
        query=query, max_results=max_results, providers=providers
    )


async def search_papers_async(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    providers: Optional[list[PaperProvider]] = None,
) -> list[Source]:
    return await PaperSearchClient().search_async(
        query=query, max_results=max_results, providers=providers
    )


def normalize_semantic_scholar_results(
    payload: dict[str, Any], max_results: int = DEFAULT_MAX_RESULTS
) -> list[Source]:
    papers = payload.get("data", [])
    if not isinstance(papers, list):
        return []

    sources: list[Source] = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        source = _source_from_semantic_scholar(paper)
        if source:
            sources.append(source)
        if len(sources) >= max_results:
            break
    return sources


def normalize_arxiv_results(
    payload: str, max_results: int = DEFAULT_MAX_RESULTS
) -> list[Source]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise PaperSearchError("arXiv returned invalid XML.") from exc

    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    sources: list[Source] = []
    for entry in root.findall("atom:entry", namespace):
        source = _source_from_arxiv_entry(entry, namespace)
        if source:
            sources.append(source)
        if len(sources) >= max_results:
            break
    return sources


def normalize_openalex_results(
    payload: dict[str, Any], max_results: int = DEFAULT_MAX_RESULTS
) -> list[Source]:
    papers = payload.get("results", [])
    if not isinstance(papers, list):
        return []

    sources: list[Source] = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        source = _source_from_openalex(paper)
        if source:
            sources.append(source)
        if len(sources) >= max_results:
            break
    return sources


def _source_from_semantic_scholar(paper: dict[str, Any]) -> Optional[Source]:
    title = _clean_text(paper.get("title"))
    if not title:
        return None

    external_ids = paper.get("externalIds")
    if not isinstance(external_ids, dict):
        external_ids = {}

    external_id = _clean_text(paper.get("paperId"))
    doi = _clean_text(external_ids.get("DOI"))
    url = _clean_text(paper.get("url")) or _doi_url(doi)
    authors = _semantic_scholar_authors(paper.get("authors"))
    published_at = _date_from_parts(paper.get("publicationDate"), paper.get("year"))

    return _build_source(
        title=title,
        url=url,
        provider=SourceProvider.SEMANTIC_SCHOLAR,
        external_id=external_id,
        doi=doi,
        authors=authors,
        published_at=published_at,
        summary=_clean_text(paper.get("abstract")),
        relevance_score=_citation_score(paper.get("citationCount")),
    )


def _source_from_arxiv_entry(
    entry: ET.Element, namespace: dict[str, str]
) -> Optional[Source]:
    title = _clean_text(_element_text(entry.find("atom:title", namespace)))
    external_url = _clean_text(_element_text(entry.find("atom:id", namespace)))
    if not title or not external_url:
        return None

    arxiv_id = external_url.rstrip("/").split("/")[-1]
    authors = [
        name
        for author in entry.findall("atom:author", namespace)
        if (name := _clean_text(_element_text(author.find("atom:name", namespace))))
    ]
    doi = _clean_text(_element_text(entry.find("arxiv:doi", namespace)))
    published_at = _parse_datetime(_element_text(entry.find("atom:published", namespace)))

    return _build_source(
        title=title,
        url=external_url,
        provider=SourceProvider.ARXIV,
        external_id=arxiv_id,
        doi=doi,
        authors=authors,
        published_at=published_at,
        summary=_clean_text(_element_text(entry.find("atom:summary", namespace))),
        relevance_score=None,
    )


def _source_from_openalex(paper: dict[str, Any]) -> Optional[Source]:
    title = _clean_text(paper.get("display_name"))
    if not title:
        return None

    doi = _normalize_doi(paper.get("doi"))
    openalex_id = _clean_text(paper.get("id"))
    url = _openalex_url(paper) or _doi_url(doi) or openalex_id
    authors = _openalex_authors(paper.get("authorships"))
    published_at = _date_from_parts(
        paper.get("publication_date"), paper.get("publication_year")
    )

    return _build_source(
        title=title,
        url=url,
        provider=SourceProvider.OPENALEX,
        external_id=openalex_id,
        doi=doi,
        authors=authors,
        published_at=published_at,
        summary=_abstract_from_openalex(paper.get("abstract_inverted_index")),
        relevance_score=_citation_score(paper.get("cited_by_count")),
    )


def _build_source(
    *,
    title: str,
    url: Optional[str],
    provider: SourceProvider,
    external_id: Optional[str],
    doi: Optional[str],
    authors: list[str],
    published_at: Optional[datetime],
    summary: Optional[str],
    relevance_score: Optional[float],
) -> Optional[Source]:
    stable_key = doi or url or external_id or title
    try:
        return Source(
            id=_source_id(stable_key),
            title=title,
            url=url,
            source_type=SourceType.PAPER,
            provider=provider,
            authors=authors,
            published_at=published_at,
            accessed_at=datetime.utcnow(),
            summary=summary,
            external_id=external_id,
            doi=doi,
            credibility_score=None,
            relevance_score=relevance_score,
            token_count=None,
        )
    except ValidationError:
        return None


def _request_json(url: str, headers: dict[str, str], timeout_seconds: int) -> dict[str, Any]:
    body = _request_text(url, headers, timeout_seconds)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise PaperSearchError("Paper provider returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise PaperSearchError("Paper provider returned an unexpected payload.")
    return data


def _request_text(url: str, headers: dict[str, str], timeout_seconds: int) -> str:
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PaperSearchError(
            f"Paper provider returned HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise PaperSearchError(f"Paper provider request failed: {exc}") from exc


def _dedupe_sources(sources: list[Source]) -> list[Source]:
    deduped: list[Source] = []
    seen: set[str] = set()
    for source in sources:
        key = (source.doi or str(source.url or "") or source.title).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _semantic_scholar_authors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for author in value:
        if isinstance(author, dict) and (name := _clean_text(author.get("name"))):
            authors.append(name)
    return authors


def _openalex_authors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for authorship in value:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if isinstance(author, dict) and (name := _clean_text(author.get("display_name"))):
            authors.append(name)
    return authors


def _abstract_from_openalex(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None

    words_by_position: dict[int, str] = {}
    for word, positions in value.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words_by_position[position] = word

    if not words_by_position:
        return None
    return " ".join(words_by_position[index] for index in sorted(words_by_position))


def _openalex_url(paper: dict[str, Any]) -> Optional[str]:
    location = paper.get("primary_location")
    if not isinstance(location, dict):
        return None

    landing_page_url = _clean_text(location.get("landing_page_url"))
    pdf_url = _clean_text(location.get("pdf_url"))
    return landing_page_url or pdf_url


def _date_from_parts(date_value: Any, year_value: Any) -> Optional[datetime]:
    parsed = _parse_datetime(date_value)
    if parsed:
        return parsed

    try:
        year = int(year_value)
    except (TypeError, ValueError):
        return None
    if 1000 <= year <= 3000:
        return datetime(year, 1, 1)
    return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None

    candidate = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y"):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def _normalize_query(query: str) -> str:
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise ValueError("query must not be empty")
    return normalized


def _normalize_max_results(max_results: int) -> int:
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    return min(max_results, 25)


def _default_paper_providers() -> list[PaperProvider]:
    raw = os.getenv("PAPER_SEARCH_PROVIDERS")
    values = raw.split(",") if raw else list(DEFAULT_PAPER_PROVIDERS)

    providers: list[PaperProvider] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized:
            continue
        try:
            providers.append(PaperProvider(normalized))
        except ValueError as exc:
            raise PaperSearchError(f"Unsupported paper provider: {value}") from exc
    return providers or [PaperProvider.OPENALEX, PaperProvider.ARXIV]


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _normalize_doi(value: Any) -> Optional[str]:
    doi = _clean_text(value)
    if not doi:
        return None
    return doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")


def _doi_url(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    return f"https://doi.org/{doi}"


def _source_id(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"P-{digest}"


def _element_text(element: Optional[ET.Element]) -> Optional[str]:
    if element is None:
        return None
    return element.text


def _citation_score(value: Any) -> Optional[float]:
    try:
        citation_count = int(value)
    except (TypeError, ValueError):
        return None
    return min(citation_count / 500, 1.0)


def _clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None
