import hashlib
import re
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.schemas.evidence import Evidence
from app.schemas.source import Source


TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}
TITLE_SIMILARITY_THRESHOLD = 0.94


def dedupe_sources(sources: list[Source]) -> list[Source]:
    deduped: list[Source] = []
    seen_keys: set[str] = set()

    for source in sources:
        keys = source_dedupe_keys(source)
        if keys & seen_keys:
            deduped = _merge_matching_source(deduped, source, keys)
            seen_keys.update(keys)
            continue

        title_match_index = _find_similar_title_index(deduped, source)
        if title_match_index is not None:
            deduped[title_match_index] = _prefer_source(deduped[title_match_index], source)
            seen_keys.update(keys)
            continue

        deduped.append(source)
        seen_keys.update(keys)

    return deduped


def dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    deduped: list[Evidence] = []
    seen: set[str] = set()

    for item in evidence:
        key = evidence_dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def source_dedupe_keys(source: Source) -> set[str]:
    keys: set[str] = set()

    if doi := normalize_doi(source.doi):
        keys.add(f"doi:{doi}")

    if arxiv_id := _arxiv_id(source.external_id):
        keys.add(f"arxiv:{arxiv_id}")

    if source.url:
        url = canonical_url(str(source.url))
        keys.add(f"url:{url}")
        if arxiv_id := _arxiv_id(url):
            keys.add(f"arxiv:{arxiv_id}")
        if doi := _doi_from_url(url):
            keys.add(f"doi:{doi}")

    if source.title:
        keys.add(f"title:{normalize_title(source.title)}")

    if source.full_text:
        keys.add(f"text:{text_fingerprint(source.full_text)}")

    return {key for key in keys if key and not key.endswith(":")}


def evidence_dedupe_key(evidence: Evidence) -> str:
    text = evidence.gist or evidence.text
    normalized = normalize_title(text)
    return f"{evidence.source_id}:{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:16]}"


def normalize_doi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    doi = value.strip().lower()
    doi = doi.removeprefix("https://doi.org/")
    doi = doi.removeprefix("http://doi.org/")
    doi = doi.removeprefix("doi:")
    doi = doi.strip().rstrip(".")
    return doi or None


def canonical_url(value: str) -> str:
    parsed = urlparse(value.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"

    query_items = []
    for key, val in parse_qsl(parsed.query, keep_blank_values=False):
        key_lower = key.lower()
        if key_lower in TRACKING_QUERY_KEYS:
            continue
        if any(key_lower.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        query_items.append((key_lower, val))

    query = urlencode(sorted(query_items))
    return urlunparse((scheme, netloc, path, "", query, ""))


def normalize_title(value: str) -> str:
    text = value.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_fingerprint(value: str) -> str:
    normalized = normalize_title(value[:5_000])
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _merge_matching_source(
    sources: list[Source], candidate: Source, keys: set[str]
) -> list[Source]:
    for index, existing in enumerate(sources):
        if source_dedupe_keys(existing) & keys:
            sources[index] = _prefer_source(existing, candidate)
            return sources
    return sources


def _prefer_source(existing: Source, candidate: Source) -> Source:
    existing_score = _source_quality_score(existing)
    candidate_score = _source_quality_score(candidate)
    primary = candidate if candidate_score > existing_score else existing
    secondary = existing if primary is candidate else candidate

    return primary.model_copy(
        update={
            "summary": primary.summary or secondary.summary,
            "full_text": primary.full_text or secondary.full_text,
            "authors": primary.authors or secondary.authors,
            "published_at": primary.published_at or secondary.published_at,
            "doi": primary.doi or secondary.doi,
            "external_id": primary.external_id or secondary.external_id,
            "credibility_score": (
                primary.credibility_score
                if primary.credibility_score is not None
                else secondary.credibility_score
            ),
            "relevance_score": (
                primary.relevance_score
                if primary.relevance_score is not None
                else secondary.relevance_score
            ),
            "token_count": primary.token_count or secondary.token_count,
        }
    )


def _source_quality_score(source: Source) -> float:
    score = 0.0
    if source.doi:
        score += 3
    if source.full_text:
        score += 3
    if source.summary:
        score += 1
    if source.authors:
        score += 1
    if source.published_at:
        score += 1
    if source.relevance_score is not None:
        score += source.relevance_score
    if source.credibility_score is not None:
        score += source.credibility_score
    return score


def _find_similar_title_index(
    sources: list[Source], candidate: Source
) -> Optional[int]:
    candidate_title = normalize_title(candidate.title)
    if len(candidate_title) < 20:
        return None

    for index, source in enumerate(sources):
        existing_title = normalize_title(source.title)
        if len(existing_title) < 20:
            continue
        similarity = SequenceMatcher(None, existing_title, candidate_title).ratio()
        if similarity >= TITLE_SIMILARITY_THRESHOLD:
            return index
    return None


def _doi_from_url(value: str) -> Optional[str]:
    parsed = urlparse(value)
    if parsed.netloc == "doi.org":
        return normalize_doi(parsed.path.lstrip("/"))
    return None


def _arxiv_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    match = re.search(
        r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)?([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None
