import hashlib
import re
from datetime import datetime

from app.schemas.evidence import Citation, Evidence, EvidenceStrength, EvidenceType
from app.schemas.source import Source, SourceType


MAX_ANALYSIS_CHARS = 40_000
MAX_GIST_CHARS = 900
MAX_EVIDENCE_TEXT_CHARS = 5_000
MAX_KEY_POINTS = 5


class ExtractionError(ValueError):
    """Raised when evidence cannot be extracted from a source."""


def extract_evidence(topic: str, source: Source) -> Evidence:
    return EvidenceExtractor().extract(topic=topic, source=source)


class EvidenceExtractor:
    def extract(self, topic: str, source: Source) -> Evidence:
        normalized_topic = _normalize_topic(topic)
        full_text = _normalize_text(source.full_text or "")
        if not full_text:
            raise ExtractionError(f"Source {source.id} has no full_text to extract.")

        analysis_text = full_text[:MAX_ANALYSIS_CHARS]
        sentences = _split_sentences(analysis_text)
        if not sentences:
            raise ExtractionError(f"Source {source.id} has no extractable sentences.")

        ranked_sentences = _rank_sentences(sentences, normalized_topic)
        selected = [sentence for sentence, _score in ranked_sentences[:MAX_KEY_POINTS]]
        if not selected:
            selected = sentences[: min(MAX_KEY_POINTS, len(sentences))]

        gist = _build_gist(selected, source)
        key_points = _build_key_points(selected)
        caveats = _build_caveats(source, full_text, ranked_sentences)
        relevance_score = _relevance_score(ranked_sentences)
        confidence = _source_confidence(source, full_text, relevance_score, caveats)
        strength = _strength_from_confidence(confidence)
        citation = _citation_from_source(source)
        text = _format_evidence_text(gist, key_points, caveats)

        return Evidence(
            id=_evidence_id(normalized_topic, source.id),
            source_id=source.id,
            text=text,
            gist=gist,
            key_points=key_points,
            caveats=caveats,
            evidence_type=EvidenceType.FINDING,
            strength=strength,
            relevance_score=relevance_score,
            confidence=confidence,
            citation=citation,
            tags=_topic_tags(normalized_topic),
        )


def _build_gist(sentences: list[str], source: Source) -> str:
    gist = " ".join(sentences[:2])
    if source.summary:
        summary = _normalize_text(source.summary)
        if summary and len(summary) > len(gist):
            gist = summary
    return _truncate(gist, MAX_GIST_CHARS)


def _build_key_points(sentences: list[str]) -> list[str]:
    points: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        point = _truncate(sentence, 350)
        key = point.lower()
        if key in seen:
            continue
        seen.add(key)
        points.append(point)
    return points


def _build_caveats(
    source: Source,
    full_text: str,
    ranked_sentences: list[tuple[str, float]],
) -> list[str]:
    caveats: list[str] = []

    if len(full_text) < 1_000:
        caveats.append("The fetched text is short, so the source may be incomplete.")
    if source.source_type not in {SourceType.PAPER, SourceType.PDF}:
        caveats.append("This is not marked as an academic paper.")
    if not source.published_at:
        caveats.append("No publication date was available.")
    if not source.authors:
        caveats.append("No author metadata was available.")
    if ranked_sentences and ranked_sentences[0][1] == 0:
        caveats.append("The source text has weak direct keyword overlap with the topic.")
    if _contains_uncertainty_language(full_text):
        caveats.append("The source uses uncertainty or limitation language.")

    return caveats or ["No major caveats were detected from the available metadata and text."]


def _format_evidence_text(gist: str, key_points: list[str], caveats: list[str]) -> str:
    lines = [f"Gist: {gist}", "", "Key points:"]
    lines.extend(f"- {point}" for point in key_points)
    lines.extend(["", "Caveats:"])
    lines.extend(f"- {caveat}" for caveat in caveats)
    return _truncate("\n".join(lines), MAX_EVIDENCE_TEXT_CHARS)


def _citation_from_source(source: Source) -> Citation:
    published_year = None
    if isinstance(source.published_at, datetime):
        published_year = source.published_at.year

    return Citation(
        source_id=source.id,
        title=source.title,
        url=source.url,
        authors=source.authors,
        published_year=published_year,
        locator="full_text",
    )


def _rank_sentences(sentences: list[str], topic: str) -> list[tuple[str, float]]:
    topic_terms = set(_keywords(topic))
    ranked: list[tuple[str, float]] = []

    for index, sentence in enumerate(sentences):
        sentence_terms = set(_keywords(sentence))
        overlap = len(topic_terms & sentence_terms)
        density = overlap / max(len(topic_terms), 1)
        length_bonus = 0.15 if 80 <= len(sentence) <= 350 else 0
        position_bonus = max(0, 0.1 - index * 0.005)
        score = density + length_bonus + position_bonus
        ranked.append((sentence, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _relevance_score(ranked_sentences: list[tuple[str, float]]) -> float:
    if not ranked_sentences:
        return 0
    top_scores = [score for _sentence, score in ranked_sentences[:MAX_KEY_POINTS]]
    return round(min(sum(top_scores) / len(top_scores), 1), 3)


def _source_confidence(
    source: Source,
    full_text: str,
    relevance_score: float,
    caveats: list[str],
) -> float:
    confidence = 0.35
    confidence += relevance_score * 0.3

    if source.source_type in {SourceType.PAPER, SourceType.PDF}:
        confidence += 0.15
    if source.authors:
        confidence += 0.08
    if source.published_at:
        confidence += 0.06
    if source.doi:
        confidence += 0.06
    if len(full_text) >= 3_000:
        confidence += 0.08

    confidence -= min(len([c for c in caveats if "No major caveats" not in c]) * 0.04, 0.2)
    return round(max(0, min(confidence, 1)), 3)


def _strength_from_confidence(confidence: float) -> EvidenceStrength:
    if confidence >= 0.75:
        return EvidenceStrength.HIGH
    if confidence >= 0.45:
        return EvidenceStrength.MEDIUM
    return EvidenceStrength.LOW


def _split_sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+", text)
    sentences: list[str] = []
    for candidate in candidates:
        sentence = _normalize_text(candidate)
        if 30 <= len(sentence) <= 700:
            sentences.append(sentence)
    return sentences


def _keywords(text: str) -> list[str]:
    stopwords = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "because",
        "between",
        "from",
        "has",
        "have",
        "into",
        "its",
        "more",
        "not",
        "that",
        "the",
        "their",
        "this",
        "through",
        "using",
        "was",
        "were",
        "with",
    }
    return [
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if word not in stopwords
    ]


def _topic_tags(topic: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for word in _keywords(topic):
        if word in seen:
            continue
        seen.add(word)
        tags.append(word)
    return tags[:8]


def _contains_uncertainty_language(text: str) -> bool:
    return bool(
        re.search(
            r"\b(may|might|could|limited|limitation|uncertain|uncertainty|"
            r"further research|small sample|not clear)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _normalize_topic(topic: str) -> str:
    normalized = _normalize_text(topic)
    if len(normalized) < 3:
        raise ExtractionError("topic must be at least 3 characters.")
    return normalized


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _evidence_id(topic: str, source_id: str) -> str:
    digest = hashlib.sha1(f"{topic}:{source_id}".encode("utf-8")).hexdigest()[:10]
    return f"E-{digest}"
