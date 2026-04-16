import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TypeVar

from app.schemas.evidence import Evidence
from app.schemas.source import Source, SourceType


T = TypeVar("T", Source, Evidence)

MAX_TEXT_CHARS = 20_000

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "between",
    "but",
    "can",
    "could",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "may",
    "more",
    "not",
    "our",
    "such",
    "than",
    "that",
    "the",
    "their",
    "this",
    "through",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
}


@dataclass(frozen=True)
class RankedItem:
    item: Source | Evidence
    score: float


def rank_sources(
    topic: str,
    sources: list[Source],
    *,
    limit: int | None = None,
) -> list[Source]:
    ranked = _rank_items(topic, sources, _source_text, _source_boost)
    results = [
        item.model_copy(update={"relevance_score": score})
        for item, score in ranked
    ]
    return results[:limit] if limit is not None else results


def rank_evidence(
    topic: str,
    evidence: list[Evidence],
    *,
    limit: int | None = None,
) -> list[Evidence]:
    ranked = _rank_items(topic, evidence, _evidence_text, _evidence_boost)
    results = [
        item.model_copy(update={"relevance_score": score})
        for item, score in ranked
    ]
    return results[:limit] if limit is not None else results


def cosine_similarity(query: str, document: str) -> float:
    query_terms = _term_counts(query)
    document_terms = _term_counts(document)
    if not query_terms or not document_terms:
        return 0.0
    return _cosine(query_terms, document_terms)


def tfidf_cosine_scores(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []

    tokenized_docs = [_tokens(document) for document in documents]
    query_tokens = _tokens(query)
    if not query_tokens:
        return [0.0 for _document in documents]

    document_frequency = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    document_count = len(documents)
    query_vector = _tfidf_vector(query_tokens, document_frequency, document_count)
    scores = []
    for tokens in tokenized_docs:
        document_vector = _tfidf_vector(tokens, document_frequency, document_count)
        scores.append(round(_cosine(query_vector, document_vector), 4))
    return scores


def _rank_items(
    topic: str,
    items: list[T],
    text_builder,
    boost_builder,
) -> list[tuple[T, float]]:
    documents = [text_builder(item)[:MAX_TEXT_CHARS] for item in items]
    similarity_scores = tfidf_cosine_scores(topic, documents)

    ranked: list[tuple[T, float]] = []
    for item, similarity in zip(items, similarity_scores):
        score = min(similarity + boost_builder(item), 1.0)
        ranked.append((item, round(score, 4)))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def _source_text(source: Source) -> str:
    parts = [
        source.title,
        source.summary or "",
        " ".join(source.authors),
        source.full_text or "",
    ]
    return " ".join(parts)


def _evidence_text(evidence: Evidence) -> str:
    parts = [
        evidence.gist or "",
        " ".join(evidence.key_points),
        evidence.text,
        " ".join(evidence.tags),
    ]
    return " ".join(parts)


def _source_boost(source: Source) -> float:
    boost = 0.0
    if source.source_type in {SourceType.PAPER, SourceType.PDF}:
        boost += 0.04
    if source.doi:
        boost += 0.03
    if source.summary:
        boost += 0.02
    if source.full_text:
        boost += 0.03
    if source.credibility_score is not None:
        boost += min(source.credibility_score, 1) * 0.04
    return boost


def _evidence_boost(evidence: Evidence) -> float:
    boost = 0.0
    if evidence.confidence is not None:
        boost += min(evidence.confidence, 1) * 0.08
    if evidence.key_points:
        boost += 0.03
    if evidence.citation:
        boost += 0.02
    return boost


def _tfidf_vector(
    tokens: list[str],
    document_frequency: Counter[str],
    document_count: int,
) -> dict[str, float]:
    term_counts = Counter(tokens)
    vector: dict[str, float] = {}

    for term, count in term_counts.items():
        term_frequency = count / max(len(tokens), 1)
        inverse_document_frequency = math.log(
            (1 + document_count) / (1 + document_frequency[term])
        ) + 1
        vector[term] = term_frequency * inverse_document_frequency

    return vector


def _term_counts(text: str) -> Counter[str]:
    return Counter(_tokens(text))


def _cosine(left: Counter[str] | dict[str, float], right: Counter[str] | dict[str, float]) -> float:
    common_terms = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in common_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if token not in STOPWORDS
    ]
