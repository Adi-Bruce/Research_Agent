import hashlib
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

from app.schemas.evidence import Citation, Evidence, EvidenceCluster
from app.schemas.report import KeyFinding, ReportDepth, ReportSection, ResearchReport
from app.schemas.source import Source


MAX_FINDINGS = 6
MAX_CLUSTER_COUNT = 6
MAX_SECTION_EVIDENCE = 8
MAX_SUMMARY_CHARS = 4_500


class SynthesisError(ValueError):
    """Raised when a report cannot be synthesized from the available evidence."""


def synthesize_report(
    topic: str,
    evidence: list[Evidence],
    sources: Optional[list[Source]] = None,
    depth: ReportDepth = ReportDepth.STANDARD,
) -> ResearchReport:
    return ReportSynthesizer().synthesize(
        topic=topic,
        evidence=evidence,
        sources=sources or [],
        depth=depth,
    )


class ReportSynthesizer:
    def synthesize(
        self,
        topic: str,
        evidence: list[Evidence],
        sources: Optional[list[Source]] = None,
        depth: ReportDepth = ReportDepth.STANDARD,
    ) -> ResearchReport:
        normalized_topic = _normalize_topic(topic)
        ranked_evidence = _rank_evidence(evidence)
        if not ranked_evidence:
            raise SynthesisError("At least one evidence object is required.")

        source_list = _dedupe_sources(sources or [])
        clusters = _cluster_evidence(ranked_evidence)
        key_findings = _build_key_findings(ranked_evidence)
        sections = _build_sections(normalized_topic, ranked_evidence, clusters)
        limitations = _build_limitations(ranked_evidence, source_list)
        follow_up_questions = _build_follow_up_questions(normalized_topic, clusters)
        executive_summary = _build_executive_summary(
            normalized_topic,
            key_findings,
            limitations,
            len(source_list) or len(_unique_source_ids(ranked_evidence)),
        )

        return ResearchReport(
            id=_report_id(normalized_topic, ranked_evidence),
            topic=normalized_topic,
            title=f"Research Report: {normalized_topic}",
            executive_summary=executive_summary,
            depth=depth,
            generated_at=datetime.utcnow(),
            key_findings=key_findings,
            sections=sections,
            evidence_clusters=clusters,
            evidence=ranked_evidence,
            sources=source_list,
            limitations=limitations,
            follow_up_questions=follow_up_questions,
        )


def _build_key_findings(evidence: list[Evidence]) -> list[KeyFinding]:
    findings: list[KeyFinding] = []
    seen: set[str] = set()

    for item in evidence:
        finding_text = item.gist or _first_sentence(item.text)
        finding_text = _truncate(_normalize_text(finding_text), 1_000)
        if not finding_text:
            continue

        key = finding_text.lower()
        if key in seen:
            continue
        seen.add(key)

        explanation = _points_to_paragraph(item.key_points)
        findings.append(
            KeyFinding(
                finding=finding_text,
                explanation=explanation,
                confidence=round(item.confidence or 0, 3),
                citations=_citations_for([item]),
            )
        )
        if len(findings) >= MAX_FINDINGS:
            break

    return findings


def _build_sections(
    topic: str,
    evidence: list[Evidence],
    clusters: list[EvidenceCluster],
) -> list[ReportSection]:
    overview = _overview_section(topic, evidence)
    sections = [overview]

    for cluster in clusters[:MAX_CLUSTER_COUNT]:
        cluster_evidence = [item for item in evidence if item.id in cluster.evidence_ids]
        content = _cluster_content(cluster.theme, cluster_evidence)
        sections.append(
            ReportSection(
                heading=cluster.theme.title(),
                content=content,
                citations=cluster.citations,
            )
        )

    caveats = _caveats_section(evidence)
    if caveats:
        sections.append(caveats)

    return sections


def _overview_section(topic: str, evidence: list[Evidence]) -> ReportSection:
    top_points = []
    for item in evidence[:MAX_SECTION_EVIDENCE]:
        if item.gist:
            top_points.append(item.gist)
        elif item.key_points:
            top_points.append(item.key_points[0])

    content = (
        f"The available evidence on {topic} points to {len(evidence)} extracted "
        f"source-level findings. The strongest signals are: "
        f"{_points_to_paragraph(top_points[:4])}"
    )
    return ReportSection(
        heading="Overview",
        content=_truncate(content, 3_000),
        citations=_citations_for(evidence[:MAX_SECTION_EVIDENCE]),
    )


def _cluster_content(theme: str, evidence: list[Evidence]) -> str:
    if not evidence:
        return f"No detailed evidence was available for {theme}."

    lines = []
    for item in evidence[:MAX_SECTION_EVIDENCE]:
        point = item.gist or _points_to_paragraph(item.key_points[:2]) or item.text
        lines.append(_truncate(_normalize_text(point), 600))
    return _points_to_paragraph(lines)


def _caveats_section(evidence: list[Evidence]) -> Optional[ReportSection]:
    caveats: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        for caveat in item.caveats:
            normalized = _normalize_text(caveat)
            key = normalized.lower()
            if key in seen or "no major caveats" in key:
                continue
            seen.add(key)
            caveats.append(normalized)

    if not caveats:
        return None

    return ReportSection(
        heading="Caveats",
        content=_points_to_paragraph(caveats[:8]),
        citations=_citations_for(evidence),
    )


def _cluster_evidence(evidence: list[Evidence]) -> list[EvidenceCluster]:
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        theme = _theme_for(item)
        grouped[theme].append(item)

    clusters: list[EvidenceCluster] = []
    for theme, items in sorted(
        grouped.items(),
        key=lambda pair: _average_confidence(pair[1]),
        reverse=True,
    ):
        summary_parts = []
        for item in items[:4]:
            if item.gist:
                summary_parts.append(item.gist)
            elif item.key_points:
                summary_parts.append(item.key_points[0])

        clusters.append(
            EvidenceCluster(
                id=_cluster_id(theme),
                theme=theme,
                summary=_truncate(_points_to_paragraph(summary_parts), 2_000),
                evidence_ids=[item.id for item in items],
                citations=_citations_for(items),
            )
        )
        if len(clusters) >= MAX_CLUSTER_COUNT:
            break

    return clusters


def _build_executive_summary(
    topic: str,
    findings: list[KeyFinding],
    limitations: list[str],
    source_count: int,
) -> str:
    finding_text = _points_to_paragraph([finding.finding for finding in findings[:4]])
    limitation_text = _points_to_paragraph(limitations[:3])
    summary = (
        f"This report synthesizes evidence on {topic} from {source_count} source"
        f"{'' if source_count == 1 else 's'}. {finding_text}"
    )
    if limitation_text:
        summary += f" Main limitations: {limitation_text}"
    return _truncate(summary, MAX_SUMMARY_CHARS)


def _build_limitations(evidence: list[Evidence], sources: list[Source]) -> list[str]:
    limitations: list[str] = []

    low_confidence = [item for item in evidence if (item.confidence or 0) < 0.45]
    if low_confidence:
        limitations.append(
            f"{len(low_confidence)} evidence item(s) have low confidence scores."
        )

    if not sources:
        limitations.append(
            "Original Source objects were not supplied, so the report relies on evidence citations."
        )

    unique_sources = _unique_source_ids(evidence)
    if len(unique_sources) < 3:
        limitations.append("The report is based on fewer than three unique sources.")

    caveat_counts = sum(
        1
        for item in evidence
        for caveat in item.caveats
        if "no major caveats" not in caveat.lower()
    )
    if caveat_counts:
        limitations.append(
            f"The extracted evidence includes {caveat_counts} caveat or uncertainty note(s)."
        )

    return limitations or ["No major synthesis limitations were detected."]


def _build_follow_up_questions(
    topic: str, clusters: list[EvidenceCluster]
) -> list[str]:
    questions = [
        f"What recent evidence would most change the conclusions about {topic}?",
        f"Which sources provide the strongest counterarguments on {topic}?",
    ]
    for cluster in clusters[:3]:
        questions.append(f"What additional evidence is needed about {cluster.theme}?")
    return questions


def _rank_evidence(evidence: list[Evidence]) -> list[Evidence]:
    valid = [item for item in evidence if item.text.strip()]
    return sorted(
        valid,
        key=lambda item: (
            item.confidence or 0,
            item.relevance_score or 0,
            len(item.key_points),
        ),
        reverse=True,
    )


def _theme_for(evidence: Evidence) -> str:
    if evidence.tags:
        return " ".join(evidence.tags[:2])

    text = " ".join([evidence.gist or "", *evidence.key_points, evidence.text])
    keywords = _keywords(text)
    if keywords:
        return " ".join(keywords[:2])
    return "general findings"


def _citations_for(evidence: list[Evidence]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for item in evidence:
        if not item.citation:
            continue
        key = f"{item.citation.source_id}:{item.citation.locator or ''}"
        if key in seen:
            continue
        seen.add(key)
        citations.append(item.citation)
    return citations


def _dedupe_sources(sources: list[Source]) -> list[Source]:
    deduped: list[Source] = []
    seen: set[str] = set()
    for source in sources:
        key = (source.doi or str(source.url or "") or source.id).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _unique_source_ids(evidence: list[Evidence]) -> set[str]:
    return {item.source_id for item in evidence if item.source_id}


def _average_confidence(evidence: list[Evidence]) -> float:
    if not evidence:
        return 0
    return sum(item.confidence or 0 for item in evidence) / len(evidence)


def _points_to_paragraph(points: list[str]) -> str:
    cleaned = [_normalize_text(point) for point in points if _normalize_text(point)]
    if not cleaned:
        return ""
    return " ".join(
        point if point.endswith((".", "!", "?")) else f"{point}."
        for point in cleaned
    )


def _first_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[0] if sentences else text


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
    words = []
    seen = set()
    for word in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower()):
        if word in stopwords or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def _normalize_topic(topic: str) -> str:
    normalized = _normalize_text(topic)
    if len(normalized) < 3:
        raise SynthesisError("topic must be at least 3 characters.")
    return normalized


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _report_id(topic: str, evidence: list[Evidence]) -> str:
    evidence_key = ",".join(item.id for item in evidence)
    digest = hashlib.sha1(f"{topic}:{evidence_key}".encode("utf-8")).hexdigest()[:10]
    return f"R-{digest}"


def _cluster_id(theme: str) -> str:
    digest = hashlib.sha1(theme.encode("utf-8")).hexdigest()[:10]
    return f"C-{digest}"
