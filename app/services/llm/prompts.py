from app.schemas.evidence import Evidence
from app.schemas.source import Source


MAX_SOURCE_TEXT_CHARS = 30_000
MAX_EVIDENCE_TEXT_CHARS = 20_000


def research_query_system_prompt() -> str:
    return (
        "You generate high-quality search queries for a research agent. "
        "Return concise, specific queries that improve recall across web pages, "
        "news, papers, datasets, and technical sources. Avoid vague wording."
    )


def research_query_user_prompt(topic: str, *, max_queries: int = 6) -> str:
    return f"""Topic:
{topic}

Task:
Generate {max_queries} search queries for researching this topic.

Requirements:
- Include at least one broad web query.
- Include at least one academic paper query.
- Include at least one query for recent evidence or updates.
- Include synonyms, technical terms, or related concepts when useful.
- Do not include explanations.
"""


def evidence_extraction_system_prompt() -> str:
    return (
        "You extract research evidence from one source at a time. "
        "Use only the provided source text and metadata. Do not invent claims, "
        "citations, authors, dates, or statistics. Be concise, skeptical, and "
        "topic-focused. If the source is weak or only loosely relevant, say so "
        "in caveats and lower confidence."
    )


def evidence_extraction_user_prompt(topic: str, source: Source) -> str:
    return f"""Research topic:
{topic}

Source metadata:
- id: {source.id}
- title: {source.title}
- url: {source.url or "not available"}
- source_type: {source.source_type.value}
- provider: {source.provider.value if source.provider else "not available"}
- authors: {", ".join(source.authors) if source.authors else "not available"}
- published_at: {source.published_at.isoformat() if source.published_at else "not available"}
- doi: {source.doi or "not available"}
- summary: {source.summary or "not available"}

Source text:
{_truncate(source.full_text or "", MAX_SOURCE_TEXT_CHARS)}

Task:
Extract one evidence object for the research topic.

The output should capture:
- gist: the source's main topic-relevant message
- key_points: 3 to 6 specific points from the source
- caveats: limitations, uncertainty, missing metadata, conflicts, weak relevance, or methodological issues
- confidence: 0 to 1 confidence in this source as evidence for the topic
- relevance_score: 0 to 1 topic relevance
- tags: short lowercase topic tags

Rules:
- Use only the source text and metadata above.
- Keep claims attributable to this source.
- Prefer concrete findings over generic summary.
- Preserve uncertainty and limitations.
- Do not quote long passages.
"""


def report_synthesis_system_prompt() -> str:
    return (
        "You synthesize research evidence into a structured report. "
        "Use only the provided evidence. Do not invent sources or citations. "
        "Resolve agreement and disagreement explicitly. Keep uncertainty visible."
    )


def report_synthesis_user_prompt(topic: str, evidence: list[Evidence]) -> str:
    return f"""Research topic:
{topic}

Evidence:
{format_evidence_for_prompt(evidence)}

Task:
Write a structured research report for the topic.

The report should include:
- a precise title
- an executive summary
- key findings with citations
- sections grouped by theme
- limitations
- follow-up questions

Rules:
- Use only the evidence above.
- Cite using source_id values.
- Do not add claims that are not supported by evidence.
- Call out weak, conflicting, or low-confidence evidence.
- Make the final answer useful to a reader deciding what the evidence supports.
"""


def source_relevance_system_prompt() -> str:
    return (
        "You judge source relevance for a research agent. "
        "Score only based on the provided metadata and text excerpt. "
        "Prefer sources with direct evidence over broad background."
    )


def source_relevance_user_prompt(topic: str, source: Source) -> str:
    return f"""Research topic:
{topic}

Source:
- id: {source.id}
- title: {source.title}
- url: {source.url or "not available"}
- source_type: {source.source_type.value}
- summary: {source.summary or "not available"}
- text excerpt: {_truncate(source.full_text or "", 8_000)}

Task:
Judge whether this source should be used for the report.

Return:
- relevance_score from 0 to 1
- one-sentence rationale
- any caveat that affects source usefulness
"""


def citation_check_system_prompt() -> str:
    return (
        "You check whether report claims are supported by cited evidence. "
        "Use only the provided claim and evidence. Be strict."
    )


def citation_check_user_prompt(claim: str, evidence: list[Evidence]) -> str:
    return f"""Claim:
{claim}

Evidence:
{format_evidence_for_prompt(evidence)}

Task:
Decide whether the claim is fully supported, partially supported, contradicted, or unsupported.

Rules:
- Use only the evidence above.
- Identify the source_ids that support or weaken the claim.
- Explain missing support briefly.
"""


def format_evidence_for_prompt(evidence: list[Evidence]) -> str:
    blocks = []
    total_chars = 0

    for item in evidence:
        block = f"""Evidence {item.id}
- source_id: {item.source_id}
- confidence: {item.confidence if item.confidence is not None else "not available"}
- relevance_score: {item.relevance_score if item.relevance_score is not None else "not available"}
- gist: {item.gist or "not available"}
- key_points:
{_bullet_list(item.key_points)}
- caveats:
{_bullet_list(item.caveats)}
- citation_title: {item.citation.title if item.citation else "not available"}
"""
        if total_chars + len(block) > MAX_EVIDENCE_TEXT_CHARS:
            break
        blocks.append(block)
        total_chars += len(block)

    return "\n".join(blocks) if blocks else "No evidence provided."


def _bullet_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def _truncate(text: str, max_chars: int) -> str:
    clean = " ".join(text.strip().split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."
