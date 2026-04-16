import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from app.schemas.source import Source


DEFAULT_TARGET_TOKENS = 450
DEFAULT_MAX_TOKENS = 750
DEFAULT_OVERLAP_TOKENS = 80
MIN_CHUNK_TOKENS = 80


@dataclass(frozen=True)
class TextChunk:
    id: str
    source_id: str
    text: str
    index: int
    token_count: int
    heading: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TextBlock:
    text: str
    heading: Optional[str]
    start_char: int
    end_char: int


def chunk_source(
    source: Source,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextChunk]:
    if not source.full_text:
        return []

    return chunk_text(
        text=source.full_text,
        source_id=source.id,
        title=source.title,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def chunk_sources(
    sources: list[Source],
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for source in sources:
        chunks.extend(
            chunk_source(
                source,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return chunks


def chunk_text(
    text: str,
    source_id: str,
    *,
    title: Optional[str] = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextChunk]:
    clean = _clean_text(text)
    if not clean:
        return []

    _validate_limits(target_tokens, max_tokens, overlap_tokens)

    blocks = _semantic_blocks(clean, title)
    chunks: list[TextChunk] = []
    pending: list[TextBlock] = []
    pending_tokens = 0

    for block in blocks:
        block_tokens = estimate_tokens(block.text)

        if block_tokens > max_tokens:
            chunks.extend(
                _flush_pending(
                    pending,
                    source_id=source_id,
                    start_index=len(chunks),
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            pending = []
            pending_tokens = 0
            chunks.extend(
                _split_large_block(
                    block,
                    source_id=source_id,
                    start_index=len(chunks),
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            continue

        if pending and pending_tokens + block_tokens > target_tokens:
            chunks.extend(
                _flush_pending(
                    pending,
                    source_id=source_id,
                    start_index=len(chunks),
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            pending = _overlap_blocks(pending, overlap_tokens)
            pending_tokens = sum(estimate_tokens(item.text) for item in pending)

        pending.append(block)
        pending_tokens += block_tokens

    chunks.extend(
        _flush_pending(
            pending,
            source_id=source_id,
            start_index=len(chunks),
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
    )

    return _renumber_chunks(chunks)


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def _semantic_blocks(text: str, title: Optional[str]) -> list[TextBlock]:
    paragraphs = _paragraphs_with_offsets(text)
    blocks: list[TextBlock] = []
    current_heading = title

    for paragraph, start, end in paragraphs:
        if _looks_like_heading(paragraph):
            current_heading = paragraph
            continue

        if estimate_tokens(paragraph) < 12 and blocks:
            previous = blocks[-1]
            merged = f"{previous.text}\n{paragraph}"
            blocks[-1] = TextBlock(
                text=merged,
                heading=previous.heading,
                start_char=previous.start_char,
                end_char=end,
            )
            continue

        blocks.append(
            TextBlock(
                text=paragraph,
                heading=current_heading,
                start_char=start,
                end_char=end,
            )
        )

    return blocks


def _flush_pending(
    blocks: list[TextBlock],
    *,
    source_id: str,
    start_index: int,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    if not blocks:
        return []

    text = "\n\n".join(block.text for block in blocks)
    token_count = estimate_tokens(text)
    if token_count > max_tokens:
        return _split_large_block(
            TextBlock(
                text=text,
                heading=blocks[0].heading,
                start_char=blocks[0].start_char,
                end_char=blocks[-1].end_char,
            ),
            source_id=source_id,
            start_index=start_index,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

    if token_count < MIN_CHUNK_TOKENS and start_index > 0:
        pass

    heading = _shared_heading(blocks)
    return [
        _make_chunk(
            source_id=source_id,
            index=start_index,
            text=text,
            heading=heading,
            start_char=blocks[0].start_char,
            end_char=blocks[-1].end_char,
        )
    ]


def _split_large_block(
    block: TextBlock,
    *,
    source_id: str,
    start_index: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    sentences = _sentences(block.text)
    if not sentences:
        return []

    chunks: list[TextChunk] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(
                    _sentence_chunk(
                        source_id, start_index + len(chunks), current, block
                    )
                )
                current = _overlap_sentences(current, overlap_tokens)
                current_tokens = sum(estimate_tokens(item) for item in current)
            chunks.extend(
                _split_by_words(
                    sentence,
                    source_id=source_id,
                    start_index=start_index + len(chunks),
                    block=block,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(
                _sentence_chunk(source_id, start_index + len(chunks), current, block)
            )
            current = _overlap_sentences(current, overlap_tokens)
            current_tokens = sum(estimate_tokens(item) for item in current)

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(
            _sentence_chunk(source_id, start_index + len(chunks), current, block)
        )

    return chunks


def _split_by_words(
    text: str,
    *,
    source_id: str,
    start_index: int,
    block: TextBlock,
    max_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    words = text.split()
    chunks: list[TextChunk] = []
    step = max(max_tokens - overlap_tokens, 1)

    for start in range(0, len(words), step):
        part = " ".join(words[start : start + max_tokens])
        if not part:
            continue
        chunks.append(
            _make_chunk(
                source_id=source_id,
                index=start_index + len(chunks),
                text=part,
                heading=block.heading,
                start_char=block.start_char,
                end_char=block.end_char,
            )
        )
        if start + max_tokens >= len(words):
            break

    return chunks


def _sentence_chunk(
    source_id: str,
    index: int,
    sentences: list[str],
    block: TextBlock,
) -> TextChunk:
    return _make_chunk(
        source_id=source_id,
        index=index,
        text=" ".join(sentences),
        heading=block.heading,
        start_char=block.start_char,
        end_char=block.end_char,
    )


def _make_chunk(
    *,
    source_id: str,
    index: int,
    text: str,
    heading: Optional[str],
    start_char: Optional[int],
    end_char: Optional[int],
) -> TextChunk:
    cleaned = _clean_text(text)
    return TextChunk(
        id=_chunk_id(source_id, index, cleaned),
        source_id=source_id,
        text=cleaned,
        index=index,
        token_count=estimate_tokens(cleaned),
        heading=heading,
        start_char=start_char,
        end_char=end_char,
        metadata={"heading": heading or ""},
    )


def _overlap_blocks(blocks: list[TextBlock], overlap_tokens: int) -> list[TextBlock]:
    if overlap_tokens <= 0:
        return []

    selected: list[TextBlock] = []
    total = 0
    for block in reversed(blocks):
        selected.insert(0, block)
        total += estimate_tokens(block.text)
        if total >= overlap_tokens:
            break
    return selected


def _overlap_sentences(sentences: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []

    selected: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        selected.insert(0, sentence)
        total += estimate_tokens(sentence)
        if total >= overlap_tokens:
            break
    return selected


def _paragraphs_with_offsets(text: str) -> list[tuple[str, int, int]]:
    paragraphs: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S(?:.*?)(?=\n{2,}|\Z)", text, flags=re.DOTALL):
        paragraph = _clean_text(match.group(0))
        if paragraph:
            paragraphs.append((paragraph, match.start(), match.end()))
    return paragraphs


def _sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    return [piece for piece in pieces if piece]


def _looks_like_heading(paragraph: str) -> bool:
    tokens = estimate_tokens(paragraph)
    if tokens > 14:
        return False
    if paragraph.endswith("."):
        return False
    if re.match(r"^\d+(\.\d+)*\s+\S+", paragraph):
        return True
    if paragraph.isupper() and tokens <= 10:
        return True
    return tokens <= 8 and not re.search(r"[.!?;:]", paragraph)


def _shared_heading(blocks: list[TextBlock]) -> Optional[str]:
    if not blocks:
        return None
    heading = blocks[0].heading
    if all(block.heading == heading for block in blocks):
        return heading
    return heading


def _renumber_chunks(chunks: list[TextChunk]) -> list[TextChunk]:
    return [
        TextChunk(
            id=_chunk_id(chunk.source_id, index, chunk.text),
            source_id=chunk.source_id,
            text=chunk.text,
            index=index,
            token_count=chunk.token_count,
            heading=chunk.heading,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            metadata=chunk.metadata,
        )
        for index, chunk in enumerate(chunks)
        if chunk.text
    ]


def _validate_limits(target_tokens: int, max_tokens: int, overlap_tokens: int) -> None:
    if target_tokens < 50:
        raise ValueError("target_tokens must be at least 50.")
    if max_tokens < target_tokens:
        raise ValueError("max_tokens must be greater than or equal to target_tokens.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative.")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_id(source_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_id}:{index}:{text}".encode("utf-8")).hexdigest()
    return f"CH-{digest[:12]}"
