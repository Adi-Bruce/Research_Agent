import asyncio
import re
from io import BytesIO
from html import unescape
from html.parser import HTMLParser
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pypdf import PdfReader

from app.schemas.source import Source


DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_CHARS = 120_000


class FetchError(RuntimeError):
    """Raised when a source cannot be fetched or converted into text."""


class SourceTextFetcher:
    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_chars = max_chars

    def fetch(self, source: Source) -> Source:
        if source.url is None:
            raise FetchError(f"Source {source.id} does not have a URL to fetch.")

        body, content_type = self._fetch_bytes(str(source.url))
        text = clean_fetched_content(body, content_type, self.max_chars)
        if not text:
            raise FetchError(f"Source {source.id} did not contain extractable text.")

        return source.model_copy(
            update={
                "full_text": text,
                "token_count": estimate_token_count(text),
            }
        )

    async def fetch_async(self, source: Source) -> Source:
        return await asyncio.to_thread(self.fetch, source)

    def _fetch_bytes(self, url: str) -> tuple[bytes, str]:
        request = Request(
            url,
            headers={
                "Accept": (
                    "text/html,application/xhtml+xml,text/plain,"
                    "application/xml;q=0.9,*/*;q=0.5"
                ),
                "Accept-Encoding": "identity",
                "User-Agent": "research-agent/0.1",
            },
            method="GET",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                return response.read(), content_type
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FetchError(f"Fetch returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise FetchError(f"Fetch request failed: {exc}") from exc

def extract_pdf_text(body: bytes) -> str:
    reader = PdfReader(BytesIO(body))
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n\n".join(pages)

def fetch_source(source: Source) -> Source:
    return SourceTextFetcher().fetch(source)


async def fetch_source_async(source: Source) -> Source:
    return await SourceTextFetcher().fetch_async(source)


def fetch_sources(sources: list[Source]) -> list[Source]:
    return [fetch_source(source) for source in sources]


async def fetch_sources_async(sources: list[Source]) -> list[Source]:
    return await asyncio.gather(*(fetch_source_async(source) for source in sources))


def clean_fetched_content(
    body: bytes,
    content_type: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    text = _decode_body(body, content_type)
    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type == "application/pdf" or body.lstrip().startswith(b"%PDF"):
        return clean_text(extract_pdf_text(body))[:max_chars]

    if _looks_like_html(text, media_type):
        text = HTMLTextExtractor.extract(text)

    return clean_text(text)[:max_chars]


def clean_text(text: str) -> str:
    text = unescape(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_token_count(text: str) -> int:
    # Cheap approximation for budgeting chunks before real tokenizer integration.
    return max(1, len(text.split()))


class HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    @classmethod
    def extract(cls, html: str) -> str:
        parser = cls()
        parser.feed(html)
        parser.close()
        return "".join(parser._parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self._append_break()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self._append_break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self._append_text(cleaned)

    def _append_break(self) -> None:
        if self._parts and self._parts[-1] != "\n":
            self._parts.append("\n")

    def _append_text(self, text: str) -> None:
        if not self._parts or self._parts[-1] == "\n":
            self._parts.append(text)
            return
        self._parts.append(f" {text}")


def _decode_body(body: bytes, content_type: str) -> str:
    charset = _charset_from_content_type(content_type) or "utf-8"
    try:
        return body.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")


def _charset_from_content_type(content_type: str) -> Optional[str]:
    for part in content_type.split(";"):
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            return value.strip("\"'")
    return None


def _looks_like_html(text: str, media_type: str) -> bool:
    if media_type in {"text/html", "application/xhtml+xml"}:
        return True
    sample = text[:1000].lower()
    return "<html" in sample or "<body" in sample or "<article" in sample
