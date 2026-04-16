import asyncio
import json
import os
from enum import Enum
from typing import Any, Optional, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError


DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 2_000
DEFAULT_TIMEOUT_SECONDS = 60

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434"

T = TypeVar("T", bound=BaseModel)


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMClientError(RuntimeError):
    """Raised when the LLM client is misconfigured or the provider call fails."""


class LLMClient:
    def __init__(
        self,
        *,
        provider: Optional[LLMProvider | str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        base_url: Optional[str] = None,
    ) -> None:
        self.provider = _provider(provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER)
        self.api_key = api_key if api_key is not None else _api_key_for(self.provider)
        self.model = model or _model_for(self.provider)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.base_url = (base_url or _base_url_for(self.provider)).rstrip("/")

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        selected_model = model or self.model
        selected_temperature = self.temperature if temperature is None else temperature
        selected_max_tokens = max_output_tokens or self.max_output_tokens

        if self.provider == LLMProvider.GEMINI:
            return self._generate_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=selected_model,
                temperature=selected_temperature,
                max_output_tokens=selected_max_tokens,
            )
        if self.provider == LLMProvider.OLLAMA:
            return self._generate_ollama(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=selected_model,
                temperature=selected_temperature,
                max_output_tokens=selected_max_tokens,
            )
        return self._generate_openai_compatible(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=selected_model,
            temperature=selected_temperature,
            max_output_tokens=selected_max_tokens,
        )

    async def generate_text_async(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        return await asyncio.to_thread(
            self.generate_text,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> T:
        schema_prompt = _structured_prompt(user_prompt, response_model)
        text = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=schema_prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return _parse_structured_response(text, response_model)

    async def generate_structured_async(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> T:
        return await asyncio.to_thread(
            self.generate_structured,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _generate_gemini(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        if not self.api_key:
            raise LLMClientError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")

        url = (
            f"{self.base_url}/models/{model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        data = _request_json(url, payload, {}, self.timeout_seconds)
        return _gemini_text(data)

    def _generate_openai_compatible(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        if not self.api_key:
            env_name = {
                LLMProvider.GROQ: "GROQ_API_KEY",
                LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
                LLMProvider.OPENAI: "OPENAI_API_KEY",
            }.get(self.provider, "API key")
            raise LLMClientError(f"{env_name} is required when LLM_PROVIDER={self.provider.value}.")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.provider == LLMProvider.OPENROUTER:
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "research-agent")

        payload = {
            "model": model,
            "messages": _messages(system_prompt, user_prompt),
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        data = _request_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers,
            self.timeout_seconds,
        )
        return _chat_completion_text(data)

    def _generate_ollama(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload = {
            "model": model,
            "messages": _messages(system_prompt, user_prompt),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
        }
        data = _request_json(
            f"{self.base_url}/api/chat",
            payload,
            {},
            self.timeout_seconds,
        )
        message = data.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        raise LLMClientError("Ollama response did not include message content.")


def get_llm_client() -> LLMClient:
    return LLMClient()


def _messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _request_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            **headers,
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMClientError(f"LLM provider returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise LLMClientError(f"LLM provider request failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClientError("LLM provider returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise LLMClientError("LLM provider returned an unexpected payload.")
    return data


def _gemini_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMClientError("Gemini response did not include candidates.")

    content = candidates[0].get("content")
    if not isinstance(content, dict):
        raise LLMClientError("Gemini response did not include content.")

    parts = content.get("parts")
    if not isinstance(parts, list):
        raise LLMClientError("Gemini response did not include parts.")

    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    text = "\n".join(part for part in text_parts if part).strip()
    if not text:
        raise LLMClientError("Gemini response did not include text.")
    return text


def _chat_completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMClientError("Chat completion response did not include choices.")

    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise LLMClientError("Chat completion response did not include message content.")
    return message["content"].strip()


def _structured_prompt(user_prompt: str, response_model: type[BaseModel]) -> str:
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    return f"""{user_prompt}

Return only valid JSON matching this JSON schema. Do not include markdown fences,
comments, or explanatory text.

JSON schema:
{schema}
"""


def _parse_structured_response(text: str, response_model: type[T]) -> T:
    json_text = _extract_json(text)
    try:
        return response_model.model_validate_json(json_text)
    except ValidationError as exc:
        raise LLMClientError(f"Structured response did not match schema: {exc}") from exc


def _extract_json(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()

    if clean.startswith("{") or clean.startswith("["):
        return clean

    start = min(
        [index for index in (clean.find("{"), clean.find("[")) if index != -1],
        default=-1,
    )
    if start == -1:
        raise LLMClientError("Structured response did not contain JSON.")

    return clean[start:].strip()


def _provider(value: LLMProvider | str) -> LLMProvider:
    if isinstance(value, LLMProvider):
        return value

    normalized = value.strip().lower()
    aliases = {
        "google": LLMProvider.GEMINI,
        "gemini": LLMProvider.GEMINI,
        "groq": LLMProvider.GROQ,
        "openrouter": LLMProvider.OPENROUTER,
        "openai": LLMProvider.OPENAI,
        "ollama": LLMProvider.OLLAMA,
        "local": LLMProvider.OLLAMA,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported LLM provider: {value}")
    return aliases[normalized]


def _api_key_for(provider: LLMProvider) -> Optional[str]:
    if provider == LLMProvider.GEMINI:
        return os.getenv("GEMINI_API_KEY")
    if provider == LLMProvider.GROQ:
        return os.getenv("GROQ_API_KEY")
    if provider == LLMProvider.OPENROUTER:
        return os.getenv("OPENROUTER_API_KEY")
    if provider == LLMProvider.OPENAI:
        return os.getenv("OPENAI_API_KEY")
    return None


def _model_for(provider: LLMProvider) -> str:
    if provider == LLMProvider.GEMINI:
        return os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    if provider == LLMProvider.GROQ:
        return os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    if provider == LLMProvider.OPENROUTER:
        return os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
    if provider == LLMProvider.OPENAI:
        return os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    if provider == LLMProvider.OLLAMA:
        return os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _base_url_for(provider: LLMProvider) -> str:
    if provider == LLMProvider.GEMINI:
        return os.getenv("GEMINI_BASE_URL") or GEMINI_BASE_URL
    if provider == LLMProvider.GROQ:
        return os.getenv("GROQ_BASE_URL") or GROQ_BASE_URL
    if provider == LLMProvider.OPENROUTER:
        return os.getenv("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL
    if provider == LLMProvider.OPENAI:
        return os.getenv("OPENAI_BASE_URL") or OPENAI_BASE_URL
    if provider == LLMProvider.OLLAMA:
        return os.getenv("OLLAMA_BASE_URL") or OLLAMA_BASE_URL
    raise ValueError(f"Unsupported LLM provider: {provider}")
