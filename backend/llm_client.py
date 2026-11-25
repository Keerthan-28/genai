"""Local Ollama LLM client wrapper for VeriReport AI.

Handles:
- Local Ollama service reachability checks (`GET /api/tags`)
- Model availability verification before selection
- Configurable model & fallback selection
- Controlled timeouts, retries, and exponential backoff
- Optional structured JSON output mode (`format: "json"`)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import requests

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

_EMBEDDING_MODEL_PREFIXES = ("nomic-embed", "all-minilm", "mxbai-embed", "bge-")


class LLMConfigurationError(RuntimeError):
    """Raised when the local Ollama server or model configuration is unavailable."""


@dataclass(frozen=True)
class ModelCheckResult:
    """Result of checking local Ollama reachability and model availability."""

    provider: str
    service_reachable: bool
    configured_model: str
    selected_model: str | None
    model_verified: bool
    available_models: list[str]
    message: str


class OllamaClientWrapper:
    """HTTP client wrapper around a local Ollama server."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._verified_model: str | None = None

    @property
    def base_url(self) -> str:
        return self.settings.ollama_base_url.rstrip("/")

    @staticmethod
    def _is_chat_model(name: str) -> bool:
        lowered = name.lower()
        return not any(lowered.startswith(prefix) for prefix in _EMBEDDING_MODEL_PREFIXES)

    def list_available_models(self) -> list[str]:
        """Query local Ollama `/api/tags` for installed models."""
        url = f"{self.base_url}/api/tags"
        resp = requests.get(url, timeout=min(15, max(5, self.settings.ollama_timeout_seconds)))
        resp.raise_for_status()
        payload = resp.json()
        models = payload.get("models", [])
        return [str(m.get("name", "")).strip() for m in models if m.get("name")]

    def verify_model_availability(self, quick: bool = False) -> ModelCheckResult:
        """Check whether Ollama is reachable and the configured chat model exists."""
        configured = self.settings.ollama_model.strip()
        fallback = self.settings.ollama_fallback_model.strip()

        try:
            if quick:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
                resp.raise_for_status()
                models = resp.json().get("models", [])
                available = [str(m.get("name", "")).strip() for m in models if m.get("name")]
            else:
                available = self._with_retries(self.list_available_models)
        except Exception as exc:
            logger.warning("Ollama availability check failed: %s", type(exc).__name__)
            return ModelCheckResult(
                provider="ollama",
                service_reachable=False,
                configured_model=configured,
                selected_model=None,
                model_verified=False,
                available_models=[],
                message=f"Ollama unreachable at {self.base_url} ({type(exc).__name__}): {exc}",
            )

        selected: str | None = None
        if configured in available:
            selected = configured
        elif fallback in available:
            selected = fallback
        else:
            chat_candidates = [m for m in available if self._is_chat_model(m)]
            if chat_candidates:
                selected = chat_candidates[0]

        self._verified_model = selected
        verified = selected is not None
        msg = (
            f"Verified Ollama model: {selected}"
            if verified
            else (
                f"Configured model '{configured}' and fallback '{fallback}' not found in Ollama. "
                f"Installed models: {available}"
            )
        )
        return ModelCheckResult(
            provider="ollama",
            service_reachable=True,
            configured_model=configured,
            selected_model=selected,
            model_verified=verified,
            available_models=available,
            message=msg,
        )

    def resolve_model(self) -> str:
        """Return a verified Ollama model name, checking availability if needed."""
        if self._verified_model:
            return self._verified_model
        check = self.verify_model_availability()
        if not check.model_verified or not check.selected_model:
            raise LLMConfigurationError(check.message)
        return check.selected_model

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        response_mime_type: str | None = None,
        response_schema: Any | None = None,
        max_tokens: int = 700,
    ) -> str:
        """Generate text or JSON from local Ollama `/api/chat` with retries and timeout."""
        model = self.resolve_model()
        url = f"{self.base_url}/api/chat"

        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if response_mime_type == "application/json" or response_schema is not None:
            payload["format"] = "json"

        def _call() -> str:
            resp = requests.post(
                url,
                json=payload,
                timeout=max(5, self.settings.ollama_timeout_seconds),
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message") or {}
            return str(message.get("content") or data.get("response") or "")

        return self._with_retries(_call)

    def _with_retries(self, fn: Callable[[], T]) -> T:
        """Execute a callable with bounded retries and exponential backoff."""
        max_attempts = max(1, self.settings.ollama_max_retries)
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    break
                delay = min(4.0, float(2 ** (attempt - 1)))
                logger.warning(
                    "Ollama call attempt %d/%d failed (%s). Retrying in %.1fs.",
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc
