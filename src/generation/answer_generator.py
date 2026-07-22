from __future__ import annotations

from typing import Protocol

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings
from src.generation.citation_builder import ensure_source_references
from src.generation.prompts import LOCAL_SYSTEM_PROMPT, SYSTEM_PROMPT, build_user_prompt
from src.models import EvidenceSource
from src.retrieval.context_builder import format_evidence


class GenerationProvider(Protocol):
    def answer(
        self,
        question: str,
        evidence: list[EvidenceSource],
        *,
        safety_identifier: str | None = None,
    ) -> str: ...


class OpenAIGenerationProvider:
    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(api_key=settings.openai_key)
        self._model = settings.openai_generation_model
        self._max_output_tokens = settings.max_answer_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def answer(
        self,
        question: str,
        evidence: list[EvidenceSource],
        *,
        safety_identifier: str | None = None,
    ) -> str:
        request: dict[str, object] = {
            "model": self._model,
            "instructions": SYSTEM_PROMPT,
            "input": build_user_prompt(question, format_evidence(evidence)),
            "max_output_tokens": self._max_output_tokens,
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "medium"},
            "store": False,
        }
        if safety_identifier:
            request["safety_identifier"] = safety_identifier
        response = self._client.responses.create(**request)
        answer = response.output_text.strip()
        if not answer:
            raise RuntimeError("The generation model returned an empty answer.")
        return ensure_source_references(answer, evidence)


class OllamaGenerationProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(timeout=settings.ollama_timeout_seconds)
        self._url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        self._model = settings.ollama_model
        self._context_window = settings.ollama_context_window
        self._max_output_tokens = settings.ollama_max_output_tokens
        self._evidence_chunks = settings.ollama_evidence_chunks
        self._keep_alive = settings.ollama_keep_alive

    def answer(
        self,
        question: str,
        evidence: list[EvidenceSource],
        *,
        safety_identifier: str | None = None,
    ) -> str:
        del safety_identifier
        local_evidence = evidence[: self._evidence_chunks]
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        question,
                        format_evidence(local_evidence),
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": {
                "num_ctx": self._context_window,
                "num_predict": self._max_output_tokens,
                "temperature": 0.0,
                "seed": 42,
            },
        }
        try:
            response = self._client.post(self._url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RuntimeError(
                "The local Ollama model is unavailable. Make sure Ollama is "
                f"running and that {self._model} is installed."
            ) from error

        data = response.json()
        answer = str((data.get("message") or {}).get("content") or "").strip()
        if not answer:
            raise RuntimeError("The local Ollama model returned an empty answer.")
        return ensure_source_references(answer, local_evidence)
