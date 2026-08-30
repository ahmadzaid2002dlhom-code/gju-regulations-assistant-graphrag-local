from __future__ import annotations

import json

import httpx
import pytest

from src.config import Settings
from src.generation.answer_generator import NvidiaGenerationProvider
from src.models import EvidenceSource, RetrievalHit


def source() -> EvidenceSource:
    return EvidenceSource(
        source_id="S1",
        hit=RetrievalHit(
            chunk_id="chunk-1",
            document_id="document-1",
            document_title="GJU Regulations",
            source_url="https://example.edu/rules.pdf",
            chunk_text="The maximum regular load is 18 credit hours.",
            section_title="Registration",
            article_number="13",
            pdf_page_start=12,
            pdf_page_end=12,
        ),
    )


def settings(**updates: object) -> Settings:
    values: dict[str, object] = {
        "openai_api_key": "test-openai-key",
        "supabase_url": "https://example.supabase.co",
        "supabase_anon_key": "test-anon-key",
        "nvidia_api_key": "test-nvidia-key",
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def test_nvidia_request_uses_requested_endpoint_model_and_grounding_prompt() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "The maximum is 18 hours [S1]."}}
                ]
            },
        )

    provider = NvidiaGenerationProvider(
        settings(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    answer = provider.answer("What is the maximum load?", [source()])

    payload = captured["payload"]
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-nvidia-key"
    assert payload["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert payload["stream"] is False
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["reasoning_budget"] == 0
    assert "SOURCE S1" in payload["messages"][1]["content"]
    assert answer == "The maximum is 18 hours [S1]."


def test_nvidia_adds_source_footer_when_model_omits_citation() -> None:
    provider = NvidiaGenerationProvider(
        settings(),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "It is 18 hours."}}]},
                )
            )
        ),
    )

    answer = provider.answer("Question", [source()])

    assert "[S1]" in answer


def test_nvidia_requires_api_key_before_request() -> None:
    provider = NvidiaGenerationProvider(settings(nvidia_api_key=""))

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY is missing"):
        provider.answer("Question", [source()])


def test_nvidia_http_error_does_not_expose_response_or_key() -> None:
    provider = NvidiaGenerationProvider(
        settings(),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    401,
                    json={"detail": "secret response details"},
                )
            )
        ),
    )

    with pytest.raises(RuntimeError) as raised:
        provider.answer("Question", [source()])

    message = str(raised.value)
    assert "401" in message
    assert "test-nvidia-key" not in message
    assert "secret response details" not in message
