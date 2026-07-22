from __future__ import annotations

import json

import httpx

from src.config import Settings
from src.generation.answer_generator import OllamaGenerationProvider
from src.models import EvidenceSource, RetrievalHit


def source(index: int) -> EvidenceSource:
    return EvidenceSource(
        source_id=f"S{index}",
        hit=RetrievalHit(
            chunk_id=f"chunk-{index}",
            document_id="document-1",
            document_title="GJU Regulations",
            source_url="https://example.edu/rules.pdf",
            chunk_text=f"Official evidence {index}",
            section_title="Registration",
            article_number=str(index),
            pdf_page_start=index,
            pdf_page_end=index,
        ),
    )


def settings(**updates: object) -> Settings:
    values: dict[str, object] = {
        "openai_api_key": "test-key",
        "supabase_url": "https://example.supabase.co",
        "supabase_anon_key": "test-anon-key",
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def test_ollama_request_uses_low_memory_deterministic_options() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Grounded answer [S1]."}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaGenerationProvider(settings(), client=client)

    answer = provider.answer("What is the rule?", [source(1)])

    assert answer == "Grounded answer [S1]."
    assert captured["think"] is False
    assert captured["stream"] is False
    assert captured["options"]["num_ctx"] == 4096
    assert captured["options"]["temperature"] == 0.0


def test_ollama_limits_evidence_to_configured_count() -> None:
    prompt = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal prompt
        payload = json.loads(request.content)
        prompt = payload["messages"][1]["content"]
        return httpx.Response(200, json={"message": {"content": "Answer [S1]."}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaGenerationProvider(
        settings(ollama_evidence_chunks=2),
        client=client,
    )

    provider.answer("Question", [source(1), source(2), source(3)])

    assert "SOURCE S1" in prompt
    assert "SOURCE S2" in prompt
    assert "SOURCE S3" not in prompt


def test_ollama_adds_source_footer_when_model_omits_citation() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"message": {"content": "Grounded answer."}},
            )
        )
    )
    provider = OllamaGenerationProvider(settings(), client=client)

    answer = provider.answer("Question", [source(1)])

    assert "[S1]" in answer
