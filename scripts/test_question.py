from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import get_settings
from src.database.repositories import SupabaseRepository
from src.database.supabase_client import create_supabase_client
from src.generation.answer_generator import (
    NvidiaGenerationProvider,
    OllamaGenerationProvider,
    OpenAIGenerationProvider,
)
from src.generation.citation_builder import source_summary
from src.generation.service import QuestionAnsweringService
from src.ingestion.embedding_service import OpenAIEmbeddingProvider
from src.retrieval.service import HybridRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask one question from the terminal.")
    parser.add_argument("question")
    parser.add_argument("--category")
    parser.add_argument("--language", default="auto", choices=("auto", "en", "ar"))
    parser.add_argument(
        "--provider",
        choices=("nvidia", "ollama", "openai"),
        default="nvidia",
        help="Answer generator to use (retrieval embeddings remain OpenAI).",
    )
    args = parser.parse_args()

    settings = get_settings()
    errors = settings.public_configuration_errors()
    errors.extend(settings.generation_configuration_errors(args.provider))
    if errors:
        parser.error("; ".join(errors))
    effective_settings = settings
    if args.provider == "ollama":
        effective_settings = settings.model_copy(
            update={
                "final_evidence_chunks": min(
                    settings.final_evidence_chunks,
                    settings.ollama_evidence_chunks,
                )
            }
        )
    repository = SupabaseRepository(create_supabase_client(effective_settings))
    embedding_provider = OpenAIEmbeddingProvider(effective_settings)
    if args.provider == "ollama":
        generator = OllamaGenerationProvider(effective_settings)
    elif args.provider == "nvidia":
        generator = NvidiaGenerationProvider(effective_settings)
    else:
        generator = OpenAIGenerationProvider(effective_settings)
    service = QuestionAnsweringService(
        HybridRetriever(repository, embedding_provider, effective_settings),
        generator,
    )
    safety_id = hashlib.sha256(b"local-test-user").hexdigest()
    result = service.answer(
        args.question,
        document_type=args.category,
        language=args.language,
        safety_identifier=safety_id,
    )
    print(result.text)
    for source in result.sources:
        print(source_summary(source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
