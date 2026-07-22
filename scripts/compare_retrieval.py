from __future__ import annotations

import argparse
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
from src.ingestion.embedding_service import OpenAIEmbeddingProvider
from src.models import RetrievalHit
from src.retrieval.service import HybridRetriever


def _print_hits(label: str, hits: list[RetrievalHit]) -> None:
    print(f"\n{label}")
    print("=" * len(label))
    for index, hit in enumerate(hits, start=1):
        location = f"Article {hit.article_number}" if hit.article_number else (hit.section_title or "Unknown section")
        reasons = "; ".join(hit.retrieval_reasons) or "direct hybrid match"
        print(
            f"{index}. {hit.document_title} | {location} | PDF page {hit.pdf_page_start} "
            f"| score={hit.final_score:.4f}"
        )
        print(f"   {reasons}")
        print(f"   retrieval paths: {', '.join(hit.retrieval_sources) or 'legacy weighted search'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the original retriever with the local GraphRAG experiment."
    )
    parser.add_argument("question", help="Student regulation question to compare")
    args = parser.parse_args()

    settings = get_settings()
    errors = settings.public_configuration_errors()
    if errors:
        raise SystemExit("Missing configuration: " + ", ".join(errors))

    repository = SupabaseRepository(create_supabase_client(settings))
    embedding_provider = OpenAIEmbeddingProvider(settings)

    legacy_settings = settings.model_copy(update={"experimental_graphrag": False})
    graph_settings = settings.model_copy(update={"experimental_graphrag": True})
    legacy = HybridRetriever(repository, embedding_provider, legacy_settings)
    graph = HybridRetriever(repository, embedding_provider, graph_settings)

    legacy_hits, _ = legacy.retrieve(args.question, language="auto")
    graph_hits, _ = graph.retrieve(args.question, language="auto")
    _print_hits("Original retriever", legacy_hits)
    _print_hits("Local GraphRAG experiment", graph_hits)


if __name__ == "__main__":
    main()
