from __future__ import annotations

from dataclasses import replace

from src.models import RetrievalHit
from src.retrieval.context_builder import build_evidence, format_evidence
from src.retrieval.fusion import apply_legal_boosts, reciprocal_rank_fusion
from src.retrieval.graph_search import expand_local_graph, extract_article_references
from src.retrieval.query_planner import create_query_plan


def hit(chunk_id: str, article: str, **scores: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="document-1",
        document_title="GJU Regulations",
        source_url="https://example.edu/regulations.pdf",
        chunk_text="Evidence text",
        section_title="Registration",
        article_number=article,
        pdf_page_start=5,
        pdf_page_end=5,
        document_status="current",
        language="en",
        **scores,
    )


class FakeArticleRepository:
    def fetch_article_chunks(
        self,
        document_id: str,
        article_numbers: list[str],
        *,
        match_count: int,
    ) -> list[dict]:
        return [
            {
                "id": f"article-{article}",
                "document_id": document_id,
                "section_id": f"section-{article}",
                "chunk_text": f"Text of Article {article}",
                "section_title": "Registration",
                "article_number": article,
                "pdf_page_start": int(article),
                "pdf_page_end": int(article),
                "document_status": "current",
                "language": "en",
            }
            for article in article_numbers[:match_count]
        ]


def test_query_plan_detects_composite_legal_question() -> None:
    plan = create_query_plan(
        "What is the course load for a student under academic warning in the final semester?"
    )

    assert plan.requires_graph is True
    assert plan.graph_depth == 2
    assert {"academic_warning", "graduation_semester", "credit_hour_limit"} <= set(plan.entities)


def test_query_plan_extracts_english_and_arabic_article_numbers() -> None:
    plan = create_query_plan("Explain Article 18 together with المادة (٢٤).")

    assert plan.explicit_articles == ["18", "24"]


def test_rank_fusion_rewards_consensus_across_retrievers() -> None:
    shared = hit("shared", "18", vector_score=0.8)
    vector_only = hit("vector-only", "19", vector_score=0.9)
    keyword_shared = replace(shared, vector_score=0.0, keyword_score=0.7)

    results = reciprocal_rank_fusion(
        {
            "vector": [vector_only, shared],
            "keyword": [keyword_shared],
            "section": [],
            "graph": [],
        }
    )

    assert results[0].chunk_id == "shared"
    assert results[0].vector_score == 0.8
    assert results[0].keyword_score == 0.7


def test_exact_article_receives_legal_boost() -> None:
    plan = create_query_plan("Explain Article 18.")
    article_18 = hit("article-18", "18")
    article_19 = hit("article-19", "19")
    article_18.fusion_score = article_19.fusion_score = 1.0

    results = apply_legal_boosts([article_19, article_18], plan)

    assert results[0].article_number == "18"
    assert "Exact article requested in the question" in results[0].retrieval_reasons


def test_local_graph_expands_references_and_adjacent_articles() -> None:
    seed = hit("seed", "18", vector_score=1.0)
    seed.chunk_text = "According to Article 12, this exception applies."
    plan = create_query_plan("What exception applies in the final semester?")

    results = expand_local_graph(FakeArticleRepository(), [seed], plan, match_count=10)
    by_article = {result.article_number: result for result in results}

    assert by_article["12"].relation_path == ["references"]
    assert by_article["17"].relation_path == ["previous_article"]
    assert by_article["19"].relation_path == ["next_article"]


def test_reference_extraction_deduplicates_article_numbers() -> None:
    assert extract_article_references("Article 12 refers again to Article (12).") == ["12"]


def test_context_explains_graph_retrieval() -> None:
    related = hit("related", "12")
    related.retrieval_reasons = ["Article 18 references Article 12"]
    related.graph_path = ["18", "12"]
    related.relation_path = ["references"]

    context = format_evidence(build_evidence([related], 1))

    assert "Retrieved because: Article 18 references Article 12" in context
    assert "Graph path: 18 -> references -> 12" in context
