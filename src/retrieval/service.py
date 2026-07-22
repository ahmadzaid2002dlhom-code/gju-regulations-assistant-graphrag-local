from __future__ import annotations

from src.config import Settings
from src.database.repositories import SupabaseRepository
from src.ingestion.embedding_service import EmbeddingProvider
from src.models import RetrievalHit
from src.retrieval.fusion import apply_legal_boosts, reciprocal_rank_fusion
from src.retrieval.graph_search import expand_local_graph
from src.retrieval.keyword_search import keyword_search
from src.retrieval.page_index_search import page_index_search
from src.retrieval.query_classifier import QueryClassification
from src.retrieval.query_planner import create_query_plan
from src.retrieval.reranker import merge_and_rerank, select_diverse_hits
from src.retrieval.vector_search import vector_search


class HybridRetriever:
    def __init__(
        self,
        repository: SupabaseRepository,
        embedding_provider: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._settings = settings

    def retrieve(
        self,
        question: str,
        *,
        document_type: str | None = None,
        language: str | None = None,
    ) -> tuple[list[RetrievalHit], QueryClassification]:
        plan = create_query_plan(question)
        classification = plan.classification
        language_filter = language
        if language_filter == "auto":
            language_filter = None if classification.language == "mixed" else classification.language
        effective_type = document_type or classification.suggested_document_type
        query_embedding = self._embedding_provider.embed_query(question)
        candidate_count = self._settings.retrieval_candidates

        vector_hits = vector_search(
            self._repository,
            query_embedding,
            match_count=candidate_count,
            document_type=effective_type,
            language=language_filter,
        )
        keyword_hits = keyword_search(
            self._repository,
            question,
            match_count=candidate_count,
            document_type=effective_type,
            language=language_filter,
        )
        section_hits = page_index_search(
            self._repository,
            question,
            match_count=max(5, candidate_count // 2),
            document_type=effective_type,
            language=language_filter,
        )
        if not self._settings.experimental_graphrag:
            reranked = merge_and_rerank(vector_hits, keyword_hits, section_hits)
            return (
                select_diverse_hits(reranked, self._settings.final_evidence_chunks),
                classification,
            )

        result_sets = {
            "vector": vector_hits,
            "keyword": keyword_hits,
            "section": section_hits,
        }
        preliminary = reciprocal_rank_fusion(result_sets)
        graph_hits: list[RetrievalHit] = []
        if plan.requires_graph:
            graph_hits = expand_local_graph(
                self._repository,
                preliminary[: self._settings.graph_seed_sections],
                plan,
                match_count=self._settings.graph_candidate_limit,
            )
        reranked = apply_legal_boosts(
            reciprocal_rank_fusion({**result_sets, "graph": graph_hits}),
            plan,
        )
        return (
            select_diverse_hits(reranked, self._settings.final_evidence_chunks),
            classification,
        )
