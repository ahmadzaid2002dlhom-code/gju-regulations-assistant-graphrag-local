from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from src.models import RetrievalHit
from src.retrieval.query_planner import QueryPlan


SOURCE_WEIGHTS = {
    "vector": 1.00,
    "keyword": 0.80,
    "section": 0.70,
    "graph": 0.90,
}


def reciprocal_rank_fusion(
    result_sets: dict[str, Sequence[RetrievalHit]],
    *,
    k: int = 60,
) -> list[RetrievalHit]:
    scores: dict[str, float] = defaultdict(float)
    hits_by_id: dict[str, RetrievalHit] = {}

    for source, hits in result_sets.items():
        weight = SOURCE_WEIGHTS.get(source, 1.0)
        for rank, hit in enumerate(hits, start=1):
            existing = hits_by_id.get(hit.chunk_id)
            if existing is None:
                existing = replace(
                    hit,
                    graph_path=list(hit.graph_path),
                    relation_path=list(hit.relation_path),
                    retrieval_reasons=list(hit.retrieval_reasons),
                    retrieval_sources=list(hit.retrieval_sources),
                )
                hits_by_id[hit.chunk_id] = existing
            else:
                existing.vector_score = max(existing.vector_score, hit.vector_score)
                existing.keyword_score = max(existing.keyword_score, hit.keyword_score)
                existing.section_score = max(existing.section_score, hit.section_score)
                existing.graph_score = max(existing.graph_score, hit.graph_score)
                if hit.graph_distance is not None:
                    existing.graph_distance = hit.graph_distance
                    existing.graph_path = list(hit.graph_path)
                    existing.relation_path = list(hit.relation_path)
                for reason in hit.retrieval_reasons:
                    if reason not in existing.retrieval_reasons:
                        existing.retrieval_reasons.append(reason)
            if source not in existing.retrieval_sources:
                existing.retrieval_sources.append(source)
            edge_weight = hit.graph_score if source == "graph" else 1.0
            scores[hit.chunk_id] += weight * edge_weight / (k + rank)

    maximum = max(scores.values(), default=1.0)
    for chunk_id, hit in hits_by_id.items():
        hit.fusion_score = scores[chunk_id] / maximum
        hit.final_score = hit.fusion_score

    return sorted(hits_by_id.values(), key=lambda item: item.final_score, reverse=True)


def apply_legal_boosts(hits: Sequence[RetrievalHit], plan: QueryPlan) -> list[RetrievalHit]:
    explicit_articles = {value.casefold() for value in plan.explicit_articles}
    results: list[RetrievalHit] = []

    for original in hits:
        hit = replace(
            original,
            graph_path=list(original.graph_path),
            relation_path=list(original.relation_path),
            retrieval_reasons=list(original.retrieval_reasons),
            retrieval_sources=list(original.retrieval_sources),
        )
        boost = 0.0
        if hit.article_number and hit.article_number.casefold() in explicit_articles:
            boost += 0.25
            hit.retrieval_reasons.append("Exact article requested in the question")
        if "exception_to" in hit.relation_path:
            boost += 0.18
        if "defines" in hit.relation_path:
            boost += 0.15
        if "references" in hit.relation_path and "references" in plan.relation_hints:
            boost += 0.08
        if hit.language and plan.language != "mixed" and hit.language == plan.language:
            boost += 0.08
        hit.final_score = hit.fusion_score + boost
        graph_only = set(hit.retrieval_sources) == {"graph"}
        explicit = "explicit_article" in hit.relation_path
        if graph_only and not explicit:
            hit.final_score *= 0.70
        results.append(hit)

    return sorted(results, key=lambda item: item.final_score, reverse=True)
