from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from src.models import RetrievalHit
from src.retrieval.query_planner import QueryPlan


REFERENCE_PATTERNS = (
    re.compile(r"\barticles?\s*\(?([0-9]+[A-Za-z]?)\)?", re.IGNORECASE),
    re.compile(r"الماد(?:ة|ه)\s*\(?([0-9٠-٩]+[A-Za-z]?)\)?", re.IGNORECASE),
)


class ArticleRepository(Protocol):
    def fetch_article_chunks(
        self,
        document_id: str,
        article_numbers: list[str],
        *,
        match_count: int,
    ) -> list[dict]: ...


def extract_article_references(text: str) -> list[str]:
    references: list[str] = []
    for pattern in REFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")).upper()
            if value not in references:
                references.append(value)
    return references


def _adjacent_articles(article_number: str | None) -> list[str]:
    if not article_number or not article_number.isdigit():
        return []
    number = int(article_number)
    values = [str(number + 1)]
    if number > 1:
        values.insert(0, str(number - 1))
    return values


def expand_local_graph(
    repository: ArticleRepository,
    seeds: Sequence[RetrievalHit],
    plan: QueryPlan,
    *,
    match_count: int = 40,
) -> list[RetrievalHit]:
    seeds_by_document: dict[str, list[RetrievalHit]] = defaultdict(list)
    for seed in seeds:
        seeds_by_document[seed.document_id].append(seed)

    expanded: list[RetrievalHit] = []
    remaining = match_count
    for document_id, document_seeds in seeds_by_document.items():
        if remaining <= 0:
            break

        relations: dict[str, tuple[str, str, float, str]] = {}

        def record_relation(
            article: str,
            relation: str,
            reason: str,
            score: float,
            source_label: str,
        ) -> None:
            key = article.casefold()
            current = relations.get(key)
            if current is None or score > current[2]:
                relations[key] = (relation, reason, score, source_label)

        for article in plan.explicit_articles:
            record_relation(
                article,
                "explicit_article",
                "Article requested explicitly",
                1.0,
                "question",
            )

        for seed in document_seeds:
            seed_label = seed.article_number or seed.section_title or "seed"
            seed_strength = max(0.20, min(seed.final_score, 1.0))
            for article in extract_article_references(seed.chunk_text):
                if article != seed.article_number:
                    record_relation(
                        article,
                        "references",
                        f"Article {seed_label} references Article {article}",
                        0.85 * seed_strength,
                        seed_label,
                    )
            if plan.requires_graph:
                for article in _adjacent_articles(seed.article_number):
                    relation = "previous_article" if int(article) < int(seed.article_number or 0) else "next_article"
                    record_relation(
                        article,
                        relation,
                        f"Article adjacent to Article {seed.article_number}",
                        0.45 * seed_strength,
                        seed_label,
                    )

        requested_articles = [value.upper() for value in relations]
        if not requested_articles:
            continue

        rows = repository.fetch_article_chunks(
            document_id,
            requested_articles,
            match_count=remaining,
        )
        seed_template = document_seeds[0]
        for row in rows:
            article = str(row.get("article_number") or "")
            relation, reason, score, source_label = relations.get(
                article.casefold(),
                ("same_topic", "Related article", 0.30, "seed"),
            )
            enriched = dict(row)
            enriched.update(
                {
                    "document_title": seed_template.document_title,
                    "source_url": seed_template.source_url,
                    "published_date": seed_template.published_date,
                    "effective_date": seed_template.effective_date,
                    "document_status": seed_template.document_status,
                    "language": row.get("language") or seed_template.language,
                    "score": score,
                }
            )
            hit = RetrievalHit.from_mapping(enriched, "graph")
            hit.graph_distance = 0 if relation == "explicit_article" else 1
            hit.graph_path = [
                source_label,
                article,
            ]
            hit.relation_path = [relation]
            hit.retrieval_reasons = [reason]
            expanded.append(hit)
            remaining -= 1
            if remaining <= 0:
                break

    return sorted(expanded, key=lambda item: item.graph_score, reverse=True)
