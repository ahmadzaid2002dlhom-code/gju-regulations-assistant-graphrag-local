from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.retrieval.query_classifier import QueryClassification, classify_query


ARTICLE_PATTERNS = (
    re.compile(r"\barticles?\s*\(?([0-9]+[A-Za-z]?)\)?", re.IGNORECASE),
    re.compile(r"الماد(?:ة|ه)\s*\(?([0-9٠-٩]+[A-Za-z]?)\)?", re.IGNORECASE),
)

RELATION_TERMS: dict[str, tuple[str, ...]] = {
    "exception_to": (
        "unless",
        "except",
        "exception",
        "notwithstanding",
        "باستثناء",
        "إلا إذا",
        "مع مراعاة",
    ),
    "references": (
        "according to",
        "pursuant to",
        "subject to",
        "وفقاً للمادة",
        "وفقا للمادة",
    ),
    "defines": (
        "defined as",
        "definition",
        "means",
        "يقصد بـ",
        "تعريف",
    ),
    "applies_to": (
        "under the conditions",
        "in the case of",
        "final semester",
        "graduation semester",
        "في حال",
        "شروط",
        "الفصل الأخير",
    ),
}

ENTITY_TERMS: dict[str, tuple[str, ...]] = {
    "academic_warning": ("academic warning", "probation", "إنذار أكاديمي", "انذار أكاديمي"),
    "graduation_semester": ("final semester", "graduation semester", "فصل التخرج", "الفصل الأخير"),
    "credit_hour_limit": ("credit hours", "course load", "الساعات المعتمدة", "العبء الدراسي"),
    "student_status": ("regular student", "student status", "الطالب المنتظم", "حالة الطالب"),
}


@dataclass(frozen=True, slots=True)
class QueryPlan:
    classification: QueryClassification
    explicit_articles: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relation_hints: list[str] = field(default_factory=list)
    requires_graph: bool = False
    graph_depth: int = 1

    @property
    def language(self) -> str:
        return self.classification.language

    @property
    def suggested_document_type(self) -> str | None:
        return self.classification.suggested_document_type


def _normalize_arabic_digits(value: str) -> str:
    return value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def _find_matches(text: str, vocabulary: dict[str, tuple[str, ...]]) -> list[str]:
    lowered = text.casefold()
    return [
        name
        for name, terms in vocabulary.items()
        if any(term.casefold() in lowered for term in terms)
    ]


def create_query_plan(text: str) -> QueryPlan:
    classification = classify_query(text)
    explicit_articles: list[str] = []
    for pattern in ARTICLE_PATTERNS:
        for match in pattern.finditer(text):
            article = _normalize_arabic_digits(match.group(1)).upper()
            if article not in explicit_articles:
                explicit_articles.append(article)

    relation_hints = _find_matches(text, RELATION_TERMS)
    entities = _find_matches(text, ENTITY_TERMS)
    composite_question = len(entities) >= 2
    requires_graph = bool(explicit_articles or relation_hints or composite_question)
    graph_depth = 2 if len(explicit_articles) > 1 or composite_question else 1

    return QueryPlan(
        classification=classification,
        explicit_articles=explicit_articles,
        entities=entities,
        relation_hints=relation_hints,
        requires_graph=requires_graph,
        graph_depth=graph_depth,
    )
