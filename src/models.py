from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class SourceDefinition(BaseModel):
    title: str
    url: HttpUrl
    department: str | None = None
    language: str = "en"
    document_type: str = "regulation"
    academic_year: str | None = None
    published_date: date | None = None
    effective_date: date | None = None
    version: str | None = None
    status: str = "current"


@dataclass(slots=True)
class ExtractedPage:
    pdf_page_number: int
    page_text: str
    printed_page_number: str | None = None
    section_title: str | None = None


@dataclass(slots=True)
class DetectedSection:
    local_key: str
    title: str
    section_type: str
    hierarchy_level: int
    page_start: int
    page_end: int
    article_number: str | None = None
    parent_local_key: str | None = None


@dataclass(slots=True)
class ArticlePiece:
    page_number: int
    text: str


@dataclass(slots=True)
class ArticleBlock:
    local_section_key: str | None
    article_number: str | None
    section_title: str | None
    pieces: list[ArticlePiece] = field(default_factory=list)


@dataclass(slots=True)
class PreparedChunk:
    chunk_index: int
    chunk_text: str
    embedding_text: str
    pdf_page_start: int
    pdf_page_end: int
    printed_page_start: str | None
    printed_page_end: str | None
    section_title: str | None
    article_number: str | None
    section_local_key: str | None
    language: str
    academic_year: str | None
    document_status: str
    token_count: int
    embedding: list[float] | None = None


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    document_title: str
    source_url: str
    chunk_text: str
    section_title: str | None
    article_number: str | None
    pdf_page_start: int
    pdf_page_end: int
    section_id: str | None = None
    printed_page_start: str | None = None
    printed_page_end: str | None = None
    published_date: str | None = None
    effective_date: str | None = None
    document_status: str | None = None
    language: str | None = None
    vector_score: float = 0.0
    keyword_score: float = 0.0
    section_score: float = 0.0
    graph_score: float = 0.0
    freshness_score: float = 0.0
    fusion_score: float = 0.0
    graph_distance: int | None = None
    graph_path: list[str] = field(default_factory=list)
    relation_path: list[str] = field(default_factory=list)
    retrieval_reasons: list[str] = field(default_factory=list)
    retrieval_sources: list[str] = field(default_factory=list)
    final_score: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        score_kind: Literal["vector", "keyword", "section", "graph"],
    ) -> "RetrievalHit":
        raw_score = float(value.get("score") or value.get("similarity") or 0.0)
        score_values = {
            "vector_score": raw_score if score_kind == "vector" else 0.0,
            "keyword_score": raw_score if score_kind == "keyword" else 0.0,
            "section_score": raw_score if score_kind == "section" else 0.0,
            "graph_score": raw_score if score_kind == "graph" else 0.0,
        }
        return cls(
            chunk_id=str(value.get("chunk_id") or value.get("id") or ""),
            document_id=str(value.get("document_id") or ""),
            document_title=str(value.get("document_title") or value.get("title") or "Untitled document"),
            source_url=str(value.get("source_url") or ""),
            chunk_text=str(value.get("chunk_text") or ""),
            section_title=value.get("section_title"),
            article_number=value.get("article_number"),
            pdf_page_start=int(value.get("pdf_page_start") or value.get("page_start") or 1),
            pdf_page_end=int(value.get("pdf_page_end") or value.get("page_end") or value.get("pdf_page_start") or 1),
            section_id=str(value["section_id"]) if value.get("section_id") else None,
            printed_page_start=value.get("printed_page_start"),
            printed_page_end=value.get("printed_page_end"),
            published_date=str(value["published_date"]) if value.get("published_date") else None,
            effective_date=str(value["effective_date"]) if value.get("effective_date") else None,
            document_status=value.get("document_status"),
            language=value.get("language"),
            **score_values,
        )


@dataclass(slots=True)
class EvidenceSource:
    source_id: str
    hit: RetrievalHit


@dataclass(slots=True)
class AssistantAnswer:
    text: str
    sources: list[EvidenceSource]


class EvaluationQuestion(BaseModel):
    question: str
    expected_document: str
    expected_article: str | None = None
    expected_page: int | None = Field(default=None, ge=1)
