from __future__ import annotations

from src.models import EvidenceSource, RetrievalHit


def build_evidence(hits: list[RetrievalHit], limit: int) -> list[EvidenceSource]:
    return [
        EvidenceSource(source_id=f"S{index}", hit=hit)
        for index, hit in enumerate(hits[:limit], start=1)
    ]


def format_evidence(sources: list[EvidenceSource]) -> str:
    blocks: list[str] = []
    for source in sources:
        hit = source.hit
        metadata = [
            f"SOURCE {source.source_id}",
            f"Document: {hit.document_title}",
        ]
        if hit.section_title:
            metadata.append(f"Section: {hit.section_title}")
        if hit.article_number:
            metadata.append(f"Article: {hit.article_number}")
        page_label = str(hit.pdf_page_start)
        if hit.pdf_page_end != hit.pdf_page_start:
            page_label += f"-{hit.pdf_page_end}"
        metadata.append(f"PDF page: {page_label}")
        if hit.published_date:
            metadata.append(f"Published: {hit.published_date}")
        if hit.effective_date:
            metadata.append(f"Effective: {hit.effective_date}")
        if hit.retrieval_reasons:
            metadata.append("Retrieved because: " + "; ".join(hit.retrieval_reasons))
        if hit.graph_path and hit.relation_path:
            path_parts = [hit.graph_path[0]]
            for relation, node in zip(hit.relation_path, hit.graph_path[1:], strict=False):
                path_parts.extend((relation, node))
            metadata.append("Graph path: " + " -> ".join(path_parts))
        metadata.append(f"Official URL: {hit.source_url}")
        metadata.append("Text:\n" + hit.chunk_text)
        blocks.append("\n".join(metadata))
    return "\n\n---\n\n".join(blocks)
