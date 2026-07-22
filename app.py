from __future__ import annotations

import hashlib
import uuid

import streamlit as st

from src.config import Settings, get_settings
from src.database.repositories import SupabaseRepository
from src.database.supabase_client import create_supabase_client
from src.generation.answer_generator import OpenAIGenerationProvider
from src.generation.citation_builder import source_page_url, source_summary
from src.generation.service import QuestionAnsweringService
from src.ingestion.embedding_service import OpenAIEmbeddingProvider
from src.retrieval.service import HybridRetriever


st.set_page_config(
    page_title="GJU Regulations Assistant",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource
def build_service(settings: Settings) -> QuestionAnsweringService:
    repository = SupabaseRepository(create_supabase_client(settings))
    embedding_provider = OpenAIEmbeddingProvider(settings)
    retriever = HybridRetriever(repository, embedding_provider, settings)
    generator = OpenAIGenerationProvider(settings)
    return QuestionAnsweringService(retriever, generator)


def visitor_identifier() -> str:
    if "visitor_id" not in st.session_state:
        st.session_state.visitor_id = str(uuid.uuid4())
    return hashlib.sha256(st.session_state.visitor_id.encode("utf-8")).hexdigest()


def main() -> None:
    st.title("GJU Student Regulations Assistant")
    st.caption("Answers grounded in indexed official university documents")

    try:
        settings = get_settings()
    except Exception as error:
        st.error("The application configuration is invalid.")
        st.code(str(error))
        st.stop()

    configuration_errors = settings.public_configuration_errors()
    if configuration_errors:
        st.info(
            "Local setup is ready, but Supabase is not connected yet. Add the "
            "following values to `.env` after creating the database: "
            + ", ".join(configuration_errors)
            + "."
        )

    with st.sidebar:
        st.header("Search options")
        language_label = st.selectbox(
            "Question language",
            ("Auto detect", "English", "Arabic", "All languages"),
        )
        category_label = st.selectbox(
            "Document category",
            ("All regulations", "General regulations", "German Year"),
        )
        st.divider()
        if settings.experimental_graphrag:
            st.caption("Local GraphRAG experiment: enabled")
        st.caption(
            "This assistant explains indexed regulations. The official document "
            "and the responsible GJU office remain authoritative."
        )

    question = st.text_area(
        "Ask a question",
        placeholder="Can I register extra credit hours during my final semester?",
        max_chars=settings.max_question_length,
        height=110,
    )
    ask = st.button("Ask", type="primary", use_container_width=True)

    if not ask:
        st.subheader("What the answer will include")
        st.write("A grounded explanation, document title, article or section, PDF page, and official link.")
        return
    if not question.strip():
        st.warning("Please enter a question.")
        return
    if configuration_errors:
        st.warning("Connect Supabase before asking questions.")
        return

    language_map = {
        "Auto detect": "auto",
        "English": "en",
        "Arabic": "ar",
        "All languages": None,
    }
    category_map = {
        "All regulations": None,
        "General regulations": "regulation",
        "German Year": "german_year",
    }

    try:
        with st.spinner("Searching the official regulations…"):
            result = build_service(settings).answer(
                question.strip(),
                document_type=category_map[category_label],
                language=language_map[language_label],
                safety_identifier=visitor_identifier(),
            )
    except Exception as error:
        st.error("The question could not be processed. Check the API and Supabase configuration.")
        with st.expander("Technical details"):
            st.code(str(error))
        return

    st.subheader("Answer")
    st.markdown(result.text)

    st.subheader("Official sources")
    if not result.sources:
        st.caption("No relevant indexed source was found.")
    for source in result.sources:
        with st.container(border=True):
            st.markdown(f"**{source_summary(source)}**")
            if source.hit.published_date:
                st.caption(f"Published: {source.hit.published_date}")
            if source.hit.source_url:
                st.link_button(
                    f"Open PDF at page {source.hit.pdf_page_start}",
                    source_page_url(source),
                )

    with st.expander("Show retrieved evidence"):
        for source in result.sources:
            st.markdown(f"**{source.source_id} — {source.hit.document_title}**")
            if source.hit.retrieval_reasons:
                st.caption("Retrieved because: " + "; ".join(source.hit.retrieval_reasons))
            st.write(source.hit.chunk_text)
            st.divider()


if __name__ == "__main__":
    main()
