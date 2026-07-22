from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr
    openai_generation_model: str = "gpt-5.6-luna"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_ocr_model: str = "gpt-5.6-luna"
    embedding_dimensions: int = Field(default=768, ge=1)

    supabase_url: str = ""
    supabase_anon_key: SecretStr = SecretStr("")
    supabase_service_role_key: SecretStr = SecretStr("")

    max_question_length: int = Field(default=800, ge=50, le=5000)
    retrieval_candidates: int = Field(default=20, ge=3, le=100)
    final_evidence_chunks: int = Field(default=5, ge=1, le=10)
    max_answer_tokens: int = Field(default=800, ge=100, le=4000)
    experimental_graphrag: bool = True
    graph_seed_sections: int = Field(default=4, ge=1, le=20)
    graph_candidate_limit: int = Field(default=40, ge=1, le=100)

    @property
    def openai_key(self) -> str:
        return self.openai_api_key.get_secret_value()

    @property
    def anon_key(self) -> str:
        return self.supabase_anon_key.get_secret_value()

    @property
    def service_role_key(self) -> str:
        return self.supabase_service_role_key.get_secret_value()

    def public_configuration_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.openai_key:
            errors.append("OPENAI_API_KEY is missing")
        if not self.supabase_url:
            errors.append("SUPABASE_URL is missing")
        if not self.anon_key:
            errors.append("SUPABASE_ANON_KEY is missing")
        return errors

    def ingestion_configuration_errors(self) -> list[str]:
        errors = self.public_configuration_errors()
        if not self.service_role_key:
            errors.append("SUPABASE_SERVICE_ROLE_KEY is missing")
        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
