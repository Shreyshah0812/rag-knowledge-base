"""
All tunable parameters live here, sourced from environment variables.
This is deliberate: every knob mentioned in BLUEPRINT.md (chunk size, top-k values,
rerank threshold, model choices) should be changeable without touching code, so the
eval loop (app/eval/run_eval.py) can be used to sweep them and produce comparable,
reproducible results.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # Postgres
    database_url: str = "postgresql://rag:rag@localhost:5432/rag_kb"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5
    rerank_threshold: float = 0.3

    # Generation
    generation_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    generation_temperature: float = 0.0
    enable_self_check: bool = True

    # Misc
    log_level: str = "INFO"
    raw_file_storage_dir: str = "/data/uploads"


settings = Settings()
