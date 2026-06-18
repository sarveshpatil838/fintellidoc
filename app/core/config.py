"""
Application configuration via environment variables.
All settings are validated at startup — if ANTHROPIC_API_KEY is missing,
the app fails fast with a clear error rather than silently degrading.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    max_retries: int = 3

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/fintellidoc"

    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_chunks: int = 5
    faiss_index_path: str = "./data/faiss_index"

    # API
    api_prefix: str = "/api/v1"
    debug: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
