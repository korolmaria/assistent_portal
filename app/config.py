"""Конфигурация проекта. Читает .env и валидирует настройки."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Все настройки проекта в одном месте."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # === vLLM endpoints ===
    vllm_vl_url: str = "http://localhost:8000/v1"
    vllm_text_url: str = "http://localhost:8010/v1"

    # === Triton ===
    triton_url: str = "localhost:8001"
    triton_bge_model: str = "bge_m3"
    triton_reranker_model: str = "qwen3_reranker"

    # === Qdrant ===
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # === Neo4j ===
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme123"

    # === Models ===
    vl_model_name: str = "qwen-vl"
    text_model_name: str = "qwen-text"

    # === Embeddings ===
    embedding_dim: int = 1024

    # === SupplierPortal / AlwaysOn API ===
    supplier_portal_url: str = "https://172.29.5.40:8443"
    supplier_portal_service_number: str = ""
    supplier_portal_verify_ssl: bool = False
    supplier_portal_timeout: int = 30

    # === HF ===
    hf_token: str = ""

    # === Project paths ===
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    logs_dir: str = "logs"


settings = Settings()
