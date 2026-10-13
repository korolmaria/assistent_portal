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
    # === SupplierPortal / AlwaysOn API ===
    supplier_portal_env: str = "dev"   # dev | prod

    # Dev
    supplier_portal_url_dev: str = "https://rest.eurotorg.by/10195"
    supplier_portal_service_number_dev: str = ""

    # Prod
    supplier_portal_url_prod: str = "https://rest.eurotorg.by/10196"
    supplier_portal_service_number_prod: str = ""

    # Общие
    supplier_portal_verify_ssl: bool = False
    supplier_portal_timeout: int = 30
    # === HF ===
    hf_token: str = ""

    # === Project paths ===
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    logs_dir: str = "logs"

    # === Производные (вычисляются) ===
    @property
    def supplier_portal_url(self) -> str:
        """URL SupplierPortal в зависимости от режима."""
        if self.supplier_portal_env == "prod":
            return self.supplier_portal_url_prod
        return self.supplier_portal_url_dev

    @property
    def supplier_portal_service_number(self) -> str:
        """ServiceNumber в зависимости от режима."""
        if self.supplier_portal_env == "prod":
            return self.supplier_portal_service_number_prod
        return self.supplier_portal_service_number_dev

settings = Settings()
