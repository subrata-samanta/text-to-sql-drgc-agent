"""
Configuration management for the Text-to-SQL Agent.
Loads environment variables and provides typed configuration access.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Provider selection ────────────────────────────────────────────────────
    # "groq"  → Groq LLMs + local SQLite + HuggingFace embeddings (default)
    # "dbrx"  → Databricks Gemini model + Databricks SQL + Databricks embeddings
    llm_provider: str = Field(default="groq", env="LLM_PROVIDER")

    # ── Groq Configuration ────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model_reasoning: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL_REASONING")
    groq_model_fast: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL_FAST")
    groq_temperature: float = Field(default=0.0, env="GROQ_TEMPERATURE")

    # ── DBRX / Databricks — Gemini 2.5 Flash ─────────────────────────────────
    dbrx_api_key: str = Field(default="", env="DBRX_API_KEY")
    dbrx_base_url: str = Field(default="", env="DBRX_BASE_URL")
    dbrx_model: str = Field(default="databricks-gemini-2-5-flash", env="DBRX_MODEL")
    dbrx_temperature: float = Field(default=0.0, env="DBRX_TEMPERATURE")
    dbrx_max_tokens: int = Field(default=2048, env="DBRX_MAX_TOKENS")

    # ── Databricks SQL connection ─────────────────────────────────────────────
    databricks_server_hostname: str = Field(default="", env="DATABRICKS_SERVER_HOSTNAME")
    databricks_http_path: str = Field(default="", env="DATABRICKS_HTTP_PATH")
    databricks_access_token: str = Field(default="", env="DATABRICKS_ACCESS_TOKEN")
    dbx_catalog: str = Field(default="dev-amer-customer-catalog", env="DBX_CATALOG")
    dbx_schema_name: str = Field(default="dev-amer-analyt-arisegenai-schema", env="DBX_SCHEMA")
    dbx_table: str = Field(default="nielsen_market_ci_new_1", env="DBX_TABLE")

    @property
    def dbx_full_table(self) -> str:
        """Fully-qualified Databricks table name with backtick quoting."""
        return f"`{self.dbx_catalog}`.`{self.dbx_schema_name}`.`{self.dbx_table}`"

    # ── Database Configuration (Groq/SQLite path) ─────────────────────────────
    database_uri: str = Field(default="sqlite:///./data/database.db", env="DATABASE_URI")

    # ── Vector Store Configuration (ChromaDB) ─────────────────────────────────
    vector_store_path: str = Field(default="./data/vector_store", env="VECTOR_STORE_PATH")
    chroma_collection_name: str = Field(default="sql_examples", env="CHROMA_COLLECTION_NAME")

    # ── Embedding Configuration ──────────────────────────────────────────────
    # groq provider → HuggingFace local model (no API cost, runs offline)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    # dbrx provider → Databricks embedding model via OpenAI-compatible endpoint
    dbrx_embedding_model: str = Field(default="databricks-gte-large-en", env="DBRX_EMBEDDING_MODEL")
    # Full Databricks serving-endpoint invocations URL for embeddings, e.g.:
    # https://<workspace>.gcp.databricks.com/serving-endpoints/databricks-gte-large-en/invocations
    # base_url and model are derived automatically from this URL.
    dbrx_embedding_endpoint_url: str = Field(default="", env="DBRX_EMBEDDING_ENDPOINT_URL")

    # ── Caching Configuration (Disk Cache) ────────────────────────────────────
    # Off by default — enable at runtime via the UI toggle, --cache CLI flag,
    # or by setting ENABLE_SEMANTIC_CACHE=true in .env
    enable_semantic_cache: bool = Field(default=False, env="ENABLE_SEMANTIC_CACHE")
    cache_similarity_threshold: float = Field(default=0.95, env="CACHE_SIMILARITY_THRESHOLD")

    # ── Agent Configuration ────────────────────────────────────────────────────
    max_iterations: int = Field(default=3, env="MAX_ITERATIONS")
    enable_self_correction: bool = Field(default=True, env="ENABLE_SELF_CORRECTION")
    enable_dynamic_few_shot: bool = Field(default=True, env="ENABLE_DYNAMIC_FEW_SHOT")
    few_shot_examples_count: int = Field(default=3, env="FEW_SHOT_EXAMPLES_COUNT")
    query_timeout_seconds: int = Field(default=30, env="QUERY_TIMEOUT_SECONDS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env for backwards compatibility


# Global settings instance
settings = Settings()
