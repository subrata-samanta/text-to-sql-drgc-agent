"""
Configuration management for the Text-to-SQL Agent.
Loads environment variables and provides typed configuration access.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # "groq"  → Groq cloud LLMs + local SQLite + HuggingFace embeddings
    # "dbrx"  → Databricks DBRX LLMs + Databricks Delta tables + Databricks embeddings
    llm_provider: Literal["groq", "dbrx"] = Field(default="groq", env="LLM_PROVIDER")

    # ── Groq Configuration ────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model_reasoning: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL_REASONING")
    groq_model_fast: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL_FAST")
    groq_temperature: float = Field(default=0.0, env="GROQ_TEMPERATURE")

    # ── Databricks Configuration ──────────────────────────────────────────────
    databricks_host: Optional[str] = Field(default=None, env="DATABRICKS_HOST")
    # Personal Access Token (PAT) — starts with "dapi…"
    databricks_token: Optional[str] = Field(default=None, env="DATABRICKS_TOKEN")
    # SQL Warehouse HTTP path — found in SQL Warehouse → Connection Details
    databricks_http_path: Optional[str] = Field(default=None, env="DATABRICKS_HTTP_PATH")
    # Unity Catalog location of the nielsen data
    databricks_catalog: Optional[str] = Field(default="hive_metastore", env="DATABRICKS_CATALOG")
    databricks_schema: Optional[str]  = Field(default="default",        env="DATABRICKS_SCHEMA")
    # Models served as Databricks Model Serving endpoints
    databricks_model_reasoning: str = Field(
        default="databricks-dbrx-instruct", env="DATABRICKS_MODEL_REASONING"
    )
    databricks_model_fast: str = Field(
        default="databricks-meta-llama-3-1-70b-instruct", env="DATABRICKS_MODEL_FAST"
    )
    # Embedding endpoint (Foundation Model or custom)
    databricks_embedding_endpoint: str = Field(
        default="databricks-bge-large-en", env="DATABRICKS_EMBEDDING_ENDPOINT"
    )

    # ── Database Configuration (Groq/local path) ──────────────────────────────
    database_uri: str = Field(default="sqlite:///./data/database.db", env="DATABASE_URI")

    # ── Vector Store Configuration (ChromaDB) ────────────────────────────────
    vector_store_path: str = Field(default="./data/vector_store", env="VECTOR_STORE_PATH")
    chroma_collection_name: str = Field(default="sql_examples", env="CHROMA_COLLECTION_NAME")

    # ── Embedding Configuration ────────────────────────────────────────────────
    # HuggingFace model used when llm_provider == "groq"
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # ── Caching Configuration (Disk Cache) ────────────────────────────────────
    enable_semantic_cache: bool = Field(default=False, env="ENABLE_SEMANTIC_CACHE")
    cache_similarity_threshold: float = Field(default=0.95, env="CACHE_SIMILARITY_THRESHOLD")

    # ── Agent Configuration ────────────────────────────────────────────────────
    max_iterations: int = Field(default=3, env="MAX_ITERATIONS")
    enable_self_correction: bool = Field(default=True, env="ENABLE_SELF_CORRECTION")
    enable_dynamic_few_shot: bool = Field(default=True, env="ENABLE_DYNAMIC_FEW_SHOT")
    few_shot_examples_count: int = Field(default=3, env="FEW_SHOT_EXAMPLES_COUNT")
    # Smart category-aware retrieval: top-N examples fetched per detected category
    few_shot_per_category_count: int = Field(default=2, env="FEW_SHOT_PER_CATEGORY_COUNT")
    query_timeout_seconds: int = Field(default=30, env="QUERY_TIMEOUT_SECONDS")

    # ── Answer Verification (feedback-loop quality gate) ──────────────────────
    enable_answer_verification: bool = Field(default=True, env="ENABLE_ANSWER_VERIFICATION")
    max_answer_verify_attempts: int  = Field(default=2,    env="MAX_ANSWER_VERIFY_ATTEMPTS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    # ── Derived helpers ────────────────────────────────────────────────────────
    @property
    def is_dbrx(self) -> bool:
        return self.llm_provider == "dbrx"

    @property
    def active_model_reasoning(self) -> str:
        return self.databricks_model_reasoning if self.is_dbrx else self.groq_model_reasoning

    @property
    def active_model_fast(self) -> str:
        return self.databricks_model_fast if self.is_dbrx else self.groq_model_fast


# Global settings instance
settings = Settings()
