"""
Configuration management for the Text-to-SQL Agent.
Loads environment variables and provides typed configuration access.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Groq Configuration
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model_reasoning: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL_REASONING")
    groq_model_fast: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL_FAST")
    groq_temperature: float = Field(default=0.0, env="GROQ_TEMPERATURE")
    
    # Database Configuration
    database_uri: str = Field(..., env="DATABASE_URI")
    
    # Vector Store Configuration (ChromaDB)
    vector_store_path: str = Field(default="./data/vector_store", env="VECTOR_STORE_PATH")
    chroma_collection_name: str = Field(default="sql_examples", env="CHROMA_COLLECTION_NAME")
    
    # Embedding Configuration (using sentence-transformers for local embeddings)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Caching Configuration (Disk Cache)
    # Off by default — enable at runtime via the UI toggle, --cache CLI flag,
    # or by setting ENABLE_SEMANTIC_CACHE=true in .env
    enable_semantic_cache: bool = Field(default=False, env="ENABLE_SEMANTIC_CACHE")
    cache_similarity_threshold: float = Field(default=0.95, env="CACHE_SIMILARITY_THRESHOLD")
    
    # Agent Configuration
    max_iterations: int = Field(default=3, env="MAX_ITERATIONS")
    enable_self_correction: bool = Field(default=True, env="ENABLE_SELF_CORRECTION")
    enable_dynamic_few_shot: bool = Field(default=True, env="ENABLE_DYNAMIC_FEW_SHOT")
    few_shot_examples_count: int = Field(default=3, env="FEW_SHOT_EXAMPLES_COUNT")
    query_timeout_seconds: int = Field(default=30, env="QUERY_TIMEOUT_SECONDS")

    # Answer Verification (feedback-loop quality gate)
    enable_answer_verification: bool = Field(default=True, env="ENABLE_ANSWER_VERIFICATION")
    max_answer_verify_attempts: int  = Field(default=2,    env="MAX_ANSWER_VERIFY_ATTEMPTS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env for backwards compatibility


# Global settings instance
settings = Settings()
