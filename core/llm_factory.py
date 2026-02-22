"""
LLM Factory — central provider abstraction.

Returns the right LangChain chat model based on settings.llm_provider:
  "groq"  → ChatGroq   (Groq cloud API)
  "dbrx"  → ChatDatabricks (Databricks Model Serving endpoint)

Usage
─────
    from core.llm_factory import get_llm

    llm          = get_llm("reasoning")           # sync, non-streaming
    llm_fast     = get_llm("fast")                # sync, non-streaming
    llm_stream   = get_llm("reasoning", streaming=True)  # for token streaming
"""

from typing import Literal
from loguru import logger
from config import settings


RoleType = Literal["reasoning", "fast"]


def get_llm(role: RoleType = "reasoning", streaming: bool = False):
    """
    Return the appropriate LangChain chat model for the current provider.

    Parameters
    ----------
    role      : "reasoning" → most capable model
                "fast"      → lighter/cheaper model for quick classifications
    streaming : whether to enable token-level streaming output

    Returns
    -------
    A LangChain BaseChatModel instance ready to use in chains.
    """
    provider = settings.llm_provider

    # ── Groq ──────────────────────────────────────────────────────────────────
    if provider == "groq":
        from langchain_groq import ChatGroq

        model = (
            settings.groq_model_reasoning
            if role == "reasoning"
            else settings.groq_model_fast
        )
        logger.debug(f"LLM factory → Groq | model={model} | streaming={streaming}")
        return ChatGroq(
            model=model,
            temperature=settings.groq_temperature,
            groq_api_key=settings.groq_api_key,
            streaming=streaming,
        )

    # ── Databricks DBRX ───────────────────────────────────────────────────────
    if provider == "dbrx":
        try:
            from langchain_databricks import ChatDatabricks
        except ImportError:
            raise ImportError(
                "langchain-databricks is not installed. "
                "Run: pip install langchain-databricks"
            )

        endpoint = (
            settings.databricks_model_reasoning
            if role == "reasoning"
            else settings.databricks_model_fast
        )
        logger.debug(f"LLM factory → Databricks | endpoint={endpoint} | streaming={streaming}")
        return ChatDatabricks(
            endpoint=endpoint,
            temperature=settings.groq_temperature,
            streaming=streaming,
        )

    raise ValueError(
        f"Unknown llm_provider '{provider}'. Supported values: 'groq', 'dbrx'."
    )


def get_embedding_function():
    """
    Return the appropriate LangChain embedding function for the current provider.

    "groq"  → HuggingFaceEmbeddings (runs locally, no API key needed)
    "dbrx"  → DatabricksEmbeddings  (Databricks Foundation Model endpoint)
    """
    provider = settings.llm_provider

    if provider == "groq":
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.debug(f"Embedding factory → HuggingFace | model={settings.embedding_model}")
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    if provider == "dbrx":
        try:
            from langchain_databricks import DatabricksEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-databricks is not installed. "
                "Run: pip install langchain-databricks"
            )
        endpoint = settings.databricks_embedding_endpoint
        logger.debug(f"Embedding factory → Databricks | endpoint={endpoint}")
        return DatabricksEmbeddings(endpoint=endpoint)

    raise ValueError(f"Unknown llm_provider '{provider}'.")
