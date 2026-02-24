"""
core/llm_factory.py — Provider-aware LLM factory.

Usage in any agent:
    from core.llm_factory import create_llm

    self.llm = create_llm("reasoning")   # heavier model — planning, generation
    self.llm = create_llm("fast")        # lighter model — retrieval, validation
    self.llm = create_llm("reasoning", streaming=True)  # for streaming responses

Provider routing (set LLM_PROVIDER in .env):
    "groq"  → ChatGroq  with groq_model_reasoning / groq_model_fast
    "dbrx"  → ChatOpenAI pointing at DBRX_BASE_URL  (Databricks Gemini 2.5 Flash)
              Both tiers use the same dbrx_model — Databricks has one endpoint.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from config import settings


def create_llm(tier: str = "reasoning", streaming: bool = False) -> BaseChatModel:
    """Return a configured chat LLM for the active provider.

    Args:
        tier:      "reasoning" → heavy/smart model | "fast" → light/cheap model.
                   For the dbrx provider both tiers map to the same model.
        streaming: When True, enable token streaming (used by the responder).

    Returns:
        A LangChain BaseChatModel instance ready to call.
    """
    provider = settings.llm_provider.lower().strip()

    if provider == "dbrx":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.dbrx_api_key,
            base_url=settings.dbrx_base_url,
            model=settings.dbrx_model,
            temperature=settings.dbrx_temperature,
            max_tokens=settings.dbrx_max_tokens,
            streaming=streaming,
        )

    # Default / "groq"
    from langchain_groq import ChatGroq
    model = (
        settings.groq_model_reasoning
        if tier == "reasoning"
        else settings.groq_model_fast
    )
    return ChatGroq(
        model=model,
        temperature=settings.groq_temperature,
        groq_api_key=settings.groq_api_key,
        streaming=streaming,
    )
