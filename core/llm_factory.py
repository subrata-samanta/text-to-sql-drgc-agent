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

Latency optimisation
────────────────────
LLM client objects are stateless after construction (all state lives in the API
call, not the object). We therefore cache one instance per (provider, tier,
streaming) combination and reuse it across every request.  This avoids the
overhead of constructing a new HTTP client, loading credentials, and validating
settings on every agent instantiation.

The streaming=True variant is intentionally cached separately because LangChain
records the flag at construction time and uses it to choose the API response
format.  Concurrent `.stream()` calls on the same object are safe — each call
opens its own HTTP request.
"""

from __future__ import annotations

from typing import Dict, Tuple
from langchain_core.language_models import BaseChatModel
from config import settings

# ── Module-level LLM instance cache ──────────────────────────────────────────
# Key: (provider, tier, streaming) → singleton LLM object
_llm_cache: Dict[Tuple[str, str, bool], BaseChatModel] = {}


def create_llm(tier: str = "reasoning", streaming: bool = False) -> BaseChatModel:
    """Return a configured chat LLM for the active provider.

    Instances are cached per (provider, tier, streaming) tuple so the
    expensive HTTP-client + credential setup only happens once per combination.

    Args:
        tier:      "reasoning" → heavy/smart model | "fast" → light/cheap model.
                   For the dbrx provider both tiers map to the same model.
        streaming: When True, enable token streaming (used by the responder).

    Returns:
        A LangChain BaseChatModel instance ready to call.
    """
    provider = settings.llm_provider.lower().strip()
    cache_key: Tuple[str, str, bool] = (provider, tier, streaming)

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    if provider == "dbrx":
        from langchain_openai import ChatOpenAI
        # "reasoning" tier → Gemini 2.5 Pro  (SQL generation only)
        # "fast" tier      → Gemini 2.5 Flash (all other tasks)
        dbrx_model = (
            settings.dbrx_model_reasoning
            if tier == "reasoning"
            else settings.dbrx_model
        )
        llm: BaseChatModel = ChatOpenAI(
            api_key=settings.dbrx_api_key,
            base_url=settings.dbrx_base_url,
            model=dbrx_model,
            temperature=settings.dbrx_temperature,
            max_tokens=settings.dbrx_max_tokens,
            streaming=streaming,
        )
    else:
        # Default / "groq"
        from langchain_groq import ChatGroq
        model = (
            settings.groq_model_reasoning
            if tier == "reasoning"
            else settings.groq_model_fast
        )
        llm = ChatGroq(
            model=model,
            temperature=settings.groq_temperature,
            groq_api_key=settings.groq_api_key,
            streaming=streaming,
        )

    _llm_cache[cache_key] = llm
    return llm
