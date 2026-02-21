"""
NL Responder Agent: converts SQL execution results into natural language answers.
Supports both sync (for graph node caching) and token-streaming (for UI).
"""

from typing import Generator, List, Dict, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from core.state import AgentState
from config import settings


_SYSTEM = """You are a concise, business-oriented data analyst who answers questions \
based on SQL query results from a Nielsen POS dataset.

Rules:
- Answer ONLY from the numbers in the provided SQL result — never make up data
- Use business-friendly language (no SQL jargon)
- Format numbers clearly: dollar amounts as $X.XXM / $X.XXK, percentages as XX.X%
- If the result is empty → "No data found for this query"
- 1–3 sentences for simple queries; a short paragraph for multi-row / comparative results
- For multi-row results: highlight the top finding first, then briefly summarize the rest
- When the question is a follow-up, use the conversation history for context but answer \
  the current question specifically

Conversation history (most recent turns, for follow-up context):
{conversation_history}"""

_USER = """Question: {question}

SQL result:
{result_preview}

Answer:"""


class NLResponderAgent:
    """Generates natural language answers from SQL results."""

    def __init__(self):
        self._base_llm_kwargs = dict(
            model=settings.groq_model_reasoning,
            temperature=0.1,
            groq_api_key=settings.groq_api_key,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("user", _USER),
        ])

    # ── helpers ──────────────────────────────────────────────────────────────

    def _history_text(self, history: Optional[List[Dict]]) -> str:
        if not history:
            return "None"
        return "\n\n".join(
            f"Q: {h['question']}\nA: {h['nl_response']}"
            for h in history[-4:]          # last 4 turns
        )

    def _build_input(self, state: AgentState, history: Optional[List[Dict]]) -> dict:
        return {
            "question": state["question"],
            "result_preview": state.get("result_preview") or "No results returned.",
            "conversation_history": self._history_text(history),
        }

    # ── sync respond (used as LangGraph node) ────────────────────────────────

    def respond(self, state: AgentState,
                conversation_history: Optional[List[Dict]] = None) -> dict:
        """Sync call — stores nl_response in graph state."""
        logger.info("RESPONDER: Generating natural language answer")

        if state.get("error"):
            msg = f"I was unable to answer your question: {state['error']}"
            return {"nl_response": msg}

        try:
            llm = ChatGroq(**self._base_llm_kwargs)
            chain = self.prompt | llm
            # Use conversation_history from state if not passed explicitly
            history = conversation_history or state.get("conversation_history")
            resp = chain.invoke(self._build_input(state, history))
            return {"nl_response": resp.content.strip()}
        except Exception as e:
            logger.error(f"Responder error: {e}")
            fallback = state.get("result_preview") or "Query executed successfully."
            return {"nl_response": fallback}

    # ── streaming respond (used by UI / CLI) ─────────────────────────────────

    def stream_respond(
        self,
        state: AgentState,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        """Token-level streaming generator for the NL answer."""

        if state.get("error"):
            yield f"I was unable to complete the query: {state['error']}"
            return

        if not state.get("result_preview"):
            yield "The query returned no results."
            return

        try:
            streaming_llm = ChatGroq(**self._base_llm_kwargs, streaming=True)
            chain = self.prompt | streaming_llm
            history = conversation_history or state.get("conversation_history")
            for chunk in chain.stream(self._build_input(state, history)):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Responder streaming error: {e}")
            yield state.get("result_preview") or "Query executed successfully."


# ── LangGraph node ───────────────────────────────────────────────────────────

def responder_node(state: AgentState) -> dict:
    """LangGraph node wrapper for NLResponderAgent (non-streaming, for caching)."""
    agent = NLResponderAgent()
    return agent.respond(state)
