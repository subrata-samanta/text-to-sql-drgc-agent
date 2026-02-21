"""
Interaction Agent — the conversational front-door for every user message.

Responsibilities
----------------
1. Classify user intent into one of five categories:
   - data_query   : clear data/SQL question → full DRGC pipeline
   - follow_up    : data question that references a prior turn → DRGC with rewritten question
   - smalltalk    : greetings / meta-questions → direct response, no SQL
   - clarification: question too vague to answer → ask for more detail
   - out_of_scope : nothing to do with Nielsen POS → politely decline

2. For follow-up questions, rewrite the question to be fully self-contained
   (e.g., "what about OREO?" → "What was OREO market share in Total Bars in 2024?")

3. For smalltalk / clarification / out_of_scope, generate the reply directly
   so the rest of the DRGC pipeline is skipped entirely.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from core.state import AgentState
from config import settings

# ── Intent vocabulary ─────────────────────────────────────────────────────────
INTENT_DATA      = "data_query"    # Needs full SQL pipeline
INTENT_FOLLOWUP  = "follow_up"     # Data question referencing a prior turn
INTENT_CHAT      = "smalltalk"     # Greetings / pleasantries / meta-questions
INTENT_VAGUE     = "clarification" # Too vague — missing dimension / filter
INTENT_OOS       = "out_of_scope"  # Nothing to do with Nielsen POS

#: Intents that must be routed to the SQL pipeline
DATA_INTENTS: set[str] = {INTENT_DATA, INTENT_FOLLOWUP}

# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM = """\
You are the conversational router for a Nielsen POS data analytics assistant.

Your ONLY job: classify the user's message and decide how to handle it.

== Intent definitions ==

data_query   — A clear, answerable question about Nielsen POS data that requires SQL.
               (sales, revenue, market share, TDP, velocity, brand/category comparisons,
                time-period analysis, YTD, growth rates, distribution, etc.)

follow_up    — A data question that refers to a previous turn.
               Signs: "same brand", "that market", "compare those two", "what about last year?",
               "break it down by customer", "and for OREO?", pronouns like "it" / "they" / "them".

smalltalk    — Greetings, thanks, pleasantries, or meta-questions about the assistant.
               Examples: "hi", "hello", "thanks", "what can you do?", "how does this work?",
               "nice work!", "can you help me?".

clarification — A data-related question that is too vague to answer without more info.
               Examples: "show me sales" (which brand? which market? which period?),
               "how is the category doing?" (no category named).

out_of_scope  — Questions unrelated to Nielsen / CPG / retail data.
               Examples: weather, general coding help, recipes, sports scores, etc.

== Output format ==
Respond ONLY with a valid JSON object — no markdown, no explanation:
{{
  "intent": "<data_query|follow_up|smalltalk|clarification|out_of_scope>",
  "confidence": <0.0–1.0>,
  "rewritten_question": "<if follow_up: self-contained rewrite; otherwise the original question>",
  "direct_response": "<for smalltalk/clarification/out_of_scope only — your reply to the user>"
}}

== Rules ==
- follow_up → rewrite the question fully self-contained using names from conversation history.
- smalltalk → be warm & brief; describe 2–3 things the agent can do if relevant.
- clarification → explain what's missing; give 1–2 concise example phrasings they could use.
- out_of_scope → politely redirect: "I'm specialised in Nielsen POS analytics. Try asking about …"
- data_query / follow_up → direct_response must be an empty string "".
- When in doubt between data_query and follow_up, choose follow_up if history is non-empty and
  the message references anything from prior turns."""

_USER = """\
Conversation history (most recent turns):
{history}

Current user message: {question}"""


class InteractionAgent:
    """
    Classifies user intent and routes to the correct workflow branch.
    Uses the fast model for minimal latency at the conversation entry point.
    """

    def __init__(self) -> None:
        self.llm = ChatGroq(
            model=settings.groq_model_fast,
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("user", _USER),
        ])
        self.chain = self.prompt | self.llm

    # ── helpers ───────────────────────────────────────────────────────────────

    def _history_text(self, history: Optional[List[Dict]]) -> str:
        if not history:
            return "None — this is the first message."
        lines: list[str] = []
        for h in history[-4:]:
            lines.append(f"User: {h['question']}")
            answer = (h.get("nl_response") or "")[:300]
            if answer:
                lines.append(f"Assistant: {answer}")
        return "\n".join(lines)

    # ── public API ────────────────────────────────────────────────────────────

    def classify(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> dict:
        """
        Classify user intent and return a routing decision dict.

        Returns
        -------
        dict with keys:
            intent            str  — one of the five intent constants
            confidence        float
            rewritten_question str  — self-contained question (differs for follow_up)
            direct_response   str  — pre-built reply for non-data intents (else "")
        """
        logger.info(f"INTERACTION: classifying — '{question[:80]}'")

        try:
            resp = self.chain.invoke({
                "question": question,
                "history": self._history_text(conversation_history),
            })
            raw = resp.content.strip()

            # Extract JSON even when the model wraps it in markdown code fences
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            payload: dict = json.loads(m.group() if m else raw)

            intent    = payload.get("intent", INTENT_DATA)
            rewritten = (payload.get("rewritten_question") or question).strip()
            direct    = payload.get("direct_response", "")
            conf      = float(payload.get("confidence", 0.9))

            logger.info(f"INTERACTION: intent={intent}  confidence={conf:.2f}")
            if intent == INTENT_FOLLOWUP:
                logger.info(f"INTERACTION: rewritten → '{rewritten}'")

            return {
                "intent": intent,
                "confidence": conf,
                "rewritten_question": rewritten,
                "direct_response": direct,
            }

        except Exception as exc:
            logger.warning(f"InteractionAgent error ({exc}) — defaulting to data_query")
            return {
                "intent": INTENT_DATA,
                "confidence": 0.5,
                "rewritten_question": question,
                "direct_response": "",
            }


# ── LangGraph node ────────────────────────────────────────────────────────────

def interaction_node(state: AgentState) -> dict:
    """
    LangGraph node: classify intent, rewrite follow-up questions, set routing fields.

    Injects into state:
        intent            — routing key used by should_route()
        direct_response   — ready-made reply for non-SQL paths
        question          — replaced with self-contained rewrite for follow-ups
    """
    agent   = InteractionAgent()
    history = state.get("conversation_history") or []
    result  = agent.classify(state["question"], history)

    updates: dict = {
        "intent":          result["intent"],
        "direct_response": result.get("direct_response", ""),
    }

    # Overwrite question with fully self-contained rewrite for follow-ups
    if result["intent"] == INTENT_FOLLOWUP:
        rewritten = result.get("rewritten_question", "").strip()
        if rewritten and rewritten != state["question"]:
            updates["question"] = rewritten

    return updates
