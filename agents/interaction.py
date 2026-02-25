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

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from core.state import AgentState
from core.llm_factory import create_llm

# ── Intent vocabulary ─────────────────────────────────────────────────────────
INTENT_DATA          = "data_query"     # Needs full SQL pipeline
INTENT_FOLLOWUP      = "follow_up"      # Data question referencing a prior turn
INTENT_CORRECTION    = "correction"     # User says prev answer was wrong; needs regen
INTENT_CHAT          = "smalltalk"      # Greetings / pleasantries / meta-questions
INTENT_VAGUE         = "clarification"  # Too vague — missing dimension / filter
INTENT_OOS           = "out_of_scope"   # Nothing to do with Nielsen POS
INTENT_RESULT_LOOKUP = "result_lookup"  # Answerable from the previous result set — no new SQL

#: Intents that must be routed to the SQL pipeline
DATA_INTENTS: set[str] = {INTENT_DATA, INTENT_FOLLOWUP, INTENT_CORRECTION}

# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM = """\
You are the conversational router for a Nielsen POS data analytics assistant.

Your ONLY job: classify the user's message and decide how to handle it.

== Intent definitions ==

data_query   — A clear, answerable question about Nielsen POS data that requires SQL.
               (sales, revenue, market share, TDP, velocity, brand/category comparisons,
                time-period analysis, YTD, growth rates, distribution, etc.)

follow_up    — A data question that refers to a previous turn but CANNOT be answered from
               the previous result alone — it needs NEW or DIFFERENT data from the database.
               Signs: "same brand", "that market", "compare those two", "what about last year?",
               "break it down by customer", "and for OREO?", pronouns like "it" / "they" / "them".

result_lookup — The question can be answered DIRECTLY from the data already returned in the
               previous turn — no new SQL is needed.
               Signs: asking for the min/max/rank/top/bottom item in a list that was just shown,
               spotting a specific value in results, "which one was highest/lowest?",
               "what was the value for X?", "how many brands were there?", "sort those",
               "which exceeded 10%?", counting or filtering the rows already returned.
               ONLY use this when the prior turn has actual result data (result_preview present).

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
  "intent": "<data_query|follow_up|result_lookup|smalltalk|clarification|out_of_scope>",
  "confidence": <0.0–1.0>,
  "rewritten_question": "<if follow_up: self-contained rewrite; otherwise the original question>",
  "direct_response": "<for smalltalk/clarification/out_of_scope only — your reply to the user>"
}}

== Rules ==
- follow_up → rewrite the question fully self-contained using names from conversation history.
- result_lookup → direct_response must be an empty string ""; the answer will be synthesised from the result data.
- smalltalk → be warm & brief; describe 2–3 things the agent can do if relevant.
- clarification → explain what's missing; give 1–2 concise example phrasings they could use.
- out_of_scope → politely redirect: "I'm specialised in Nielsen POS analytics. Try asking about …"
- data_query / follow_up → direct_response must be an empty string "".
- When in doubt between data_query and follow_up, choose follow_up if history is non-empty and
  the message references anything from prior turns.
- Prefer result_lookup over follow_up when the previous turn result_preview contains the data
  needed to answer directly."""

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
        self.llm = create_llm("fast")
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
            # Include a truncated result table so the classifier can decide
            # whether the next question is answerable from it.
            preview = (h.get("result_preview") or "").strip()
            if preview:
                # Cap at 800 chars to stay within prompt budget
                lines.append(f"[result data]\n{preview[:800]}")
        return "\n".join(lines)

    def answer_from_result(
        self,
        question: str,
        result_preview: str,
        nl_response: str,
    ) -> str:
        """Use the LLM to answer `question` from the prior result table."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Nielsen POS analytics assistant. "
             "Answer the user's question using ONLY the data table shown below. "
             "Be concise and precise. Do not run new queries or make up data."),
            ("user",
             "Previous assistant answer:\n{nl_response}\n\n"
             "Data returned by the previous query:\n{preview}\n\n"
             "User follow-up question: {question}\n\n"
             "Answer directly from the data above:"),
        ])
        chain = prompt | self.llm
        resp = chain.invoke({
            "nl_response": (nl_response or "")[:400],
            "preview":     result_preview[:2000],
            "question":    question,
        })
        return resp.content.strip()

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
            feedback  = payload.get("feedback_summary", "").strip()
            conf      = float(payload.get("confidence", 0.9))

            logger.info(f"INTERACTION: intent={intent}  confidence={conf:.2f}")
            if intent == INTENT_FOLLOWUP:
                logger.info(f"INTERACTION: rewritten → '{rewritten}'")
            if intent == INTENT_CORRECTION:
                logger.info(f"INTERACTION: correction feedback → '{feedback}'")

            return {
                "intent": intent,
                "confidence": conf,
                "rewritten_question": rewritten,
                "feedback_summary": feedback,
                "direct_response": direct,
            }

        except Exception as exc:
            logger.warning(f"InteractionAgent error ({exc}) — defaulting to data_query")
            return {
                "intent": INTENT_DATA,
                "confidence": 0.5,
                "rewritten_question": question,
                "feedback_summary": "",
                "direct_response": "",
            }


# ── LangGraph node ────────────────────────────────────────────────────────────

def _build_filter_correction(
    question: str,
    history: List[Dict],
) -> Optional[dict]:
    """
    If the last history turn contains a pending_filter_clarification, the user
    is answering our clarification question.  Build a correction update directly
    — no LLM classification needed.

    Returns a state-update dict (same shape as interaction_node's return) or
    None if this is not a filter-clarification response.
    """
    if not history:
        return None
    last = history[-1]
    pending = last.get("filter_clarification")  # list of {column, sql_value, …}
    draft_sql = last.get("sql")
    if not pending or not draft_sql:
        return None

    # Build an explicit user_feedback string describing each filter that needs fixing.
    # The user's raw answer (question) is the authoritative source for the new values.
    col_list = ", ".join(
        f"'{p['column']}'='{p['sql_value']}'" for p in pending
    )
    feedback = (
        f"Filter clarification response — the user answered: \"{question}\". "
        f"The following filter(s) could not be automatically matched and need "
        f"to be corrected using the user's answer: {col_list}. "
        f"Replace each unresolved filter value in the SQL with the value the "
        f"user just specified. If the answer implies a specific column value, "
        f"use it; otherwise keep the closest match."
    )

    logger.info(
        f"INTERACTION: detected filter-clarification response "
        f"(columns: {col_list}) — shortcutting to correction intent"
    )

    return {
        "intent":          INTENT_CORRECTION,
        "direct_response": "",
        "user_feedback":   feedback,
        "previous_sql":    draft_sql,
    }


# ── Module-level singleton ────────────────────────────────────────────────────
# InteractionAgent is stateless beyond __init__; self.llm and self.chain are
# immutable after construction.  The cached LLM is shared safely across threads.
_interaction_agent = InteractionAgent()


def interaction_node(state: AgentState) -> dict:
    """
    LangGraph node: classify intent, rewrite follow-up questions, set routing fields.

    Injects into state:
        intent            — routing key used by should_route()
        direct_response   — ready-made reply for non-SQL paths
        question          — replaced with self-contained rewrite for follow-ups
        user_feedback     — correction summary (correction intent only)
        previous_sql      — SQL from the prior turn being corrected
    """
    agent   = _interaction_agent
    history = state.get("conversation_history") or []

    # ── Fast-path: answering a filter clarification ───────────────────────
    # If the previous turn asked "which brand did you mean?", the user's reply
    # is deterministically a correction — skip LLM classification entirely.
    fc_update = _build_filter_correction(state["question"], history)
    if fc_update is not None:
        return fc_update

    result  = agent.classify(state["question"], history)

    updates: dict = {
        "intent":          result["intent"],
        "direct_response": result.get("direct_response", ""),
        "user_feedback":   None,
        "previous_sql":    None,
    }

    # ── result_lookup: answer directly from previous result data ──────────
    if result["intent"] == INTENT_RESULT_LOOKUP:
        last = history[-1] if history else {}
        prior_preview  = (last.get("result_preview") or "").strip()
        prior_response = (last.get("nl_response") or "").strip()
        if prior_preview:
            logger.info("INTERACTION: result_lookup — answering from prior result data")
            direct = agent.answer_from_result(
                state["question"], prior_preview, prior_response
            )
        else:
            # No result data in history — fall back to data_query
            logger.warning("INTERACTION: result_lookup but no prior result_preview — falling back to data_query")
            updates["intent"] = INTENT_FOLLOWUP
            direct = ""
        updates["direct_response"] = direct
        return updates

    # Overwrite question with fully self-contained rewrite for follow-ups
    if result["intent"] == INTENT_FOLLOWUP:
        rewritten = result.get("rewritten_question", "").strip()
        if rewritten and rewritten != state["question"]:
            updates["question"] = rewritten

    # For corrections: capture feedback + the SQL that was wrong
    elif result["intent"] == INTENT_CORRECTION:
        raw_feedback = result.get("feedback_summary", "").strip()
        user_msg     = state["question"]
        # Combine LLM-extracted summary with the raw user message
        updates["user_feedback"] = (
            f"{raw_feedback} | User said: {user_msg}" if raw_feedback else user_msg
        )
        # Pull the most-recent SQL from conversation history
        for h in reversed(history):
            if h.get("sql"):
                updates["previous_sql"] = h["sql"]
                break
        logger.info(
            f"CORRECTION: feedback='{updates['user_feedback']}' "
            f"prev_sql={'yes' if updates['previous_sql'] else 'not found'}"
        )

    return updates
