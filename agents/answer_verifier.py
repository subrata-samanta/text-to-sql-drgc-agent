"""
Answer Verifier Agent — feedback loop quality gate.

After the responder produces a natural-language answer, this agent uses an LLM
to check whether the answer actually addresses what the user asked.

Verdict taxonomy
────────────────
• "satisfied"           — answer fully addresses the question → pipeline ends
• "wrong_data"          — the *data* returned by SQL is incorrect / incomplete
                          → loop back to the generator so SQL is re-written
• "wrong_presentation"  — the data is correct but the answer is unclear / missing
                          key numbers → loop back to the responder only

A maximum number of verification attempts prevents infinite loops.
"""

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger

from core.state import AgentState
from core.llm_factory import get_llm
from config import settings

# ─── Prompt ──────────────────────────────────────────────────────────────────

_SYSTEM = """You are a rigorous QA evaluator for an AI analytics assistant.

Your job: decide whether the ANSWER fully and correctly responds to the QUESTION.

You must return a JSON object with exactly these keys:
  "verdict"   : one of "satisfied" | "wrong_data" | "wrong_presentation"
  "feedback"  : short, actionable description of what is missing or wrong
                (≤ 2 sentences). Empty string "" when verdict is "satisfied".
  "rationale" : one sentence explaining your reasoning.

Verdict definitions
───────────────────
satisfied          — The answer directly and completely addresses the question
                     with the correct numbers / categories. Minor wording issues
                     are acceptable.

wrong_data         — The answer contains incorrect or incomplete data, or the SQL
                     returned the wrong rows / aggregation so the numbers don't
                     match what was asked. Requires a different SQL query.

wrong_presentation — The underlying data appears correct (right numbers exist)
                     but the answer:
                       • ignores part of the question
                       • is too vague / generic
                       • fails to highlight the key insight the user wanted
                     Fixable by re-wording WITHOUT re-running SQL.

Rules
─────
- Base your verdict ONLY on whether the ANSWER addresses the QUESTION.
- Do NOT penalise for missing columns that were never asked for.
- If the answer says "no data found" but the question was clearly answerable,
  classify as wrong_data.
- If the answer is a helpful error/clarification that honestly reflects system
  limits, classify as satisfied.
- Return ONLY the raw JSON — no markdown fences, no additional text."""

_USER = """QUESTION:
{question}

ANSWER:
{nl_response}

Evaluate and return JSON:"""

# ─── Agent ────────────────────────────────────────────────────────────────────

MAX_ANSWER_VERIFY_ATTEMPTS = 2   # overridden by settings if present


class AnswerVerifierAgent:
    """LLM-based answer quality verifier."""

    def __init__(self):
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("user",   _USER),
        ])
        self._parser = JsonOutputParser()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _max_attempts(self) -> int:
        return getattr(settings, "max_answer_verify_attempts", MAX_ANSWER_VERIFY_ATTEMPTS)

    # ── core ──────────────────────────────────────────────────────────────────

    def verify(self, state: AgentState) -> dict:
        """
        Evaluate whether `nl_response` fully answers `question`.

        Returns a partial state dict:
          answer_satisfies_question : bool
          answer_verdict            : "satisfied" | "wrong_data" | "wrong_presentation"
          answer_feedback           : str   (actionable note for downstream agents)
          answer_verify_attempts    : int   (incremented each call)
        """
        question    = state.get("question", "")
        nl_response = state.get("nl_response", "")
        attempts    = (state.get("answer_verify_attempts") or 0) + 1

        logger.info(
            f"ANSWER VERIFIER: attempt {attempts}/{self._max_attempts()} | "
            f"question: {question[:80]}"
        )

        # ── guard: skip verification for non-data / error / clarification paths ──
        intent = state.get("intent", "data_query")
        if intent not in ("data_query", "follow_up", "correction"):
            logger.info("ANSWER VERIFIER: non-data intent — skipping, marking satisfied")
            return {
                "answer_satisfies_question": True,
                "answer_verdict":            "satisfied",
                "answer_feedback":           "",
                "answer_verify_attempts":    attempts,
            }

        if state.get("error"):
            logger.info("ANSWER VERIFIER: error state — skipping, marking satisfied")
            return {
                "answer_satisfies_question": True,
                "answer_verdict":            "satisfied",
                "answer_feedback":           "",
                "answer_verify_attempts":    attempts,
            }

        if not nl_response.strip():
            logger.warning("ANSWER VERIFIER: empty nl_response → wrong_presentation")
            return {
                "answer_satisfies_question": False,
                "answer_verdict":            "wrong_presentation",
                "answer_feedback":           "The answer is empty. Please generate a response.",
                "answer_verify_attempts":    attempts,
            }

        # ── LLM evaluation ───────────────────────────────────────────────────
        try:
            llm   = get_llm("fast")
            chain = self._prompt | llm | self._parser
            raw: dict = chain.invoke({
                "question":    question,
                "nl_response": nl_response,
            })

            verdict: str  = str(raw.get("verdict", "satisfied")).strip().lower()
            feedback: str = str(raw.get("feedback", "")).strip()
            rationale     = str(raw.get("rationale", "")).strip()

            # Normalise unknown verdicts to satisfied (fail-safe)
            if verdict not in ("satisfied", "wrong_data", "wrong_presentation"):
                logger.warning(
                    f"ANSWER VERIFIER: unexpected verdict '{verdict}' → treating as satisfied"
                )
                verdict = "satisfied"

            satisfied = verdict == "satisfied"
            logger.info(
                f"ANSWER VERIFIER: verdict={verdict} | "
                f"feedback={feedback!r} | rationale={rationale!r}"
            )

            return {
                "answer_satisfies_question": satisfied,
                "answer_verdict":            verdict,
                "answer_feedback":           feedback,
                "answer_verify_attempts":    attempts,
            }

        except Exception as exc:
            logger.error(f"ANSWER VERIFIER: LLM call failed ({exc}) — marking satisfied")
            return {
                "answer_satisfies_question": True,
                "answer_verdict":            "satisfied",
                "answer_feedback":           "",
                "answer_verify_attempts":    attempts,
            }


# ── LangGraph node ───────────────────────────────────────────────────────────

def answer_verifier_node(state: AgentState) -> dict:
    """LangGraph node wrapper for AnswerVerifierAgent."""
    return AnswerVerifierAgent().verify(state)
