"""
LangGraph workflow orchestration for the Text-to-SQL agent.
Implements the DRGC (Decomposition-Retrieval-Generation-Correction) framework.

Design principles
─────────────────
• No semantic cache — every question runs the full pipeline, always fresh.
• No redundant fetching — schema + few-shot are retrieved ONCE per turn in a
  single parallel node (context_builder); downstream retry loops reuse what is
  already in state.
• Validation feedback is injected into the generator on every retry, so the
  model knows exactly what it got wrong rather than regenerating blindly.
• Correction intent is first-class — user_feedback + previous_sql flow all the
  way to the generator so corrections target the actual mistake.
• Reflector (execution-error fixer) loops back through post_reflect_filter so
  that a structurally-fixed SQL is also re-checked for filter value correctness
  before hitting the DB again.

Graph topology
──────────────
  init
   │
   ▼
  interaction ──[non-data / result_lookup]──► direct_respond ──► END
   │
   │[data: data_query / follow_up / correction]
   ▼
  planner
   │
   ▼
  context_builder          ← schema-linking + few-shot retrieval in ONE parallel node
   │
   ▼
  generator  ◄────────────────────────────────────────────────────────────────┐
   │                                                              [regenerate: │
   ▼                                                         validation_issues │
  filter_resolver                                                 injected]    │
   │[needs_clarification]──► direct_respond ──► END                           │
   │[ok]                                                                       │
   ▼                                                                           │
  sql_validator ──[fail, retries left]────────────────────────────────────────┘
   │[pass / force_execute]
   ▼
  executor ◄──────────────────────────────────────────────────────────────────┐
   │[success]──► responder ──► END                                            │
   │[error, retries left]──► reflector ──► post_reflect_filter ───────────────┘
   │[error, max iters]──► responder (graceful error) ──► END
"""

from typing import Literal, List, Dict, Optional, Generator
from langgraph.graph import StateGraph, END
from loguru import logger
import time

from core.state import AgentState
from agents import (
    interaction_node,
    planner_node,
    context_builder_node,   # schema-linking + few-shot retrieval (parallel)
    generator_node,
    filter_resolver_node,
    executor_node,
    reflector_node,
    responder_node,
    validator_node,
)
from agents.validator import MAX_VALIDATION_ATTEMPTS
from agents.interaction import DATA_INTENTS
from config import settings

# ── Human-readable node labels (UI + CLI) ────────────────────────────────────
NODE_LABELS: Dict[str, str] = {
    "init":                "⏱️  Initialising",
    "interaction":         "🧠  Understanding your message",
    "direct_respond":      "💬  Responding directly",
    "planner":             "📋  Planning the query",
    "context_builder":     "🔍  Building context (schema + examples)",
    "generator":           "⚙️  Generating SQL",
    "filter_resolver":     "🔎  Verifying filter values",
    "sql_validator":       "✅  Validating SQL",
    "executor":            "▶️  Executing SQL",
    "reflector":           "🔄  Self-correcting SQL",
    "post_reflect_filter": "🔎  Re-checking filters after correction",
    "responder":           "💬  Generating answer",
}


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def should_route(state: AgentState) -> Literal["data_pipeline", "direct_respond"]:
    """
    Gate after interaction:
    • DATA_INTENTS (data_query / follow_up / correction) → full DRGC pipeline
    • Everything else → direct_respond (no SQL)
    """
    intent = state.get("intent", "data_query")
    if intent in DATA_INTENTS:
        return "data_pipeline"
    return "direct_respond"


def should_validate_result(
    state: AgentState,
) -> Literal["execute", "regenerate", "force_execute"]:
    """
    Gate after sql_validator.

    PASSED                                    → execute
    FAILED + retries available                → regenerate  (generator re-runs WITH
                                                validation_issues already in state)
    FAILED + retries exhausted
         OR generator set should_retry=False
         OR no SQL produced                  → force_execute (responder handles gracefully)
    """
    passed   = state.get("validation_passed", True)
    attempts = state.get("sql_validation_attempts", 0)

    if passed:
        logger.info("SQL Validator: PASSED → executing")
        return "execute"

    issues = state.get("validation_issues", [])
    logger.warning(f"SQL Validator: FAILED — {issues}")

    # Generator itself bailed — no point regenerating
    if not state.get("should_retry", True) or not state.get("sql_query"):
        logger.warning("Generator failed / should_retry=False → force_execute")
        return "force_execute"

    if attempts < MAX_VALIDATION_ATTEMPTS:
        logger.info(f"Regenerating SQL (attempt {attempts}/{MAX_VALIDATION_ATTEMPTS})")
        return "regenerate"

    logger.warning("Max validation attempts reached → force_execute")
    return "force_execute"


def should_continue(
    state: AgentState,
) -> Literal["reflect", "end", "success"]:
    """
    Gate after executor.

    No error             → success  (normal path)
    Error + retries left → reflect  (reflector fixes SQL, re-checks filters)
    Error + max iters    → end      (responder explains failure gracefully)
    """
    if state.get("error") is None:
        logger.info("✓ Query successful")
        return "success"

    if state.get("iterations", 0) >= settings.max_iterations:
        logger.warning(f"✗ Max iterations ({settings.max_iterations}) reached")
        return "end"

    if not settings.enable_self_correction:
        logger.warning("✗ Self-correction disabled")
        return "end"

    if not state.get("should_retry", True):
        logger.warning("✗ should_retry=False")
        return "end"

    logger.info(f"↻ SQL correction (iteration {state.get('iterations', 0) + 1})")
    return "reflect"


# ─────────────────────────────────────────────────────────────────────────────
# Utility nodes
# ─────────────────────────────────────────────────────────────────────────────

def add_start_time(state: AgentState) -> dict:
    return {"start_time": time.time()}


def direct_respond_node(state: AgentState) -> dict:
    """
    Terminal node for non-data paths, filter clarification, and result_lookup.
    Moves direct_response → nl_response so the UI renders it identically to
    a regular data answer.
    """
    return {"nl_response": state.get("direct_response") or ""}


def post_reflect_filter_node(state: AgentState) -> dict:
    """
    After reflector rewrites the SQL, re-run filter value verification ONLY if
    the filter resolver has not already run for this query turn.

    Filter resolution is intentionally a one-shot operation per user query:
    once the filter values have been corrected and the SQL rewritten, the
    corrections should be treated as ground truth for all subsequent retry
    loops.  Re-running the resolver on every reflection iteration would
    waste LLM calls, risk double-correcting already-fixed values, and add
    latency to every self-correction cycle.

    The resolver IS skipped when ``filter_log`` is already present in state
    (set by the filter_resolver node on the first pass).
    """
    if state.get("filter_log") is not None:
        logger.debug(
            "post_reflect_filter: filter_log already present — "
            "skipping (filter resolution is one-shot per query)."
        )
        return {}

    from agents.filter_resolver import _filter_resolver_agent
    result = _filter_resolver_agent.resolve(state)
    # Do not surface clarification inside the reflection loop
    result.pop("needs_clarification", None)
    result.pop("direct_response", None)
    result.pop("pending_filter_clarification", None)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and wire the LangGraph workflow."""
    logger.info("Building optimised Text-to-SQL graph (no cache)…")

    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("init",                add_start_time)
    workflow.add_node("interaction",         interaction_node)
    workflow.add_node("direct_respond",      direct_respond_node)
    workflow.add_node("planner",             planner_node)
    workflow.add_node("context_builder",     context_builder_node)
    workflow.add_node("generator",           generator_node)
    workflow.add_node("filter_resolver",     filter_resolver_node)
    workflow.add_node("sql_validator",       validator_node)
    workflow.add_node("executor",            executor_node)
    workflow.add_node("reflector",           reflector_node)
    workflow.add_node("post_reflect_filter", post_reflect_filter_node)
    workflow.add_node("responder",           responder_node)

    # ── Entry ─────────────────────────────────────────────────────────────────
    workflow.set_entry_point("init")
    workflow.add_edge("init", "interaction")

    # ── Intent gate ───────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "interaction",
        should_route,
        {
            "data_pipeline":  "planner",
            "direct_respond": "direct_respond",
        },
    )
    workflow.add_edge("direct_respond", END)

    # ── DRGC pipeline ─────────────────────────────────────────────────────────

    # Planner → context_builder (schema + few-shot in one parallel step)
    workflow.add_edge("planner", "context_builder")

    # Context is now in state; generate SQL.
    # On retry, generator reads validation_issues from state automatically.
    workflow.add_edge("context_builder", "generator")

    # Generator always feeds filter_resolver (SQL changed on every run)
    workflow.add_edge("generator", "filter_resolver")

    # Filter gate
    workflow.add_conditional_edges(
        "filter_resolver",
        lambda s: "clarify" if s.get("needs_clarification") else "validate",
        {
            "clarify":  "direct_respond",
            # When sql_validator is disabled, route straight to executor;
            # the critic agent handles any execution-time errors instead.
            "validate": "sql_validator" if settings.enable_sql_validator else "executor",
        },
    )

    # Validation gate (only reached when enable_sql_validator=True)
    # "regenerate" loops back to generator — schema/few-shot/plan already in state,
    # so context_builder is NOT re-run (no redundant fetching).
    workflow.add_conditional_edges(
        "sql_validator",
        should_validate_result,
        {
            "execute":       "executor",
            "regenerate":    "generator",   # ← loops; reuses existing context
            "force_execute": "executor",
        },
    )

    # Execution gate
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "success": "responder",
            "reflect": "reflector",
            "end":     "responder",   # graceful failure path
        },
    )

    # Self-correction loop:
    # reflector rewrites SQL → re-check affected filter values → execute again
    workflow.add_edge("reflector", "post_reflect_filter")
    # post_reflect_filter always routes to executor (never interrupts with clarification)
    workflow.add_conditional_edges(
        "post_reflect_filter",
        lambda s: "execute",   # unconditional — suppress clarification in loop
        {"execute": "executor"},
    )

    # Terminal
    workflow.add_edge("responder", END)

    logger.info("Graph built successfully")
    return workflow


def compile_graph():
    workflow = build_graph()
    app = workflow.compile()
    logger.info("Graph compiled and ready")
    return app


# Global compiled graph (imported by app.py / cli.py)
graph = compile_graph()


# ─────────────────────────────────────────────────────────────────────────────
# State factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_initial_state(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
) -> AgentState:
    return {
        # Input
        "question":          question,
        # Planning
        "plan":              None,
        "plan_steps":        None,
        # Context (built once by context_builder, reused on retries)
        "relevant_tables":   None,
        "schema_context":    None,
        "schema_metadata":   None,
        "few_shot_examples": None,
        # Generation
        "sql_query":         None,
        "sql_explanation":   None,
        # Execution
        "query_result":      None,
        "result_preview":    None,
        "execution_time_ms": None,
        # Error handling
        "error":             None,
        "error_type":        None,
        # Control flow
        "iterations":        0,
        "should_retry":      True,
        # Response
        "nl_response":       None,
        # Intent / routing
        "intent":            None,
        "direct_response":   None,
        # Conversation
        "conversation_history": conversation_history or [],
        "messages":          [],
        "start_time":        None,
        # SQL validation
        "validation_passed":       None,
        "validation_issues":       [],
        "sql_validation_attempts": 0,
        # Correction
        "user_feedback":     None,
        "previous_sql":      None,
        "entity_column_map": None,
        # Filter value resolution
        "filter_log":        None,
        "needs_clarification": False,
        "pending_filter_clarification": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(question: str, conversation_history: Optional[List[Dict]] = None) -> dict:
    """Synchronous single-turn execution."""
    logger.info("=" * 60)
    logger.info(f"Running Text-to-SQL Agent | question: {question[:80]}")
    logger.info("=" * 60)

    initial_state = _make_initial_state(question, conversation_history)
    try:
        final_state = graph.invoke(initial_state)
        if final_state.get("start_time"):
            total_ms = (time.time() - final_state["start_time"]) * 1000
            final_state["total_latency_ms"] = total_ms
            logger.info(f"Total latency: {total_ms:.0f} ms")
        return final_state
    except Exception as exc:
        logger.error(f"Graph execution error: {exc}")
        return {**initial_state, "error": str(exc), "should_retry": False}


def stream_agent_steps(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
) -> Generator[dict, None, None]:
    """
    Step-by-step streaming generator for the UI / CLI.

    Yields
    ------
    {"type": "step",  "node": str, "label": str, "output": dict}
    {"type": "final", "state": dict}
    """
    initial_state = _make_initial_state(question, conversation_history)
    accumulated: dict = dict(initial_state)

    try:
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    accumulated.update(node_output)
                yield {
                    "type":   "step",
                    "node":   node_name,
                    "label":  NODE_LABELS.get(node_name, node_name),
                    "output": node_output,
                }
    except Exception as exc:
        logger.error(f"Streaming graph error: {exc}")
        accumulated["error"] = str(exc)

    if accumulated.get("start_time"):
        accumulated["total_latency_ms"] = (
            time.time() - accumulated["start_time"]
        ) * 1000

    yield {"type": "final", "state": accumulated}


def stream_nl_response(
    state: dict,
    conversation_history: Optional[List[Dict]] = None,
) -> Generator[str, None, None]:
    """
    Token-level NL answer streamer, called by UI / CLI after graph completes.

    Short-circuits when nl_response is already pre-built (non-data intents,
    filter clarification, result_lookup) — zero extra LLM call in those cases.
    """
    prebuilt = state.get("nl_response") or state.get("direct_response") or ""
    if prebuilt:
        yield prebuilt
        return

    if state.get("needs_clarification"):
        yield state.get("direct_response") or "I need more information before I can run this query."
        return

    from agents.responder import _nl_responder_agent
    yield from _nl_responder_agent.stream_respond(state, conversation_history)


async def run_agent_async(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
) -> dict:
    """Async variant for FastAPI / async callers."""
    initial_state = _make_initial_state(question, conversation_history)
    try:
        final_state = await graph.ainvoke(initial_state)
        if final_state.get("start_time"):
            final_state["total_latency_ms"] = (
                time.time() - final_state["start_time"]
            ) * 1000
        return final_state
    except Exception as exc:
        logger.error(f"Async graph error: {exc}")
        return {**initial_state, "error": str(exc), "should_retry": False}


