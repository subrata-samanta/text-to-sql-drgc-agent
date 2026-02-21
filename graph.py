"""
LangGraph workflow orchestration for the Text-to-SQL agent.
Implements the DRGC (Decomposition-Retrieval-Generation-Correction) framework.
"""

from typing import Literal, List, Dict, Optional, Generator
from langgraph.graph import StateGraph, END
from loguru import logger
import time

from core.state import AgentState
from agents import (
    interaction_node,
    planner_node,
    schema_linker_node,
    generator_node,
    filter_resolver_node,
    executor_node,
    reflector_node,
    responder_node,
    validator_node,
)
from agents.validator import MAX_VALIDATION_ATTEMPTS
from agents.interaction import DATA_INTENTS
from tools import semantic_cache, few_shot_retriever
from config import settings

# Human-readable labels for each graph node (used by UI + CLI)
NODE_LABELS: Dict[str, str] = {
    "init":             "⏱️  Initializing",
    "interaction":      "🧠  Understanding your message",
    "direct_respond":   "💬  Responding directly",
    "check_cache":      "⚡  Checking semantic cache",
    "planner":          "📋  Planning the query",
    "retrieve_few_shot":"📚  Retrieving few-shot examples",
    "schema_retriever": "🔍  Linking Nielsen schema columns",
    "generator":        "⚙️  Generating SQL (CTE)",
    "filter_resolver":  "🔎  Verifying filter values against DB",
    "sql_validator":    "✅  Validating SQL correctness",
    "executor":         "▶️  Executing SQL",
    "reflector":        "🔄  Self-correcting SQL",
    "cache_result":     "💾  Caching result",
    "responder":        "💬  Generating natural language answer",
}


def should_continue(state: AgentState) -> Literal["reflect", "end", "cache_success"]:
    """
    Determines the next step in the workflow after query execution.
    
    Decision flow:
    - If query succeeded: cache result and end
    - If max iterations reached: end with error
    - If self-correction disabled: end with error
    - Otherwise: attempt to fix the error
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name: "cache_success", "end", or "reflect"
    """
    # If there's no error, cache and end
    if state.get("error") is None:
        logger.info("✓ Query successful - caching and ending workflow")
        return "cache_success"
    
    # If max iterations reached, stop
    if state.get("iterations", 0) >= settings.max_iterations:
        logger.warning(f"✗ Max iterations ({settings.max_iterations}) reached - ending workflow")
        return "end"
    
    # If self-correction is disabled, stop
    if not settings.enable_self_correction:
        logger.warning("✗ Self-correction disabled - ending workflow")
        return "end"
    
    # If should_retry flag is False, stop
    if not state.get("should_retry", True):
        logger.warning("✗ Retry flag is False - ending workflow")
        return "end"
    
    # Otherwise, attempt reflection/correction
    logger.info(f"↻ Attempting correction (iteration {state.get('iterations', 0) + 1})")
    return "reflect"


def add_start_time(state: AgentState) -> dict:
    """Add timestamp at start of workflow."""
    return {"start_time": time.time()}


def check_cache_node(state: AgentState) -> dict:
    """
    Checks if we have a cached result for this question.
    Uses semantic similarity to find matching previous queries.
    Cache is always bypassed when the user has provided correction feedback
    (we must regenerate, not return the same wrong answer).
    """
    question = state["question"]

    # Never serve cache for correction requests
    if state.get("user_feedback"):
        logger.info("Cache bypassed — correction mode active")
        return {"cache_hit": False}

    cached = semantic_cache.get(question)

    if cached:
        logger.info("✓ Using cached result")
        return {
            **cached,
            "cache_hit": True,
            "should_retry": False
        }
    else:
        return {"cache_hit": False}


def retrieve_few_shot_node(state: AgentState) -> dict:
    """
    Retrieves similar SQL examples from vector store to guide generation.
    These examples help the LLM write better SQL queries.
    """
    if not settings.enable_dynamic_few_shot:
        return {}
    
    question = state["question"]
    examples = few_shot_retriever.retrieve(question)
    
    logger.info(f"Retrieved {len(examples)} few-shot examples")
    return {"few_shot_examples": examples}


def should_use_cache(state: AgentState) -> Literal["use_cache", "continue"]:
    """
    Decides whether to use cached result or continue with full workflow.
    Cache hits are still routed to the SQL validator to confirm correctness.
    """
    if state.get("cache_hit", False):
        return "use_cache"
    return "continue"


def should_validate_result(
    state: AgentState,
) -> Literal["execute", "use_cached_result", "regenerate", "bypass_cache", "force_execute"]:
    """
    Routes after sql_validator:

    Validation PASSED:
      - cache hit  → "use_cached_result"  (query_result already in state → responder)
      - fresh SQL  → "execute"             (run the SQL)

    Validation FAILED:
      - cache hit              → "bypass_cache"   (planner; cache_hit reset)
      - fresh, under limit     → "regenerate"     (back to generator)
      - fresh, at/over limit   → "force_execute"  (give up, try the SQL anyway)
    """
    passed    = state.get("validation_passed", True)
    cache_hit = state.get("cache_hit", False)
    attempts  = state.get("sql_validation_attempts", 0)

    if passed:
        if cache_hit:
            logger.info("SQL Validator: PASSED (cache) → using cached result")
            return "use_cached_result"
        logger.info("SQL Validator: PASSED → executing SQL")
        return "execute"

    # Validation failed
    issues = state.get("validation_issues", [])
    logger.warning(f"SQL Validator: FAILED — {issues}")

    if cache_hit:
        logger.info("Bypassing cache and re-planning...")
        return "bypass_cache"

    # If the generator itself failed (set should_retry=False), there is no SQL
    # to improve by regenerating.  Skip the retry loop and surface the error.
    if not state.get("should_retry", True):
        logger.warning("Generator failed with should_retry=False — skipping regeneration")
        return "force_execute"

    if attempts < MAX_VALIDATION_ATTEMPTS:
        logger.info(f"Regenerating SQL (attempt {attempts})...")
        return "regenerate"

    logger.warning("Max validation attempts reached — forwarding SQL to executor")
    return "force_execute"


def reset_cache_and_sql(state: AgentState) -> dict:
    """
    Intermediate node: clears cache_hit + sql_query so the planner/generator
    start fresh after a failed cache-hit validation.
    """
    return {
        "cache_hit": False,
        "sql_query": None,
        "query_result": None,
        "result_preview": None,
        "plan": None,
        "plan_steps": None,
        "filter_log": None,
        "needs_clarification": False,
        "pending_filter_clarification": None,
    }


def should_route(state: AgentState) -> Literal["data_pipeline", "direct_respond"]:
    """
    Routes after the interaction node:
    - data intents (data_query / follow_up) → full DRGC SQL pipeline
    - everything else (smalltalk / clarification / out_of_scope) → direct response
    """
    intent = state.get("intent", "data_query")
    if intent in DATA_INTENTS:
        return "data_pipeline"
    return "direct_respond"


def direct_respond_node(state: AgentState) -> dict:
    """
    Lightweight terminal node for non-data intents.
    Moves the pre-built direct_response into nl_response so the UI
    can render it identically to a regular answer.
    """
    return {"nl_response": state.get("direct_response") or ""}


def cache_result_node(state: AgentState) -> dict:
    """
    Stores successful query results in semantic cache for future use.
    Only caches when query executed without errors.
    """
    # Only cache if query was successful
    if state.get("error") is None and state.get("sql_query"):
        result_to_cache = {
            "sql_query": state["sql_query"],
            "query_result": state.get("query_result"),
            "result_preview": state.get("result_preview"),
            "plan": state.get("plan"),
            "relevant_tables": state.get("relevant_tables")
        }
        
        semantic_cache.set(state["question"], result_to_cache)
    
    return {}


def build_graph() -> StateGraph:
    """
    Builds the LangGraph workflow for the Text-to-SQL agent.
    
    Workflow:
    1. Initialize timestamp
    2. Check cache (if hit, skip to end)
    3. Plan: Break down question into logical steps
    4. Retrieve: Get few-shot examples from vector store
    5. Schema Link: Find relevant tables/columns
    6. Generate: Write SQL query
    7. Execute: Run query and validate
    8. On error: Reflect and retry (up to max_iterations)
    9. On success: Cache result and end
    
    Returns:
        Configured StateGraph ready for compilation
    """
    logger.info("Building Text-to-SQL agent graph...")
    
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # === ADD NODES ===
    # Each node represents a step in the workflow
    workflow.add_node("init", add_start_time)  # Track execution time
    workflow.add_node("interaction", interaction_node)  # Classify intent & route
    workflow.add_node("direct_respond", direct_respond_node)  # Shortcut for non-data
    workflow.add_node("check_cache", check_cache_node)  # Try to use cached result
    workflow.add_node("planner", planner_node)  # Decompose question into steps
    workflow.add_node("retrieve_few_shot", retrieve_few_shot_node)  # Get example queries
    workflow.add_node("schema_retriever", schema_linker_node)  # Find relevant tables
    workflow.add_node("generator", generator_node)  # Generate SQL
    workflow.add_node("filter_resolver", filter_resolver_node)  # Verify & correct filter values
    workflow.add_node("sql_validator", validator_node)  # Validate SQL vs question
    workflow.add_node("reset_cache", reset_cache_and_sql)  # Clear stale cache state
    workflow.add_node("executor", executor_node)  # Execute and validate
    workflow.add_node("reflector", reflector_node)  # Fix errors if any
    workflow.add_node("cache_result", cache_result_node)  # Store successful result
    workflow.add_node("responder", responder_node)  # NL answer after execution
    
    # === DEFINE WORKFLOW ===
    workflow.set_entry_point("init")
    
    # Step 1: classify intent
    workflow.add_edge("init", "interaction")

    # Step 2: route based on intent
    workflow.add_conditional_edges(
        "interaction",
        should_route,
        {
            "data_pipeline":  "check_cache",  # needs SQL
            "direct_respond": "direct_respond",  # smalltalk / clarification / OOS
        }
    )

    # Non-data path terminates here
    workflow.add_edge("direct_respond", END)
    
    # Cache hit → validate the cached SQL; miss → plan fresh
    workflow.add_conditional_edges(
        "check_cache",
        should_use_cache,
        {
            "use_cache": "sql_validator",   # validate before trusting cache
            "continue":  "planner",
        }
    )

    # Linear flow through the DRGC pipeline
    workflow.add_edge("planner", "retrieve_few_shot")
    workflow.add_edge("retrieve_few_shot", "schema_retriever")
    workflow.add_edge("schema_retriever", "generator")

    # After generation, resolve filter values against real DB values
    workflow.add_edge("generator", "filter_resolver")

    # After filter resolution: if clarification needed route to direct_respond,
    # otherwise proceed to validation
    workflow.add_conditional_edges(
        "filter_resolver",
        lambda s: "clarify" if s.get("needs_clarification") else "validate",
        {
            "clarify":  "direct_respond",
            "validate": "sql_validator",
        }
    )

    # Validator routing
    workflow.add_conditional_edges(
        "sql_validator",
        should_validate_result,
        {
            "execute":            "executor",      # fresh SQL, validated
            "use_cached_result":  "responder",     # cache hit, validated – skip re-exec
            "regenerate":         "generator",     # fresh SQL, needs rework
            "bypass_cache":       "reset_cache",   # cache was wrong → full regen
            "force_execute":      "executor",      # gave up validating → try anyway
        }
    )

    # After cache reset, start from planner
    workflow.add_edge("reset_cache", "planner")
    
    # After execution, decide: success (cache), error (reflect), or give up (end)
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "end": "responder",          # error path still gets an NL answer
            "cache_success": "cache_result",
            "reflect": "reflector"
        }
    )

    workflow.add_edge("cache_result", "responder")
    workflow.add_edge("responder", END)
    
    # After reflection, retry execution
    workflow.add_edge("reflector", "executor")
    
    logger.info("Graph built successfully")
    return workflow


def compile_graph():
    """
    Compile the workflow graph.
    
    Returns:
        Compiled graph ready for execution
    """
    workflow = build_graph()
    app = workflow.compile()
    logger.info("Graph compiled and ready")
    return app


# Create global graph instance
graph = compile_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Shared state factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_initial_state(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
) -> AgentState:
    return {
        "question": question,
        "plan": None,
        "plan_steps": None,
        "relevant_tables": None,
        "schema_context": None,
        "schema_metadata": None,
        "sql_query": None,
        "sql_explanation": None,
        "few_shot_examples": None,
        "query_result": None,
        "result_preview": None,
        "execution_time_ms": None,
        "error": None,
        "error_type": None,
        "nl_response": None,
        "intent": None,
        "direct_response": None,
        "iterations": 0,
        "should_retry": True,
        "messages": [],
        "start_time": None,
        "cache_hit": False,
        "conversation_history": conversation_history or [],
        # SQL validation
        "validation_passed": None,
        "validation_issues": [],
        "sql_validation_attempts": 0,
        # Feedback / correction
        "user_feedback": None,
        "previous_sql":  None,
        "entity_column_map": None,
        # Filter value resolution
        "filter_log": None,
        "needs_clarification": False,
        "pending_filter_clarification": None,
    }


def run_agent(question: str, conversation_history: Optional[List[Dict]] = None) -> dict:
    """
    Execute the Text-to-SQL agent for a given question.
    Optionally pass conversation_history for follow-up question support.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Running Text-to-SQL Agent")
    logger.info(f"Question: {question}")
    logger.info(f"{'='*60}")

    initial_state = _make_initial_state(question, conversation_history)

    try:
        final_state = graph.invoke(initial_state)

        if final_state.get("start_time"):
            total_time = (time.time() - final_state["start_time"]) * 1000
            logger.info(f"Total execution time: {total_time:.2f}ms")
            final_state["total_latency_ms"] = total_time

        if final_state.get("error"):
            logger.error(f"✗ Agent failed: {final_state['error']}")
        else:
            logger.info(f"✓ Agent succeeded")
            logger.info(f"SQL: {final_state.get('sql_query', 'N/A')}")

        return final_state

    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        return {**initial_state, "error": str(e), "should_retry": False}


# ─────────────────────────────────────────────────────────────────────────────
# Streaming API (step events + token streaming)
# ─────────────────────────────────────────────────────────────────────────────

def stream_agent_steps(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
) -> Generator[dict, None, None]:
    """
    Generator that streams agent execution step-by-step.

    Yields dicts of two shapes:
      {"type": "step",  "node": str, "label": str, "output": dict}
      {"type": "final", "state": dict}   ← last event, accumulated final state
    """
    initial_state = _make_initial_state(question, conversation_history)
    accumulated: dict = dict(initial_state)

    try:
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    accumulated.update(node_output)
                yield {
                    "type": "step",
                    "node": node_name,
                    "label": NODE_LABELS.get(node_name, node_name),
                    "output": node_output,
                }
    except Exception as e:
        logger.error(f"Streaming graph error: {e}")
        accumulated["error"] = str(e)

    # Attach total latency
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
    Token-level streaming generator for the natural language answer.
    Call this AFTER stream_agent_steps completes and you have the final state.

    For non-data intents (smalltalk / clarification / out_of_scope), the
    nl_response is already pre-computed by direct_respond_node — yield it
    directly without calling the LLM again.
    """
    # Short-circuit 1: nl_response already pre-computed
    # This covers:
    #  - Non-data intents (smalltalk / clarification / out_of_scope)
    #  - Filter-value clarification mid-data-pipeline (needs_clarification=True)
    prebuilt = state.get("nl_response") or state.get("direct_response") or ""
    if prebuilt:
        yield prebuilt
        return

    # Short-circuit 2: data intent but explicitly flagged as clarification
    if state.get("needs_clarification"):
        yield state.get("direct_response") or "I need more information before I can run this query."
        return

    from agents.responder import NLResponderAgent
    agent = NLResponderAgent()
    yield from agent.stream_respond(state, conversation_history)


async def run_agent_async(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
) -> dict:
    """Asynchronous version of run_agent."""
    logger.info(f"{'='*60}")
    logger.info(f"Running Text-to-SQL Agent (Async)")
    logger.info(f"Question: {question}")
    logger.info(f"{'='*60}")

    initial_state = _make_initial_state(question, conversation_history)

    try:
        final_state = await graph.ainvoke(initial_state)

        if final_state.get("start_time"):
            total_time = (time.time() - final_state["start_time"]) * 1000
            final_state["total_latency_ms"] = total_time

        return final_state

    except Exception as e:
        logger.error(f"Async graph execution error: {e}")
        return {**initial_state, "error": str(e), "should_retry": False}
