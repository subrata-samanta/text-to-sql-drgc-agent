"""
Critic Agent (Refiner): Validates, executes, and corrects SQL queries.
"""

import re
import difflib
from typing import Dict, List, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger
from core.state import AgentState
from core.database import db_manager
from config import settings

# ── All real column names for the nielsen_pos table (used for fuzzy matching) ─
# These are fetched lazily and cached in _get_table_columns().
_COLUMN_CACHE: Optional[List[str]] = None


def _get_table_columns(table: str = "nielsen_pos") -> List[str]:
    """Return the actual column list for *table*, cached after first call."""
    global _COLUMN_CACHE
    if _COLUMN_CACHE is not None:
        return _COLUMN_CACHE
    try:
        rows, err, _ = db_manager.execute_query(
            f"SELECT * FROM {table} LIMIT 0", timeout=10
        )
        if err:
            raise RuntimeError(err)
        # rows is [] when LIMIT 0 — use the DB inspector instead
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(db_manager.engine)
        cols = [c["name"] for c in inspector.get_columns(table)]
        _COLUMN_CACHE = cols
        logger.debug(f"CRITIC: Loaded {len(cols)} columns from {table}")
        return cols
    except Exception as exc:
        logger.warning(f"CRITIC: Could not load column list ({exc}) — fuzzy matching disabled")
        return []


def _extract_bad_column(error_msg: str) -> Optional[str]:
    """
    Parse a SQLite 'no such column' error and return the problematic column name.

    Handles patterns like:
      • no such column: foo
      • table nielsen_pos has no column named foo
      • ambiguous column name: foo
    """
    patterns = [
        r"no such column[:\s]+[\"']?(\w+)[\"']?",
        r"has no column named[:\s]+[\"']?(\w+)[\"']?",
        r"ambiguous column name[:\s]+[\"']?(\w+)[\"']?",
        r"Unknown column[:\s]+[\"']?(\w+)[\"']?",
    ]
    for pat in patterns:
        m = re.search(pat, error_msg, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


# ── Prompt for semantic column resolution (fallback when fuzzy fails) ─────────

_COL_RESOLVE_SYSTEM = """You are a database schema expert for a Nielsen POS dataset.
A SQL query failed because it used a non-existent column name. Your job is to
identify which REAL column the query author most likely intended.

Return a JSON object with exactly these keys:
  "best_match"   : the single best matching real column name (must be from the list provided)
  "confidence"   : "high" | "medium" | "low"
  "reason"       : one short sentence explaining why

Return ONLY the raw JSON — no markdown fences, no extra text."""

_COL_RESOLVE_USER = """Question the user asked:
{question}

Bad column name used in the SQL:
{bad_col}

Real column names available in the table:
{col_list}

Which real column did the query author intend to use?"""


class CriticAgent:
    """
    Validates, executes, and corrects SQL queries.
    Implements closed-loop error correction with execution feedback.
    Includes intelligent column-name resolution for 'no such column' errors.
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.groq_model_reasoning,
            temperature=settings.groq_temperature,
            groq_api_key=settings.groq_api_key
        )
        self._fast_llm = ChatGroq(
            model=settings.groq_model_fast,
            temperature=0.0,
            groq_api_key=settings.groq_api_key,
        )
        self._col_resolve_prompt = ChatPromptTemplate.from_messages([
            ("system", _COL_RESOLVE_SYSTEM),
            ("user",   _COL_RESOLVE_USER),
        ])
        
        # Prompt for error correction
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL debugging expert. A query failed and you must fix it.

Your Task:
1. Analyze the error message carefully
2. Review the schema to understand what went wrong
3. Identify the specific issue (wrong column, incorrect join, syntax error, etc.)
4. Generate a CORRECTED SQL query

Common Error Patterns:
- Column does not exist → Check schema for correct column names
- Table does not exist → Verify table name spelling
- Syntax error → Check SQL dialect requirements
- Ambiguous column → Add table aliases
- Join error → Verify foreign key relationships

CRITICAL — THIS IS SQLite (NOT MySQL / SQL Server):
- "no such function: YEAR"    → replace YEAR(col) with: year_nielsen column OR (year_month / 100)
- "no such function: MONTH"   → replace MONTH(col) with: year_month % 100
- "no such function: QUARTER" → use quarter_nielsen column directly
- "no such function: NOW"     → use date('now')
- "no such function: GETDATE"  → use date('now')
- "no such function: ISNULL"  → use COALESCE(a, b)
- "no such function: NVL"     → use COALESCE(a, b)
- "near FROM: syntax error"   → remove trailing comma before FROM / GROUP BY / ORDER BY
- TOP N not supported         → use LIMIT N
- year_month is a YYYYMM INTEGER (e.g. 202301); do NOT wrap it in YEAR() or strftime()

IMPORTANT: Return ONLY the fixed SQL query (no explanations, no markdown)

SCHEMA:
{schema_context}

ORIGINAL QUESTION:
{question}

FAILED SQL:
{sql_query}

ERROR MESSAGE:
{error}

{column_correction_block}

LOGICAL PLAN (reference):
{plan}

Generate the CORRECTED SQL:"""),
            ("user", "Fix the query:")
        ])
    
    def _resolve_bad_column_smart(
        self,
        bad_col: str,
        question: str,
        real_cols: List[str],
    ) -> Optional[str]:
        """
        Resolve a bad column name to the best-matching real column using a
        two-stage approach:

        Stage 1 — Fuzzy matching (difflib):
          Fast, zero LLM cost. Works well for minor typos / underscore variants
          (e.g. 'sales_dollar_value' → 'sales_dollar',
                'year'              → 'year_nielsen',
                'sub_category_name' → 'sub_category').

        Stage 2 — Semantic LLM call (fallback):
          When fuzzy confidence is low (cutoff < 0.6), ask the fast LLM to pick
          the best column given the user's question as context.
          Example: 'revenue' → 'sales_dollar' even though the strings differ.

        Returns the resolved column name, or None if resolution fails.
        """
        if not real_cols:
            return None

        bad_lower = bad_col.lower()

        # ── Stage 1: fuzzy match ───────────────────────────────────────────────
        close = difflib.get_close_matches(
            bad_lower,
            [c.lower() for c in real_cols],
            n=1,
            cutoff=0.60,
        )
        if close:
            # Map back to original case
            matched = real_cols[[c.lower() for c in real_cols].index(close[0])]
            logger.info(
                f"CRITIC column resolver: fuzzy '{bad_col}' → '{matched}'  "
                f"(score ≥ 0.60)"
            )
            return matched

        # ── Stage 2: semantic LLM fallback ────────────────────────────────────
        logger.info(
            f"CRITIC column resolver: fuzzy failed for '{bad_col}' — "
            "falling back to semantic LLM resolution"
        )
        try:
            chain = self._col_resolve_prompt | self._fast_llm | JsonOutputParser()
            result: dict = chain.invoke({
                "question": question,
                "bad_col":  bad_col,
                "col_list": "\n".join(f"  - {c}" for c in real_cols),
            })
            best = str(result.get("best_match", "")).strip()
            confidence = str(result.get("confidence", "")).strip()
            reason = str(result.get("reason", "")).strip()

            if best and best in real_cols:
                logger.info(
                    f"CRITIC column resolver: LLM '{bad_col}' → '{best}'  "
                    f"confidence={confidence} | {reason}"
                )
                return best

            # LLM returned something not in real_cols — do a final fuzzy pass
            if best:
                fallback = difflib.get_close_matches(
                    best.lower(),
                    [c.lower() for c in real_cols],
                    n=1,
                    cutoff=0.50,
                )
                if fallback:
                    matched = real_cols[
                        [c.lower() for c in real_cols].index(fallback[0])
                    ]
                    logger.info(
                        f"CRITIC column resolver: LLM suggestion '{best}' "
                        f"snapped to '{matched}'"
                    )
                    return matched

        except Exception as exc:
            logger.warning(f"CRITIC column resolver: LLM call failed ({exc})")

        logger.warning(
            f"CRITIC column resolver: could not resolve '{bad_col}' — "
            "leaving it to the reflection prompt"
        )
        return None

    def _build_column_correction_block(
        self,
        error_msg: str,
        question: str,
    ) -> str:
        """
        If the error is a 'column not found' type, resolve the correct column
        and return a formatted correction block to inject into the reflection
        prompt. Returns an empty string for non-column errors.
        """
        error_type = self._classify_error(error_msg)
        if error_type not in ("column_not_found", "ambiguous_column"):
            return ""

        bad_col = _extract_bad_column(error_msg)
        if not bad_col:
            return ""

        real_cols = _get_table_columns()
        resolved = self._resolve_bad_column_smart(bad_col, question, real_cols)

        if resolved:
            return (
                f"[COLUMN CORRECTION — MUST APPLY]\n"
                f"The column '{bad_col}' does not exist in the table.\n"
                f"Based on the user's question, the correct column to use is: '{resolved}'.\n"
                f"Replace every occurrence of '{bad_col}' with '{resolved}' in the fixed SQL."
            )

        # Could not resolve — give the LLM the full column list as a hint
        if real_cols:
            col_hint = ", ".join(real_cols[:40])  # cap at 40 to avoid prompt bloat
            return (
                f"[COLUMN NOTE]\n"
                f"The column '{bad_col}' does not exist. "
                f"Real available columns are:\n{col_hint}\n"
                f"Pick the most appropriate column for the user's question."
            )

        return ""

    def execute_and_validate(self, state: AgentState) -> dict:
        """
        Execute SQL query and handle results/errors.

        Args:
            state: Current agent state

        Returns:
            Updated state with results or error information
        """
        logger.info("CRITIC: Executing and validating SQL query")
        
        sql_query = state.get("sql_query")
        if not sql_query:
            return {
                "error": "No SQL query to execute",
                "should_retry": False
            }
        
        try:
            # Execute the query
            result, error, exec_time = db_manager.execute_query(
                sql_query,
                timeout=settings.query_timeout_seconds
            )
            
            if error:
                # Query failed - prepare for reflection
                logger.warning(f"Query execution failed: {error}")
                error_type = self._classify_error(error)
                
                return {
                    "error": error,
                    "error_type": error_type,
                    "query_result": None,
                    "execution_time_ms": exec_time,
                    "should_retry": True
                }
            else:
                # Query succeeded
                logger.info(f"Query executed successfully in {exec_time:.2f}ms")
                result_preview = self._format_result_preview(result)
                
                return {
                    "query_result": result,
                    "result_preview": result_preview,
                    "execution_time_ms": exec_time,
                    "error": None,
                    "should_retry": False
                }
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "error": str(e),
                "error_type": "runtime",
                "should_retry": True
            }
    
    def reflect_and_fix(self, state: AgentState) -> dict:
        """
        Analyze error and generate corrected SQL.
        Intelligently resolves wrong column names before handing off to the LLM.
        """
        logger.info("CRITIC: Reflecting on error and fixing SQL")
        
        iterations = state.get("iterations", 0)
        
        # Check if we've exceeded max iterations
        if iterations >= settings.max_iterations:
            logger.error(f"Max iterations ({settings.max_iterations}) reached")
            return {
                "should_retry": False,
                "error": f"Failed to generate valid SQL after {settings.max_iterations} attempts"
            }
        
        question      = state["question"]
        plan          = state.get("plan", "")
        schema_context = state.get("schema_context", "")
        sql_query     = state.get("sql_query", "")
        error         = state.get("error", "")

        # ── Intelligent column resolution ─────────────────────────────────────
        column_correction_block = self._build_column_correction_block(error, question)
        if column_correction_block:
            logger.info(f"CRITIC: Column correction block injected:\n{column_correction_block}")

        try:
            chain = self.reflection_prompt | self.llm
            
            response = chain.invoke({
                "question":                question,
                "plan":                    plan,
                "schema_context":          schema_context,
                "sql_query":               sql_query,
                "error":                   error,
                "column_correction_block": column_correction_block,
            })
            
            # Clean the fixed SQL
            from agents.generator import SQLGeneratorAgent
            generator = SQLGeneratorAgent()
            fixed_sql = generator._clean_sql(response.content)
            
            logger.info(f"Generated corrected SQL (iteration {iterations + 1})")
            logger.debug(f"Fixed SQL: {fixed_sql}")
            
            return {
                "sql_query":  fixed_sql,
                "iterations": iterations + 1,
                "should_retry": True,
            }
            
        except Exception as e:
            logger.error(f"Reflection error: {e}")
            return {
                "error": f"Failed to correct SQL: {str(e)}",
                "should_retry": False,
            }
    
    def _classify_error(self, error_msg: str) -> str:
        """
        Classify error type for better handling.
        
        Args:
            error_msg: Error message from database
            
        Returns:
            Error category
        """
        error_lower = error_msg.lower()
        
        if "column" in error_lower and ("does not exist" in error_lower or "not found" in error_lower):
            return "column_not_found"
        elif "table" in error_lower and ("does not exist" in error_lower or "not found" in error_lower):
            return "table_not_found"
        elif "syntax" in error_lower:
            return "syntax_error"
        elif "ambiguous" in error_lower:
            return "ambiguous_column"
        elif "timeout" in error_lower:
            return "timeout"
        else:
            return "runtime_error"
    
    def _format_result_preview(self, result, max_rows: int = 5) -> str:
        """
        Format query result for display.
        
        Args:
            result: Query result (list of rows or message)
            max_rows: Maximum rows to include in preview
            
        Returns:
            Formatted string preview
        """
        if isinstance(result, str):
            return result
        
        if not result:
            return "Query returned no results"
        
        try:
            # Handle list of dicts (preferred) or SQLAlchemy Row objects
            first = result[0]
            if isinstance(first, dict):
                rows = result[:max_rows]
            elif hasattr(first, '_mapping'):
                rows = [dict(row._mapping) for row in result[:max_rows]]
            elif hasattr(first, '_asdict'):
                rows = [row._asdict() for row in result[:max_rows]]
            else:
                return str(result[:max_rows])

            preview = f"Returned {len(result)} row(s). Preview:\n"
            for i, row in enumerate(rows, 1):
                preview += f"Row {i}: {row}\n"

            if len(result) > max_rows:
                preview += f"... ({len(result) - max_rows} more rows)"

            return preview
                
        except Exception as e:
            logger.warning(f"Could not format result: {e}")
            return str(result)[:500]  # Truncate to 500 chars


# Node functions for LangGraph
def executor_node(state: AgentState) -> dict:
    """LangGraph node wrapper for execution."""
    agent = CriticAgent()
    return agent.execute_and_validate(state)


def reflector_node(state: AgentState) -> dict:
    """LangGraph node wrapper for reflection/correction."""
    agent = CriticAgent()
    return agent.reflect_and_fix(state)
