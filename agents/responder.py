"""
NL Responder Agent: converts SQL execution results into natural language answers.
Supports both sync (for graph node caching) and token-streaming (for UI).
"""

from typing import Generator, List, Dict, Optional, Tuple
import re

import sqlglot
import sqlglot.expressions as exp
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from core.state import AgentState


# ── Normal-result prompt ──────────────────────────────────────────────────────

_SYSTEM = """You are a concise, business-oriented data analyst who answers questions \
based on SQL query results from a Nielsen POS dataset.

Rules:
- Answer ONLY from the numbers in the provided SQL result — never make up data
- Use business-friendly language (no SQL jargon)
- Format numbers clearly: dollar amounts as $X.XXM / $X.XXK, percentages as XX.X%
- 1–3 sentences for simple queries; a short paragraph for multi-row / comparative results
- For multi-row results: highlight the top finding first, then briefly summarize the rest
- When the question is a follow-up, use the conversation history for context but answer \
  the current question specifically

Response structure when a qualifier was dropped (check "Filter / query notes"):
  Step 1 — Answer: Present the numbers from the SQL result, fully and clearly.
  Step 2 — One-sentence transparency note: After the numbers, add a single sentence \
    explaining that the result covers the broader matched category (e.g. "Cookies") \
    because the specific qualifier (e.g. "gluten-free") is not tracked as an attribute \
    in this dataset.
  Step 3 — Ask + Suggest: Ask the user if they would like more details, then offer \
    2–3 concrete follow-up questions.  IMPORTANT: every example in Step 3 MUST use \
    real values taken from the "Suggestion context" section — do NOT invent brand names, \
    categories, or sub-categories; only reference values visible in that section.

If "Filter / query notes" says "Filters matched the question exactly.", skip Steps 2 and 3 \
and answer normally.

Conversation history (most recent turns, for follow-up context):
{conversation_history}"""

_USER = """Question: {question}

Filter / query notes (qualifiers from the question that could NOT be applied to the SQL):
{filter_transparency}

SQL clauses actually used:
{sql_query_snippet}

SQL result:
{result_preview}

Suggestion context — real database values from the matched category (use ONLY these for follow-up suggestions):
{suggestion_context}

Answer:"""

# ── No-data / graceful-failure prompt ────────────────────────────────────────

_SYSTEM_NO_DATA = """You are a helpful Nielsen POS analytics assistant. \
A user's query returned no rows from the database.

Your job:
1. Acknowledge that no data was found for the exact combination they asked about.
2. Explain the most likely reason (filter value not present, time period out of range, \
   combination doesn't exist in the dataset, etc.).
3. Using the "Available options" section below, tell the user what VALUES actually exist \
   in the database for each dimension they queried — so they can refine their question.
4. Suggest 1–2 rephrased questions they could try.

Keep the tone helpful and brief. Do NOT mention SQL or technical terms."""

_USER_NO_DATA = """User question: {question}

Filters / dimensions used in the query:
{filters_used}

Available options in the database for those dimensions:
{available_options}

Write a helpful response (2–4 short paragraphs):"""

# ── Error / exhausted-retries prompt ─────────────────────────────────────────

_SYSTEM_ERROR = """You are a helpful Nielsen POS analytics assistant. \
A user's data question could not be answered due to a technical issue.

Your job:
1. Apologise briefly and explain in plain language that the query could not be completed.
2. Describe the likely cause in simple terms (no SQL jargon, no error codes).
3. Using the "Available options" section, suggest what the user CAN ask about.
4. Give 1–2 example rephrased questions.

Keep the tone calm, professional, and helpful."""

_USER_ERROR = """User question: {question}

Technical issue summary: {error_summary}

Available options in the database (sampled):
{available_options}

Write a helpful response (2–3 short paragraphs):"""


class NLResponderAgent:
    """Generates natural language answers from SQL results."""

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("user", _USER),
        ])
        self.prompt_no_data = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_NO_DATA),
            ("user", _USER_NO_DATA),
        ])
        self.prompt_error = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_ERROR),
            ("user", _USER_ERROR),
        ])

    # ── helpers ──────────────────────────────────────────────────────────────
    def _make_llm(self, streaming: bool = False):
        """Create a provider-aware LLM instance for response generation."""
        from core.llm_factory import create_llm
        return create_llm("reasoning", streaming=streaming)
    def _history_text(self, history: Optional[List[Dict]]) -> str:
        if not history:
            return "None"
        return "\n\n".join(
            f"Q: {h['question']}\nA: {h['nl_response']}"
            for h in history[-4:]
        )

    @staticmethod
    def _build_filter_transparency(state: AgentState) -> str:
        """
        Build a plain-English summary of what the SQL actually filtered on vs
        what the user originally asked for.  Uses the filter_log written by
        FilterResolverAgent and the sql_query itself.

        Returns a short bullet list (or "Filters matched the question exactly.").
        """
        lines: list[str] = []

        # 1. Filter corrections recorded by the filter resolver
        filter_log: list[dict] = state.get("filter_log") or []
        for entry in filter_log:
            action  = entry.get("action", "")
            col     = entry.get("column", "")
            asked   = entry.get("original_value") or entry.get("sql_value", "")
            matched = entry.get("db_match", "")
            matched_col = entry.get("matched_column") or col

            if action in ("corrected", "fuzzy_match"):
                lines.append(
                    f"- '{asked}' was not found exactly; closest match used: "
                    f"'{matched}' ({matched_col})"
                )
            elif action == "column_corrected":
                lines.append(
                    f"- '{asked}' was re-mapped from column '{col}' to "
                    f"'{matched_col}' (value: '{matched}')"
                )
            elif action == "not_found":
                lines.append(
                    f"- '{asked}' ({col}) had no match in the database; "
                    f"filter was dropped from the query"
                )
            elif action == "business_rule":
                lines.append(
                    f"- '{asked}' was expanded via a business rule to "
                    f"'{matched}' ({matched_col})"
                )

        # 2. Scan the question for qualifiers the SQL likely cannot capture
        question: str = state.get("question") or ""
        sql: str      = state.get("sql_query") or ""

        _IMPLICIT_QUALIFIERS = [
            # (question keyword patterns, explanation)
            (r"\bgluten[- ]?free\b", "gluten-free attribute"),
            (r"\borganic\b",          "organic attribute"),
            (r"\bsugar[- ]?free\b",   "sugar-free attribute"),
            (r"\blow[- ]?sodium\b",   "low-sodium attribute"),
            (r"\bvegan\b",            "vegan attribute"),
            (r"\bkosher\b",           "kosher attribute"),
            (r"\bpremium\b",          "premium tier"),
            (r"\bprivate[- ]?label\b","private-label segment"),
            (r"\bnew\b",              "new/innovation flag"),
            (r"\bseasonal\b",         "seasonal flag"),
        ]
        for pattern, label in _IMPLICIT_QUALIFIERS:
            if re.search(pattern, question, re.IGNORECASE):
                # Only flag if the qualifier doesn’t appear in the SQL text
                if not re.search(pattern, sql, re.IGNORECASE):
                    lines.append(
                        f"- '{label}' is not a stored column in this dataset; "
                        f"the query ran without that filter"
                    )

        if not lines:
            return "Filters matched the question exactly."

        return "\n".join(lines)

    def _sql_snippet(self, state: AgentState, max_chars: int = 600) -> str:
        """Return a readable excerpt of the SQL (WHERE + GROUP BY clauses only)."""
        sql = (state.get("sql_query") or "").strip()
        if not sql:
            return "(not available)"
        # Try to show just the WHERE..GROUP BY part for brevity
        try:
            m = re.search(r'(WHERE\b.+?)(?:ORDER BY|LIMIT|$)',
                          sql, re.IGNORECASE | re.DOTALL)
            if m:
                snippet = m.group(1).strip()
                if len(snippet) > max_chars:
                    snippet = snippet[:max_chars] + " …"
                return snippet
        except Exception:
            pass
        return sql[:max_chars] + (" …" if len(sql) > max_chars else "")

    def _build_suggestion_context(self, state: AgentState) -> str:
        """
        When a qualifier was silently dropped (e.g. gluten-free), query the DB
        using the SAME WHERE conditions that were actually executed, and return
        real values (brands, sub-categories, products) from that matched scope.
        The LLM uses these to compose concrete, factual follow-up suggestions.
        Returns an empty string when no qualifier was dropped.
        """
        # Only needed when something was dropped
        transparency = self._build_filter_transparency(state)
        if transparency == "Filters matched the question exactly.":
            return "(none — filters matched exactly)"

        from core.database import db_manager  # lazy import

        sql = state.get("sql_query") or ""
        fallback_table = self._extract_main_table(sql)
        if not fallback_table:
            return "(table not identified)"

        # Reuse the executed WHERE clause so samples are scoped to the matched category
        where_match = re.search(
            r'(WHERE\b.+?)(?:GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING|$)',
            sql, re.IGNORECASE | re.DOTALL
        )
        where_clause = where_match.group(1).strip() if where_match else ""

        # Columns that produce useful follow-up suggestions
        suggestion_cols = [
            "BRAND_NAME", "SUB_CATEGORY_NAME", "CATEGORY_NAME",
            "PRODUCT_DESCRIPTION", "MANUFACTURER_NAME",
        ]
        lines: list[str] = []
        for col in suggestion_cols:
            try:
                q = f'SELECT DISTINCT "{col}" FROM "{fallback_table}"'
                if where_clause:
                    q += f" {where_clause}"
                q += f' WHERE "{col}" IS NOT NULL ORDER BY "{col}" LIMIT 8' \
                     if not where_clause else \
                     f' AND "{col}" IS NOT NULL ORDER BY "{col}" LIMIT 8'
                rows, err, _ = db_manager.execute_query(q, timeout=10)
                if err or not rows:
                    continue
                vals: list[str] = []
                for r in rows:
                    if isinstance(r, dict):
                        v = list(r.values())[0]
                    elif hasattr(r, "_mapping"):
                        v = list(dict(r._mapping).values())[0]
                    else:
                        v = str(r)
                    if v:
                        vals.append(str(v))
                if vals:
                    lines.append(f"• {col}: {', '.join(vals)}")
            except Exception as exc:
                logger.debug(f"Suggestion-context query failed for {col}: {exc}")
                continue

        return "\n".join(lines) or "(no sample values available)"

    def _build_input(self, state: AgentState, history: Optional[List[Dict]]) -> dict:
        filter_transparency = self._build_filter_transparency(state)
        return {
            "question":             state["question"],
            "result_preview":       state.get("result_preview") or "No results returned.",
            "conversation_history": self._history_text(history),
            "filter_transparency":  filter_transparency,
            "sql_query_snippet":    self._sql_snippet(state),
            "suggestion_context":   self._build_suggestion_context(state),
        }

    # ── no-data helpers ───────────────────────────────────────────────────────

    def _extract_filter_columns(self, sql: str) -> List[Tuple[str, str]]:
        """
        Parse the SQL and return (table, column) pairs that appear in WHERE
        clauses and JOIN conditions — these are the dimensions the user queried on.
        Falls back to a simple regex if sqlglot can't parse the dialect.
        Returns up to 6 unique pairs to avoid too many DB round-trips.
        """
        pairs: list[Tuple[str, str]] = []
        try:
            tree = sqlglot.parse_one(sql, error_level=sqlglot.ErrorLevel.IGNORE)
            if tree is None:
                raise ValueError("parse returned None")

            # Collect FROM / JOIN tables (first alias → real name mapping)
            table_map: dict[str, str] = {}
            for tbl in tree.find_all(exp.Table):
                real = tbl.name or ""
                alias = tbl.alias or real
                if real:
                    table_map[alias.upper()] = real

            # Walk every Column node inside a Where or Join
            def _in_where_or_join(node) -> bool:
                p = node.parent
                while p is not None:
                    if isinstance(p, (exp.Where, exp.Join)):
                        return True
                    p = p.parent
                return False

            seen: set[Tuple[str, str]] = set()
            for col in tree.find_all(exp.Column):
                if not _in_where_or_join(col):
                    continue
                col_name = col.name or ""
                tbl_alias = (col.table or "").upper()
                real_tbl = table_map.get(tbl_alias, tbl_alias) or (
                    list(table_map.values())[0] if table_map else ""
                )
                key = (real_tbl, col_name)
                if col_name and key not in seen:
                    seen.add(key)
                    pairs.append(key)
                    if len(pairs) >= 6:
                        break
        except Exception:
            # Regex fallback — grab words after WHERE / AND / OR before = / IN / LIKE
            for m in re.finditer(
                r'(?:WHERE|AND|OR)\s+(?:\w+\.)?(\w+)\s*(?:=|IN|LIKE|IS)',
                sql, re.IGNORECASE
            ):
                col_name = m.group(1)
                if col_name.upper() not in ("NULL", "NOT", "TRUE", "FALSE"):
                    pairs.append(("", col_name))
                if len(pairs) >= 6:
                    break

        return pairs

    def _extract_main_table(self, sql: str) -> str:
        """Return the first FROM table name from the SQL."""
        try:
            tree = sqlglot.parse_one(sql, error_level=sqlglot.ErrorLevel.IGNORE)
            if tree:
                tbl = tree.find(exp.Table)
                if tbl:
                    return tbl.name or ""
        except Exception:
            pass
        m = re.search(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
        return m.group(1) if m else ""

    def _fetch_available_options(
        self,
        filter_cols: List[Tuple[str, str]],
        fallback_table: str,
    ) -> Tuple[str, str]:
        """
        For each (table, column) pair run a quick DISTINCT query and
        return two strings:
            filters_used    — human-readable list of the column names queried on
            available_opts  — bullet list of column: [val1, val2, …]
        Silently skips columns that fail or have too many distinct values (>50).
        """
        from core.database import db_manager  # lazy import — avoid circular

        filters_used_lines: list[str] = []
        option_lines: list[str] = []

        seen_cols: set[str] = set()
        for tbl, col in filter_cols:
            if col.lower() in seen_cols:
                continue
            seen_cols.add(col.lower())
            filters_used_lines.append(f"• {col}" + (f" (table: {tbl})" if tbl else ""))

            use_tbl = tbl or fallback_table
            if not use_tbl:
                option_lines.append(f"• {col}: (table unknown — cannot sample)")
                continue
            try:
                rows, err, _ = db_manager.execute_query(
                    f'SELECT DISTINCT "{col}" FROM "{use_tbl}" '
                    f'WHERE "{col}" IS NOT NULL ORDER BY "{col}" LIMIT 20',
                    timeout=10,
                )
                if err or not rows:
                    option_lines.append(f"• {col}: (no sample available)")
                    continue
                vals = []
                for r in rows:
                    if isinstance(r, dict):
                        v = list(r.values())[0]
                    elif hasattr(r, "_mapping"):
                        v = list(dict(r._mapping).values())[0]
                    else:
                        v = str(r)
                    vals.append(str(v))
                # If too many values, just show count
                if len(vals) >= 20:
                    option_lines.append(f"• {col}: many values — sample: {', '.join(vals[:8])}, …")
                else:
                    option_lines.append(f"• {col}: {', '.join(vals)}")
            except Exception as exc:
                logger.debug(f"Available-options query failed for {col}: {exc}")
                option_lines.append(f"• {col}: (could not retrieve options)")

        return (
            "\n".join(filters_used_lines) or "• (no specific filters detected)",
            "\n".join(option_lines) or "• (no options available)",
        )

    def _build_no_data_response(
        self,
        state: AgentState,
        streaming: bool = False,
    ):
        """
        Build a helpful 'no data found' response by sampling the DB for
        available values and handing them to a dedicated LLM prompt.
        Returns a generator (streaming=True) or a plain string (streaming=False).
        """
        sql = state.get("sql_query") or ""
        question = state["question"]
        filter_cols = self._extract_filter_columns(sql)
        fallback_table = self._extract_main_table(sql)
        filters_used, available_opts = self._fetch_available_options(
            filter_cols, fallback_table
        )

        inp = {
            "question": question,
            "filters_used": filters_used,
            "available_options": available_opts,
        }

        if streaming:
            llm = self._make_llm(streaming=True)
            chain = self.prompt_no_data | llm

            def _gen():
                for chunk in chain.stream(inp):
                    if chunk.content:
                        yield chunk.content
            return _gen()
        else:
            llm = self._make_llm()
            chain = self.prompt_no_data | llm
            return chain.invoke(inp).content.strip()

    def _build_error_response(
        self,
        state: AgentState,
        streaming: bool = False,
    ):
        """
        Build a graceful error response with available-options context.
        Used when the agent exhausted all SQL retries.
        """
        sql = state.get("sql_query") or ""
        question = state["question"]
        raw_error = state.get("error") or "Unknown error"

        # Summarise the error without exposing raw SQL details
        error_summary = (
            "The system could not find the right combination of data for your question. "
            f"Technical detail: {raw_error[:200]}"
        )

        # Sample the main table for general available options
        fallback_table = self._extract_main_table(sql)
        filter_cols = self._extract_filter_columns(sql)
        # If no filter cols detected, try to probe a few key columns generically
        if not filter_cols and fallback_table:
            filter_cols = []   # _fetch_available_options will return minimal info
        _, available_opts = self._fetch_available_options(filter_cols, fallback_table)

        inp = {
            "question": question,
            "error_summary": error_summary,
            "available_options": available_opts,
        }

        if streaming:
            llm = self._make_llm(streaming=True)
            chain = self.prompt_error | llm

            def _gen():
                for chunk in chain.stream(inp):
                    if chunk.content:
                        yield chunk.content
            return _gen()
        else:
            llm = self._make_llm()
            chain = self.prompt_error | llm
            return chain.invoke(inp).content.strip()

    # ── public helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _result_is_empty(state: AgentState) -> bool:
        """True when the query ran but returned zero data rows."""
        result = state.get("query_result")
        if result is None:
            return False          # not yet executed, or error path
        if isinstance(result, list):
            return len(result) == 0
        if isinstance(result, str):
            return False          # e.g. "Query executed successfully"
        return False

    # ── sync respond (used as LangGraph node) ────────────────────────────────

    def respond(self, state: AgentState,
                conversation_history: Optional[List[Dict]] = None) -> dict:
        """Sync call — stores nl_response in graph state."""
        logger.info("RESPONDER: Generating natural language answer")

        # ── exhausted retries / hard error ──
        if state.get("error"):
            logger.info("RESPONDER: error path — generating graceful error response")
            try:
                msg = self._build_error_response(state, streaming=False)
            except Exception as exc:
                logger.warning(f"Error-response builder failed ({exc}), using fallback")
                msg = (
                    f"I was unable to answer your question due to a technical issue. "
                    f"Please try rephrasing or contact support.\n"
                    f"Detail: {state['error']}"
                )
            return {"nl_response": msg}

        # ── empty result set ──
        if self._result_is_empty(state):
            logger.info("RESPONDER: zero rows — generating no-data guidance response")
            try:
                msg = self._build_no_data_response(state, streaming=False)
            except Exception as exc:
                logger.warning(f"No-data response builder failed ({exc}), using fallback")
                msg = "No data was found for your query. Please try different filters or a different time period."
            return {"nl_response": msg}

        # ── normal path ──
        try:
            llm = self._make_llm()
            chain = self.prompt | llm
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

        # ── exhausted retries / hard error ──
        if state.get("error"):
            logger.info("RESPONDER: streaming error path — graceful error response")
            try:
                yield from self._build_error_response(state, streaming=True)
            except Exception as exc:
                logger.warning(f"Streaming error-response failed ({exc})")
                yield (
                    "I was unable to complete your query due to a technical issue. "
                    "Please try rephrasing your question."
                )
            return

        # ── empty result set ──
        if self._result_is_empty(state):
            logger.info("RESPONDER: streaming zero-rows path — no-data guidance")
            try:
                yield from self._build_no_data_response(state, streaming=True)
            except Exception as exc:
                logger.warning(f"Streaming no-data response failed ({exc})")
                yield "No data was found for your query. Please try different filters or a different time period."
            return

        # ── no result_preview at all (should rarely happen) ──
        if not state.get("result_preview"):
            yield "The query returned no results."
            return

        # ── normal path ──
        try:
            streaming_llm = self._make_llm(streaming=True)
            chain = self.prompt | streaming_llm
            history = conversation_history or state.get("conversation_history")
            for chunk in chain.stream(self._build_input(state, history)):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Responder streaming error: {e}")
            yield state.get("result_preview") or "Query executed successfully."


# ── Module-level singleton ───────────────────────────────────────────────────
# NLResponderAgent only holds ChatPromptTemplate objects (immutable) after
# __init__.  The LLM is obtained via _make_llm() → create_llm() which now
# returns the globally cached instance.
_nl_responder_agent = NLResponderAgent()


# ── LangGraph node ───────────────────────────────────────────────────────────

def responder_node(state: AgentState) -> dict:
    """LangGraph node wrapper for NLResponderAgent (non-streaming, for caching)."""
    return _nl_responder_agent.respond(state)
