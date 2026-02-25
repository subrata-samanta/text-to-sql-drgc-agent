"""
SQL Generator Agent: Translates logical plans into SQL queries.
"""

import re

from langchain_core.prompts import ChatPromptTemplate
from core.llm_factory import create_llm
from loguru import logger
from core.state import AgentState
from config import settings

# Provider-aware table name substituted into few-shot example SQL at injection time.
_TABLE_REF: str = (
    settings.dbx_full_table
    if settings.llm_provider.lower() == "dbrx"
    else "nielsen_pos"
)


def _fix_dbrx_table_quoting(sql: str) -> str:
    """
    Normalise the fully-qualified Databricks table reference in generated SQL.

    LLMs occasionally quote catalog/schema/table parts with single or double
    quotes instead of backticks, or omit quotes entirely, producing invalid SQL:

        FROM 'dev-amer-customer-catalog'.'schema'.'table'   -- single-quoted
        FROM "dev-amer-customer-catalog"."schema"."table"   -- double-quoted
        FROM dev-amer-customer-catalog.schema.table         -- unquoted

    We replace every such variant with the canonical backtick form from settings.
    Four explicit patterns cover single-quoted, double-quoted, backtick (already
    correct but re-normalised for safety), and unquoted.
    """
    cat   = re.escape(settings.dbx_catalog)
    sch   = re.escape(settings.dbx_schema_name)
    tbl   = re.escape(settings.dbx_table)
    canon = settings.dbx_full_table   # `cat`.`sch`.`tbl`

    # Build one pattern per quoting style: single, double, backtick, none.
    _variants = [
        rf"'{{cat}}'\.'{{sch}}'\.'{{tbl}}'",       # 'cat'.'sch'.'tbl'
        rf"\"{{cat}}\"\.\"{{sch}}\"\.\"{{tbl}}\"",  # "cat"."sch"."tbl"
        rf"`{{cat}}`\.`{{sch}}`\.`{{tbl}}`",       # `cat`.`sch`.`tbl`  (already correct)
        rf"{{cat}}\.{{sch}}\.{{tbl}}",              # cat.sch.tbl  (no quotes)
    ]
    # Substitute the actual escaped names into each pattern string.
    patterns = [
        v.format(cat=cat, sch=sch, tbl=tbl)
        for v in _variants
    ]

    fixed = sql
    for pat in patterns:
        fixed = re.sub(pat, canon, fixed, flags=re.IGNORECASE)

    if fixed != sql:
        logger.info("Generator: normalised Databricks table reference quoting in generated SQL.")
    return fixed


_BASE_SYSTEM = """\
You are an expert SQL engineer for a Nielsen retail analytics platform.
Your job is to write correct, efficient SQL that faithfully implements the
business logic of this organisation.

════════════════════════════════════════════════════════════════════════
PRIORITY ORDER  (follow from highest to lowest)
════════════════════════════════════════════════════════════════════════
  1. FEW-SHOT EXAMPLES  — these encode REAL business logic and must be
     replicated as closely as possible.  SQL patterns, CTE structure,
     metric formulas, and filter logic shown in the examples ARE the
     ground truth.  Any deviation requires explicit justification.
  2. SCHEMA & CTE RULES — use only the columns and table in the schema;
     obey every mandatory SQL rule listed there.
  3. LOGICAL PLAN — translate the plan into SQL, but never override the
     business logic patterns established in the examples.
  4. USER QUESTION — the ultimate goal; resolve ambiguity using the
     examples as reference.
════════════════════════════════════════════════════════════════════════

{few_shot_block}\
SCHEMA:
{schema_context}

LOGICAL PLAN:
{plan}

USER QUESTION:
{question}

INSTRUCTIONS:
- IF examples were provided above, replicate their CTE structure, metric
  definitions, and filter logic.  Treat them as templates, not hints.
- Return ONLY executable SQL — no markdown fences, no prose explanation.
- Think step by step (Chain-of-Thought) before writing the final query,
  but strip the reasoning from your final answer.
"""

_FEW_SHOT_HEADER = """\
════════════════════════════════════════════════════════════════════════
MANDATORY BUSINESS-LOGIC EXAMPLES  (highest priority — follow exactly)
════════════════════════════════════════════════════════════════════════
The SQL patterns below were written by business analysts and encode the
EXACT formulas, filters, and CTE conventions required.  Study each
example carefully before writing your query.

{examples}
════════════════════════════════════════════════════════════════════════

"""


def _build_few_shot_block(examples: list) -> str:
    """Render few-shot examples as an inline system-prompt section.

    Any ``{table_name}`` placeholder in the YAML SQL is replaced with the
    provider-specific table reference so dbrx examples show the fully-qualified
    Databricks table name rather than the literal placeholder text.
    """
    if not examples:
        return ""
    parts = []
    for i, ex in enumerate(examples, 1):
        q   = ex.get("question", "").strip()
        sql = ex.get("sql", "").strip()
        # Substitute the YAML {table_name} placeholder with the real table ref
        sql = sql.replace("{table_name}", _TABLE_REF)
        parts.append(f"Example {i}:\n  Question: {q}\n  SQL:\n{sql}\n")
    rendered = "\n".join(parts)
    return _FEW_SHOT_HEADER.format(examples=rendered)


# ─────────────────────────────────────────────────────────────────────────────
# SQLite compatibility patcher
# Fixes common MySQL / SQL-Server functions that do not exist in SQLite.
# Applied automatically after every LLM generation — before the SQL is used.
# ─────────────────────────────────────────────────────────────────────────────

# Columns known to store YYYYMM integers (never wrap in strftime)
_INT_TIME_COLS = re.compile(
    r"\b(year_month|year_nielsen|quarter_nielsen|month_num|period_num|week_num)\b",
    re.IGNORECASE,
)

def _fix_sqlite_compat(sql: str) -> str:
    """
    Auto-patch non-SQLite function calls before the query reaches the database.

    Handles:
      YEAR(year_month)     → year_month / 100          (YYYYMM integer → year)
      YEAR(period_date)    → CAST(strftime('%Y', period_date) AS INTEGER)
      MONTH(year_month)    → year_month % 100
      MONTH(period_date)   → CAST(strftime('%m', period_date) AS INTEGER)
      QUARTER(anything)    → 'quarter_nielsen'  (column exists on the table)
      DATE_FORMAT(c, f)    → strftime(sqlite_fmt, c)
      GETDATE() / NOW()    → date('now')
      ISNULL(a, b)         → COALESCE(a, b)
      NVL(a, b)            → COALESCE(a, b)
      TOP N                → removed (caller must add LIMIT N)
      Trailing comma       → removed before FROM / WHERE / GROUP / ORDER / HAVING
    """
    original = sql

    # ── 1. YEAR(col) ──────────────────────────────────────────────────────────
    def _replace_year(m: re.Match) -> str:
        col = m.group(1).strip()
        if _INT_TIME_COLS.match(col.split()[0]):   # already an integer column
            return f"({col} / 100)"
        return f"CAST(strftime('%Y', {col}) AS INTEGER)"

    sql = re.sub(r"\bYEAR\s*\(([^)]+)\)", _replace_year, sql, flags=re.IGNORECASE)

    # ── 2. MONTH(col) ─────────────────────────────────────────────────────────
    def _replace_month(m: re.Match) -> str:
        col = m.group(1).strip()
        if _INT_TIME_COLS.match(col.split()[0]):
            return f"({col} % 100)"
        return f"CAST(strftime('%m', {col}) AS INTEGER)"

    sql = re.sub(r"\bMONTH\s*\(([^)]+)\)", _replace_month, sql, flags=re.IGNORECASE)

    # ── 3. QUARTER(col) → quarter_nielsen column ──────────────────────────────
    sql = re.sub(r"\bQUARTER\s*\([^)]+\)", "quarter_nielsen", sql, flags=re.IGNORECASE)

    # ── 4. DATE_FORMAT(col, '%Y') → strftime('%Y', col) ──────────────────────
    def _replace_date_format(m: re.Match) -> str:
        col, fmt = m.group(1).strip(), m.group(2).strip()
        # Map MySQL format specifiers to SQLite ones (they happen to match)
        return f"strftime({fmt}, {col})"

    sql = re.sub(
        r"\bDATE_FORMAT\s*\(([^,]+),\s*([^)]+)\)",
        _replace_date_format, sql, flags=re.IGNORECASE,
    )

    # ── 5. GETDATE() / NOW() → date('now') ────────────────────────────────────
    sql = re.sub(r"\bGETDATE\s*\(\s*\)", "date('now')", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bNOW\s*\(\s*\)", "date('now')", sql, flags=re.IGNORECASE)

    # ── 6. ISNULL(a, b) / NVL(a, b) → COALESCE(a, b) ────────────────────────
    sql = re.sub(r"\bISNULL\s*\(", "COALESCE(", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bNVL\s*\(", "COALESCE(", sql, flags=re.IGNORECASE)

    # ── 7. TOP N → strip it (LIMIT should be at end) ─────────────────────────
    sql = re.sub(r"\bSELECT\s+TOP\s+\d+\s+", "SELECT ", sql, flags=re.IGNORECASE)

    # ── 8. Trailing comma before clause keywords ──────────────────────────────
    sql = re.sub(
        r",\s*(FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION|EXCEPT|INTERSECT)\b",
        r" \1",
        sql, flags=re.IGNORECASE,
    )

    if sql != original:
        changed = []
        checks = [
            (r"\bYEAR\s*\(", "YEAR()"),
            (r"\bMONTH\s*\(", "MONTH()"),
            (r"\bQUARTER\s*\(", "QUARTER()"),
            (r"\bDATE_FORMAT\s*\(", "DATE_FORMAT()"),
            (r"\b(GETDATE|NOW)\s*\(", "GETDATE/NOW()"),
            (r"\b(ISNULL|NVL)\s*\(", "ISNULL/NVL()"),
        ]
        for pat, label in checks:
            if re.search(pat, original, flags=re.IGNORECASE):
                changed.append(label)
        if re.search(r",\s*(FROM|WHERE|GROUP|ORDER|HAVING)", original, flags=re.IGNORECASE):
            changed.append("trailing-comma")
        logger.info(f"SQLite compat patcher fixed: {', '.join(changed) or 'misc'}")

    return sql


class SQLGeneratorAgent:
    """
    Translates logical plans into valid SQL queries using Chain-of-Thought reasoning.
    Gives the highest priority to few-shot business-logic examples by embedding
    them directly inside the system prompt.
    """

    def __init__(self):
        self.llm = create_llm("reasoning")

        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", _BASE_SYSTEM),
            ("user", "Write the SQL query now:"),
        ])
    
    def generate(self, state: AgentState, few_shot_examples=None) -> dict:
        """
        Generate SQL query from plan and schema.
        Few-shot examples are embedded at the TOP of the system prompt so the
        LLM treats them as the highest-priority business-logic reference.
        """
        logger.info("SQL GENERATOR: Creating SQL query from plan")

        question      = state["question"]
        plan          = state.get("plan", "")

        # ── Smart retry: inject validation failure feedback ───────────────────
        validation_issues = state.get("validation_issues") or []
        if validation_issues and state.get("sql_validation_attempts", 0) > 0:
            issues_block = "\n".join(f"  - {iss}" for iss in validation_issues)
            plan = (plan or "") + (
                f"\n\n[VALIDATION ISSUES — fix ALL of these in the new SQL]\n"
                f"{issues_block}"
            )

        # ── Correction intent: inject user feedback + previous SQL ────────────
        user_feedback = state.get("user_feedback") or ""
        previous_sql  = state.get("previous_sql") or ""
        if user_feedback:
            plan = (plan or "") + (
                f"\n\n[USER CORRECTION]\nUser says: {user_feedback}"
            )
            if previous_sql:
                plan += f"\nPrevious SQL for reference (fix the issue):\n{previous_sql}"

        schema_context = state.get("schema_context", "")

        if not schema_context:
            logger.error("No schema context available")
            return {
                "error": "Cannot generate SQL without schema context",
                "should_retry": False
            }

        # Build the few-shot block (empty string if no examples)
        use_examples = (
            few_shot_examples
            and settings.enable_dynamic_few_shot
        )
        few_shot_block = _build_few_shot_block(few_shot_examples) if use_examples else ""

        if use_examples:
            logger.info(
                f"SQL GENERATOR: Injecting {len(few_shot_examples)} "
                "few-shot example(s) into system prompt (highest priority)"
            )
        else:
            logger.info("SQL GENERATOR: No few-shot examples available — using schema + plan only")

        try:
            chain = self.generation_prompt | self.llm

            response = chain.invoke({
                "question":       question,
                "plan":           plan,
                "schema_context": schema_context,
                "few_shot_block": few_shot_block,
            })

            sql = self._clean_sql(response.content)
            logger.info(f"Generated SQL ({len(sql)} characters)")
            logger.debug(f"SQL: {sql}")

            return {
                "sql_query":       sql,
                "sql_explanation": response.content,
            }

        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return {
                "error": f"SQL generation failed: {str(e)}",
                "should_retry": False
            }

    def _clean_sql(self, raw_sql: str) -> str:
        """
        Clean SQL output from LLM response.
        Removes markdown formatting and extracts SQL query.
        
        Args:
            raw_sql: Raw SQL from LLM
            
        Returns:
            Cleaned SQL string
        """
        # Remove markdown code blocks
        sql = raw_sql.replace("```sql", "").replace("```", "").strip()
        
        # Extract SQL from response (if it contains reasoning + SQL)
        # Look for SQL keywords: SELECT, WITH, INSERT, UPDATE, DELETE
        lines = sql.split("\n")
        sql_start_idx = None
        
        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]):
                sql_start_idx = i
                break
        
        if sql_start_idx is not None:
            sql = "\n".join(lines[sql_start_idx:])

        sql = sql.strip()

        # ── Truncation guard ──────────────────────────────────────────────────
        # If the LLM hit the max_tokens ceiling the SQL will be cut mid-query.
        # Reliable heuristics (checked in order, OR-combined):
        #
        #   1. Unbalanced parentheses — any multi-CTE query truncated before its
        #      final closing paren will have more '(' than ')'.
        #
        #   2. Last meaningful token is a mid-clause keyword that can never legally
        #      end a SQL statement (AND, OR, WHERE, FROM, ON, JOIN, CASE, WHEN, …).
        #
        # Both checks are applied; if either fires the SQL is treated as truncated.
        _open_parens  = sql.count("(")
        _close_parens = sql.count(")")
        _paren_unbalanced = _open_parens > _close_parens

        # Strip trailing whitespace / semicolon to inspect the real last token
        _sql_for_check = sql.rstrip().rstrip(";").rstrip()
        _ends_mid_clause = bool(re.search(
            r"\b(AND|OR|WHERE|FROM|JOIN|ON|CASE|WHEN|THEN|AS|BY|,)\s*$",
            _sql_for_check, re.IGNORECASE,
        ))

        _looks_truncated = bool(sql) and (_paren_unbalanced or _ends_mid_clause)

        if _looks_truncated:
            reason = []
            if _paren_unbalanced:
                reason.append(f"unbalanced parens: {_open_parens} open vs {_close_parens} close")
            if _ends_mid_clause:
                reason.append("ends mid-clause")
            logger.error(
                f"Generator: SQL appears truncated ({'; '.join(reason)}) — "
                f"likely hit max_tokens limit ({settings.dbrx_max_tokens}). "
                f"Last 120 chars: ...{sql[-120:]!r}"
            )
            raise ValueError(
                f"Generated SQL was truncated by the LLM token limit "
                f"(max_tokens={settings.dbrx_max_tokens}). "
                "Increase DBRX_MAX_TOKENS in .env and retry."
            )

        # For dbrx: normalise any incorrect quoting of the fully-qualified table
        # name (LLMs sometimes emit single/double quotes instead of backticks).
        if settings.llm_provider.lower() == "dbrx":
            sql = _fix_dbrx_table_quoting(sql)
        else:
            # Auto-patch SQLite incompatible functions (YEAR, MONTH, QUARTER, etc.)
            sql = _fix_sqlite_compat(sql)

        return sql


# ── Module-level singleton ───────────────────────────────────────────────────
# SQLGeneratorAgent is stateless beyond __init__; the generation_prompt and
# self.llm are immutable after construction.
_sql_generator_agent = SQLGeneratorAgent()


# Node function for LangGraph
def generator_node(state: AgentState) -> dict:
    """LangGraph node wrapper for SQLGeneratorAgent."""
    # Get few-shot examples from state (retrieved in previous step)
    few_shot = state.get("few_shot_examples", None)
    return _sql_generator_agent.generate(state, few_shot_examples=few_shot)
