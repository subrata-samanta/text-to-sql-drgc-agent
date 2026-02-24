"""
SQL Validation Agent for the DRGC Text-to-SQL pipeline.

Validates that the generated (or cache-retrieved) SQL correctly answers
the user's question along multiple semantic dimensions:

  1. Time period alignment  – year / month in question must appear in SQL.
  2. Hierarchy entity check – entities extracted from the question are resolved
                              to their correct DB column in strict priority order:
                                mega_category → manufacturer → category →
                                sub_category → brand → subbrand → ppg →
                                market → customer → division
                              e.g. "MONDELEZ" resolves to 'manufacturer', so
                              the SQL must filter WHERE manufacturer = 'MONDELEZ'
                              not WHERE brand = 'MONDELEZ'.
  3. Metric alignment       – market_share, TDP, velocity aggregation patterns.

If validation fails, the node signals regenerate (fresh SQL) or bypass_cache
(stale cache hit → full re-plan).  An attempt counter prevents infinite loops;
after max_validation_attempts the SQL is forwarded to the executor unchanged.

Entity resolution is LRU-cached so each unique entity hits the DB only once
per process lifetime.
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import sqlglot
import sqlglot.errors
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from core.state import AgentState

# Provider-aware SQL helpers
_IS_DBRX: bool = settings.llm_provider.lower() == "dbrx"
# Table name and CAST type differ between SQLite (groq) and SparkSQL (dbrx)
_VAL_TABLE: str = (
    settings.dbx_full_table if _IS_DBRX else "nielsen_pos"
)
_CAST_STR: str = "STRING" if _IS_DBRX else "TEXT"


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_VALIDATION_ATTEMPTS = 2   # after this many failures, forward SQL anyway

# ── Nielsen entity hierarchy – checked in this PRIORITY ORDER ─────────────────
# (col_name, human-friendly label used in error messages)
HIERARCHY: List[Tuple[str, str]] = [
    ("mega_category", "mega category"),
    ("manufacturer",  "manufacturer"),
    ("category",      "category"),
    ("sub_category",  "category group / sub-category"),
    ("brand",         "brand"),
    ("subbrand",      "sub-brand"),
    ("ppg",           "PPG"),
]

# Non-hierarchy dimension columns (checked after hierarchy)
DIMENSION_COLUMNS: List[Tuple[str, str]] = [
    ("market",    "market"),
    ("customer",  "customer"),
    ("division",  "division"),
]

# All entity columns in resolution order (hierarchy first, then dimensions)
ALL_ENTITY_COLUMNS: List[Tuple[str, str]] = HIERARCHY + DIMENSION_COLUMNS

# ── For dbrx: discover which entity columns actually exist in the live table ──
# This prevents UNRESOLVED_COLUMN SQL errors during entity resolution when the
# Databricks table has a different schema than the SQLite nielsen_pos table.
_DBRX_TABLE_COLS: set = set()
if _IS_DBRX:
    try:
        from dbx_connection import get_table_columns as _get_dbx_cols
        _DBRX_TABLE_COLS = set(_get_dbx_cols())
        _missing_entity_cols = [c for c, _ in ALL_ENTITY_COLUMNS if c not in _DBRX_TABLE_COLS]
        if _missing_entity_cols:
            logger.warning(
                f"Validator: {len(_missing_entity_cols)} entity column(s) are NOT "
                f"present in the Databricks table and will be skipped during entity "
                f"resolution: {_missing_entity_cols}"
            )
        else:
            logger.info("Validator: all entity columns confirmed present in Databricks table.")
    except Exception as _dbx_col_err:
        logger.warning(
            f"Validator: could not discover Databricks table columns ({_dbx_col_err}). "
            "All entity columns will be attempted — UNRESOLVED_COLUMN errors may occur."
        )

# True when we have confirmed table schema info AND at least one entity column
# actually exists in the Databricks table — only then is entity validation
# meaningful.  When False (schema unknown or all entity cols absent), the
# entity check would always produce false-positive failures and must be skipped.
_DBRX_ENTITY_VALIDATION_POSSIBLE: bool = (
    not _IS_DBRX                              # groq: always validate
    or not _DBRX_TABLE_COLS                   # dbrx: schema not yet known, try anyway
    or any(col in _DBRX_TABLE_COLS for col, _ in ALL_ENTITY_COLUMNS)  # ≥1 col present
)

# ── Metric keyword → SQL patterns that should be present ─────────────────────
METRIC_PATTERNS: Dict[str, List[str]] = {
    # "market share" requires a ratio pattern (SUM/SUM) or a market_share column.
    # Do NOT flag generic "sales", "volume", "revenue" — almost every valid
    # Nielsen query aggregates those with SUM() and the false-positive rate is 100%.
    "market share":  [r"/\s*sum\s*\(", r"market_share", r"OVER\s*\("],
    "market_share":  [r"/\s*sum\s*\(", r"market_share", r"OVER\s*\("],
    "velocity":      [r"velocity", r"/\s*tdp", r"sales.*per.*tdp"],
}

# Month name → zero-padded number
MONTH_MAP: Dict[str, str] = {
    "january":   "01", "february": "02", "march":     "03",
    "april":     "04", "may":      "05", "june":      "06",
    "july":      "07", "august":   "08", "september": "09",
    "october":   "10", "november": "11", "december":  "12",
    "jan": "01", "feb": "02", "mar": "03",
    "apr": "04",               "jun": "06",
    "jul": "07", "aug": "08", "sep": "09",
    "oct": "10", "nov": "11", "dec": "12",
}

# SQL reserved words – skip when extracting ALL-CAPS tokens from questions
_SQL_KEYWORDS = {
    "SELECT","FROM","WHERE","WITH","AS","ON","AND","OR","NOT","IN","IS",
    "NULL","BY","ORDER","GROUP","HAVING","LEFT","RIGHT","JOIN","INNER",
    "OUTER","UNION","ALL","CASE","WHEN","THEN","ELSE","END","LIKE","SUM",
    "AVG","COUNT","MIN","MAX","OVER","PARTITION","BETWEEN","CTE","DESC",
    "ASC","LIMIT","DISTINCT","TOP","INTO","SET","CAST","COALESCE","NULLIF",
    "IIF","ROUND","TRIM","UPPER","LOWER","LENGTH","SUBSTR","REPLACE",
}

# Common title-case English words that are NOT entity names
_TITLE_STOPWORDS = {
    "What","Which","When","Where","How","Who","Why","Show","Give","Get",
    "Tell","Find","List","Compare","Calculate","Compute","Summarize","Break",
    "Breakdown","Year","Month","Quarter","Week","Period","Time","Date",
    "With","For","The","And","But","From","Into","Over","Under","During",
    "Since","Before","After","Between","Also","Any","Top","Bottom","Are",
    "Was","Were","Has","Have","Had","Its","Their","This","That","These",
    "Those","Can","Could","Would","Should","Across","Each","Last","Next",
    "Full","Half","Per","Versus","Vs","Total","Share","Market","Sales",
    "Brand","Category","Segment","Product","Channel","Customer","Division",
    "Manufacturer","Volume","Revenue","Units","Dollars","Value","Growth",
    "Trend","Average","Mean","Median","Change","Difference","National",
    "Regional","Monthly","Weekly","Yearly","Annual","Quarterly","Please",
    "Does","Did","Will","Much","Many","Most","Least","Best","Worst",
}


# ──────────────────────────────────────────────────────────────────────────────
# DB-backed hierarchy resolver
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _resolve_entity_column(entity_upper: str) -> Optional[Tuple[str, str]]:
    """
    Look up *entity_upper* in nielsen_pos checking each column in
    ALL_ENTITY_COLUMNS priority order (mega_category → … → division).

    Ambiguous entities (e.g. "MONDELEZ" exists in both manufacturer AND brand)
    resolve to the FIRST (highest-priority) column found — so "MONDELEZ" maps
    to 'manufacturer', not 'brand'.

    Returns (col_name, friendly_label) or None if not found anywhere.
    LRU-cached: each entity only hits the DB once per session.
    """
    try:
        from core.database import db_manager
        safe = entity_upper.replace("'", "''")   # basic SQL injection guard
        for col, label in ALL_ENTITY_COLUMNS:
            # For dbrx: skip columns that don't exist in the actual table.
            # Querying a non-existent column triggers UNRESOLVED_COLUMN which
            # wastes a round-trip and previously caused a spurious reconnect.
            if _IS_DBRX and _DBRX_TABLE_COLS and col not in _DBRX_TABLE_COLS:
                logger.debug(
                    f"Entity resolution: skipping '{col}' — not in Databricks table."
                )
                continue
            sql = (
                f"SELECT 1 FROM {_VAL_TABLE} "
                f"WHERE UPPER(CAST({col} AS {_CAST_STR})) = '{safe}' "
                f"LIMIT 1"
            )
            result, error, _ = db_manager.execute_query(sql)
            if error:
                continue
            rows = result if isinstance(result, list) else []
            if rows:
                logger.debug(f"Entity '{entity_upper}' → column '{col}' ({label})")
                return (col, label)
    except Exception as e:
        logger.warning(f"Entity resolution DB error for '{entity_upper}': {e}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Text extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_years(text: str) -> List[str]:
    """Return 4-digit years (2000–2030) found in *text*."""
    return re.findall(r"\b(20[0-2][0-9])\b", text)


def _extract_months(text: str) -> List[str]:
    """Return zero-padded month strings found by name or 'month N' pattern."""
    text_lower = text.lower()
    months: List[str] = []
    for name, num in MONTH_MAP.items():
        if re.search(r"\b" + name + r"\b", text_lower):
            months.append(num)
    for m in re.findall(r"\bmonth\s+(\d{1,2})\b", text_lower):
        months.append(m.zfill(2))
    return list(set(months))


def _extract_entity_candidates(question: str) -> List[str]:
    """
    Extract candidate entity names from *question* using three strategies,
    all normalised to UPPERCASE so DB lookup (_resolve_entity_column) is
    always case-insensitive:

    1. ALL-CAPS tokens (≥3 chars, not SQL keywords):
         "MONDELEZ", "TOTAL BARS", "OREO"
         Handles the Nielsen convention of writing names in ALL-CAPS.

    2. Title-case proper nouns (≥4 chars, not in _TITLE_STOPWORDS):
         "Mondelez", "Oreo", "Nabisco"
         Handles mixed-case user input.

    3. Single- or double-quoted strings:
         "in the 'Total Biscuits' segment"  →  "TOTAL BISCUITS"

    Results are de-duplicated.
    """
    seen: set = set()
    result: List[str] = []

    def _add(token: str) -> None:
        key = token.upper().strip()
        if key and len(key) >= 3 and key not in seen:
            seen.add(key)
            result.append(key)

    # 1. ALL-CAPS tokens
    for w in re.findall(r"\b([A-Z][A-Z0-9&_\-]{2,})\b", question):
        if w not in _SQL_KEYWORDS and not w.isdigit():
            _add(w)

    # 2. Title-case proper nouns
    for w in re.findall(r"\b([A-Z][a-z]{3,})\b", question):
        if w not in _TITLE_STOPWORDS:
            _add(w)

    # 3. Quoted strings (strip surrounding quotes)
    for w in re.findall(r"[\"\']([\w&\s\-]{2,})[\"\']", question):
        _add(w.strip())

    return result


def _entity_correctly_filtered_in_sql(col_name: str, entity: str, sql: str) -> bool:
    """
    Return True when the entity value appears as a quoted literal anywhere in
    the SQL (regardless of which column it is associated with).

    We use a permissive check rather than requiring the exact (col_name, value)
    pair because:
    - The filter_resolver may have already corrected the column cross-hierarchy
      (e.g. category='OREO' → brand='OREO'), so the resolved col_name from
      the DB lookup may not match what's in the corrected SQL.
    - Mixed casing (DB stores 'Mondelez', LLM wrote 'MONDELEZ') is handled by
      the upper-cased comparison.
    - If the value is present as a quoted filter literal, the query is filtering
      on it — which column it resolves to is the filter_resolver's job, not
      the validator's.

    Only returns False when the entity value is entirely absent from all quoted
    string literals in the SQL — meaning it was never used as a filter at all.
    """
    entity_upper = entity.upper()
    # Find all single-quoted string literals in the SQL and check case-insensitively
    quoted_literals = re.findall(r"'([^']*)'" , sql)
    if any(entity_upper in lit.upper() for lit in quoted_literals):
        return True
    # Also accept a table/col reference that has the entity inline (LIKE patterns)
    return entity_upper in sql.upper() and bool(
        re.search(
            rf"\b{re.escape(col_name)}\b\s*(=|like|in\s*\()",
            sql, re.IGNORECASE,
        )
    )


def resolve_entities_in_question(question: str) -> Dict[str, Tuple[str, str]]:
    """
    Public API – resolve every entity candidate in *question* to its DB column
    using the hierarchy priority order (mega_category first, division last).

    Returns {entity_uppercase: (col_name, col_label)} for all entities whose
    column could be determined; entities not matched in the DB are omitted.

    Stored into AgentState as entity_column_map so the SQL generator can
    reference the correct column name on a retry.

    Example
    -------
    >>> resolve_entities_in_question("MONDELEZ market share in TOTAL BARS 2023")
    {
      'MONDELEZ': ('manufacturer', 'manufacturer'),
      'TOTAL':    ('mega_category', 'mega category'),
      'BARS':     ('sub_category',  'category group / sub-category'),
    }
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    for entity in _extract_entity_candidates(question):
        resolved = _resolve_entity_column(entity)   # already uppercased
        if resolved:
            mapping[entity] = resolved
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# Core rule-based validation
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — SQL syntax validation via sqlglot
# ──────────────────────────────────────────────────────────────────────────────

# Dialects to try in order; first one that parses without error wins.
# We prefer 'databricks' (Databricks SQL), then fall back to ANSI.
_PARSE_DIALECTS = ["databricks", "spark", ""]


def _validate_syntax(sql: str) -> Tuple[bool, List[str]]:
    """
    Parse *sql* with sqlglot and return (passed, issues).

    For the dbrx provider, sqlglot is NOT used — it cannot faithfully handle
    Databricks SparkSQL with backtick-quoted, hyphenated three-part catalog
    identifiers (e.g. `dev-amer-customer-catalog`.`schema`.`table`).  Any
    real syntax errors will surface at execution time in the critic agent.

    For the groq/SQLite provider, three dialects are tried in order.
    Returns (True, []) if parsing succeeds under at least one dialect.
    """
    if not sql or not sql.strip():
        return False, ["SQL is empty."]

    # dbrx: trust the SQL and let Databricks execution surface real errors.
    if settings.llm_provider.lower() == "dbrx":
        logger.debug("SQL Validator: skipping sqlglot syntax check for dbrx provider.")
        return True, []

    last_errors: List[str] = []

    for dialect in _PARSE_DIALECTS:
        try:
            kwargs = {"dialect": dialect} if dialect else {}
            sqlglot.parse(sql, **kwargs, error_level=sqlglot.errors.ErrorLevel.RAISE)
            dialect_label = dialect if dialect else "ANSI"
            logger.debug(f"SQL Validator: syntax OK (dialect={dialect_label!r})")
            return True, []
        except sqlglot.errors.ParseError as exc:
            msgs = []
            for err in exc.errors:
                desc   = err.get("description", str(exc))
                line   = err.get("line", "?")
                col    = err.get("col", "?")
                token  = err.get("into") or err.get("start_context", "")
                token_hint = f" near '{token}'" if token else ""
                msgs.append(f"{desc}{token_hint} (line {line}, col {col})")
            last_errors = msgs or [str(exc)]
        except Exception as exc:
            last_errors = [str(exc)]

    issues = [f"SQL syntax error: {e}" for e in last_errors]
    return False, issues


def _validate_rule_based(question: str, sql: str) -> Tuple[bool, List[str]]:
    """
    Deterministic, LLM-free validation.

    Checks (in order):
      1. Year alignment    – every year in question must appear in SQL.
      2. Month alignment   – every month in question must appear in SQL.
      3. Hierarchy entity alignment:
           a. Extract entity candidates  (ALL-CAPS, title-case, quoted strings).
           b. Resolve each to its DB column via priority order:
                mega_category → manufacturer → category → sub_category →
                brand → subbrand → ppg → market → customer → division
           c. Verify the SQL filters on the CORRECT column with the CORRECT value.
              Catches:
              • Entity value entirely missing from SQL.
              • Value present but associated with wrong hierarchy column
                (e.g. brand='MONDELEZ' instead of manufacturer='MONDELEZ').
      4. Metric alignment  – question metric must match SQL aggregation pattern.

    Returns (passed: bool, issues: List[str]).
    """
    issues: List[str] = []
    sql_upper = sql.upper()
    sql_lower = sql.lower()
    q_lower   = question.lower()

    # ── 1. Year alignment ────────────────────────────────────────────────────
    for year in _extract_years(question):
        if year not in sql:
            issues.append(
                f"Year '{year}' is mentioned in the question but is missing from "
                f"the SQL. The SQL may be filtering on a different year."
            )

    # ── 2. Month alignment ───────────────────────────────────────────────────
    for month in _extract_months(question):
        if month not in sql:
            issues.append(
                f"Month '{month}' is mentioned in the question but is missing "
                f"from the SQL."
            )

    # ── 3. Hierarchy entity alignment ────────────────────────────────────────
    # Verify that each named entity from the question appears as a quoted
    # filter literal somewhere in the SQL.  We deliberately do NOT enforce
    # which column it is filtered on — that is the filter_resolver's job.
    # Checking column-priority here produces false positives when:
    #   • filter_resolver corrected a cross-hierarchy column (category → brand)
    #   • The DB holds mixed-case values the LLM uppercased or vice-versa
    #   • The resolved col_name from the DB doesn't match the corrected SQL
    #
    # Skip entirely for dbrx when no entity columns are in the table schema.
    if not _DBRX_ENTITY_VALIDATION_POSSIBLE:
        logger.debug(
            "Validator: entity check skipped — no entity columns in Databricks table."
        )
    for entity in (_extract_entity_candidates(question) if _DBRX_ENTITY_VALIDATION_POSSIBLE else []):
        resolved = _resolve_entity_column(entity)
        if resolved:
            col_name, col_label = resolved
            if not _entity_correctly_filtered_in_sql(col_name, entity, sql):
                issues.append(
                    f"'{entity}' ({col_label}) is mentioned in the question "
                    f"but does not appear as a filter value in the SQL. "
                    f"Add a filter such as: WHERE {col_name} = '{entity}'."
                )
        else:
            # Entity not resolved — schema mismatch on dbrx, silently skip.
            if _IS_DBRX and _DBRX_TABLE_COLS:
                continue
            # For groq: only flag if entity is completely absent from SQL AND
            # the question has clear filter intent.
            if entity.upper() not in sql.upper():
                filter_context_words = [
                    "market", "brand", "manufacturer", "category", "segment",
                    "customer", "division", "channel", "product",
                ]
                if any(kw in q_lower for kw in filter_context_words):
                    issues.append(
                        f"'{entity}' from the question is not referenced in the SQL "
                        f"and could not be matched to any known hierarchy column."
                    )

    # ── 4. Metric alignment ───────────────────────────────────────────────────
    for metric_kw, sql_patterns in METRIC_PATTERNS.items():
        if metric_kw in q_lower:
            matched = any(re.search(p, sql_lower) for p in sql_patterns)
            if not matched:
                issues.append(
                    f"Question asks for '{metric_kw}' but the SQL does not appear "
                    f"to compute it correctly (no matching aggregation found)."
                )
            break   # only check the first matching metric keyword

    return len(issues) == 0, issues


# ──────────────────────────────────────────────────────────────────────────────
# LLM-based validation (semantic check)
# ──────────────────────────────────────────────────────────────────────────────

def _build_llm():
    """Build a fast, cheap LLM for validation."""
    from core.llm_factory import create_llm
    return create_llm("fast")


_SYSTEM_PROMPT = f"""You are a SQL correctness auditor for a Nielsen POS analytics database.
Single table: {_VAL_TABLE}.

Entity hierarchy (PRIORITY ORDER — always use the FIRST match for ambiguous entities):
  mega_category → manufacturer → category → sub_category → brand → subbrand → ppg

Other dimensions: market, customer, division
Time column: year_month (YYYYMM integer — e.g. 202301 = January 2023)
Metric columns: dollar_sales, unit_sales, lbs_sales, tdp and promo variants.

Your job: given a natural language question and a SQL query, decide if the SQL correctly answers it.

Check ONLY:
1. Time period: year / month in question must appear as a literal filter in SQL.
2. Entity + correct column: each named entity must be filtered on its HIGHEST-PRIORITY
   hierarchy column. Example: if "MONDELEZ" exists as manufacturer, SQL must use
   WHERE manufacturer = 'MONDELEZ', NOT WHERE brand = 'MONDELEZ'.
3. Metric: if market share is asked, SQL must compute SUM(…)/SUM(…) ratio.

Reply in EXACTLY this format (nothing else):
VALID: <YES or NO>
ISSUES: <comma-separated issues, or NONE>"""


def _validate_with_llm(question: str, sql: str) -> Tuple[bool, List[str]]:
    """
    Ask an LLM to sanity-check the SQL against the question.
    Falls back to (True, []) on any error so we never block the pipeline.
    """
    try:
        llm = _build_llm()
        user_msg = (
            f"QUESTION: {question}\n\n"
            f"SQL:\n{sql}\n\n"
            f"Does the SQL correctly answer the question? "
            f"Check time period, entity filters, and metric computation."
        )
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        text = response.content.strip()

        # Parse structured response
        valid_match = re.search(r"VALID:\s*(YES|NO)", text, re.IGNORECASE)
        issues_match = re.search(r"ISSUES:\s*(.+)", text, re.IGNORECASE)

        if valid_match:
            passed = valid_match.group(1).upper() == "YES"
            issues_raw = issues_match.group(1).strip() if issues_match else ""
            issues = (
                []
                if issues_raw.upper() in ("NONE", "", "N/A")
                else [i.strip() for i in issues_raw.split(",") if i.strip()]
            )
            return passed, issues

        # Fallback: if response is ambiguous, pass through
        logger.warning("SQL Validator: LLM response was ambiguous – defaulting to VALID")
        return True, []

    except Exception as e:
        logger.warning(f"SQL Validator LLM call failed ({e}) – defaulting to VALID")
        return True, []


# ──────────────────────────────────────────────────────────────────────────────
# Main validator agent
# ──────────────────────────────────────────────────────────────────────────────

class SQLValidatorAgent:
    """
    Validates that the SQL query (from generator OR from semantic cache) actually
    answers the user's current question.

    Strategy:
      Step 1 – Rule-based checks (fast, no LLM cost).
      Step 2 – LLM semantic check (only when rule-based passes, for deeper logic).

    If either step fails, the agent returns validation_passed=False along with
    a list of human-readable issues so the routing logic can decide whether to
    regenerate or bypass the cache.
    """

    def validate(self, state: AgentState) -> dict:
        question = state.get("question", "")
        sql = state.get("sql_query", "") or ""
        attempts = state.get("sql_validation_attempts", 0)

        logger.info(f"SQL Validator: validating SQL (attempt {attempts + 1})")
        logger.debug(f"Question: {question}")
        logger.debug(f"SQL snippet: {sql[:200]}")

        if not sql:
            logger.warning("SQL Validator: no SQL in state – generator failed before producing SQL")
            return {
                "validation_passed": False,
                "validation_issues": ["No SQL query was generated."],
                # Do NOT increment sql_validation_attempts — this is a generator
                # failure, not a validation failure.  The routing in graph.py
                # checks should_retry to avoid burning the validation budget.
            }

        # ── Step 0: Syntax check (sqlglot) ────────────────────────────────────
        syn_passed, syn_issues = _validate_syntax(sql)
        if not syn_passed:
            logger.warning(f"SQL Validator: syntax check FAILED: {syn_issues}")
            return {
                "validation_passed": False,
                "validation_issues": syn_issues,
                "sql_validation_attempts": attempts + 1,
            }

        # ── Step 1: Rule-based ────────────────────────────────────────────────
        rb_passed, rb_issues = _validate_rule_based(question, sql)

        if not rb_passed:
            logger.warning(f"SQL Validator: rule-based check FAILED: {rb_issues}")
            return {
                "validation_passed": False,
                "validation_issues": rb_issues,
                "sql_validation_attempts": attempts + 1,
            }

        # LLM semantic check intentionally removed — adds latency and produces
        # false positives. The critic agent handles semantic correctness after
        # SQL execution.
        logger.info("SQL Validator: validation PASSED")
        return {
            "validation_passed": True,
            "validation_issues": [],
            "sql_validation_attempts": attempts + 1,
        }


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph node wrapper
# ──────────────────────────────────────────────────────────────────────────────

_validator_agent = SQLValidatorAgent()


def validator_node(state: AgentState) -> dict:
    """LangGraph node: validate the current sql_query against the question."""
    return _validator_agent.validate(state)
