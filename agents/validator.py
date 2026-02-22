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

from core.state import AgentState
from config import settings


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

# ── Metric keyword → SQL patterns that should be present ─────────────────────
METRIC_PATTERNS: Dict[str, List[str]] = {
    "market share": [r"sum\s*\(", r"/\s*sum\s*\(", r"market_share"],
    "market_share": [r"sum\s*\(", r"/\s*sum\s*\(", r"market_share"],
    "tdp":          [r"\btdp\b",  r"avg\s*\("],
    "velocity":     [r"velocity", r"/\s*tdp", r"sales.*per.*tdp"],
    "distribution": [r"\btdp\b",  r"distribution"],
    "sales":        [r"sales",    r"sum\s*\("],
    "volume":       [r"volume",   r"units", r"sum\s*\("],
    "revenue":      [r"revenue",  r"sales", r"sum\s*\("],
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
            sql = (
                f"SELECT 1 FROM nielsen_pos "
                f"WHERE UPPER(CAST({col} AS TEXT)) = '{safe}' "
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
    Return True when the SQL contains a WHERE-style filter using BOTH:
      • the correct column name  (e.g.  manufacturer)
      • the entity value         (e.g.  MONDELEZ)

    Matches patterns like:
        WHERE manufacturer = 'MONDELEZ'
        WHERE  manufacturer LIKE '%MONDELEZ%'
        WHERE  manufacturer IN ('MONDELEZ', ...)

    Two-step check:
      a. Quick gate: is the entity value present anywhere in the SQL?
      b. Is col_name used in a comparison / membership expression?
    """
    if entity.upper() not in sql.upper():
        return False   # value completely absent

    # col_name followed by a comparison or membership operator
    col_pattern = rf"\b{re.escape(col_name)}\b\s*(=|like|in\s*\()"
    return bool(re.search(col_pattern, sql, re.IGNORECASE))


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

    Tries the Databricks dialect first (matching our target execution engine),
    then Spark SQL, then generic ANSI.  A query that cannot be parsed under any
    dialect returns a concrete error message pinpointing the problematic token
    and line/column position so the generator can fix it.

    Returns (True, []) if parsing succeeds under at least one dialect.
    """
    if not sql or not sql.strip():
        return False, ["SQL is empty."]

    last_errors: List[str] = []

    for dialect in _PARSE_DIALECTS:
        try:
            kwargs = {"dialect": dialect} if dialect else {}
            sqlglot.parse(sql, **kwargs, error_level=sqlglot.errors.ErrorLevel.RAISE)
            # Parsed successfully — report the winning dialect for transparency
            dialect_label = dialect if dialect else "ANSI"
            logger.debug(f"SQL Validator: syntax OK (dialect={dialect_label!r})")
            return True, []
        except sqlglot.errors.ParseError as exc:
            msgs = []
            for err in exc.errors:
                # Each error dict contains: description, line, col, start_context, ...
                desc   = err.get("description", str(exc))
                line   = err.get("line", "?")
                col    = err.get("col", "?")
                token  = err.get("into") or err.get("start_context", "")
                token_hint = f" near '{token}'" if token else ""
                msgs.append(f"{desc}{token_hint} (line {line}, col {col})")
            last_errors = msgs or [str(exc)]
        except Exception as exc:          # safeguard – never block the pipeline
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
    # Entities that exist in multiple columns (e.g. MONDELEZ as manufacturer
    # AND brand) always resolve to the FIRST matching column in the hierarchy
    # so the SQL must filter on that highest-priority column.
    for entity in _extract_entity_candidates(question):
        resolved = _resolve_entity_column(entity)   # already uppercased

        if resolved:
            col_name, col_label = resolved
            correctly_filtered = _entity_correctly_filtered_in_sql(col_name, entity, sql)

            if not correctly_filtered:
                entity_in_sql = entity.upper() in sql_upper
                if not entity_in_sql:
                    issues.append(
                        f"'{entity}' ({col_label}) is mentioned in the question "
                        f"but is missing from the SQL. "
                        f"Expected: WHERE {col_name} = '{entity}'."
                    )
                else:
                    # Value present but under the wrong column (hierarchy mismatch)
                    issues.append(
                        f"'{entity}' appears in the SQL but is NOT filtered on "
                        f"the correct column '{col_name}' ({col_label}). "
                        f"In the Nielsen hierarchy '{entity}' resolves to "
                        f"{col_label} — use: WHERE {col_name} = '{entity}'."
                    )
        else:
            # Entity not found in any DB column; only flag in a filter context
            entity_in_sql = entity.upper() in sql_upper
            filter_context_words = [
                "market", "brand", "manufacturer", "category", "segment",
                "customer", "division", "channel", "product",
            ]
            if not entity_in_sql and any(kw in q_lower for kw in filter_context_words):
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
    """Build a fast, cheap LLM for validation (uses the groq_model_fast slot)."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_model_fast,
        groq_api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=256,
    )


_SYSTEM_PROMPT = """You are a SQL correctness auditor for a Nielsen POS analytics database.
Single table: nielsen_pos.

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

Reply in EXACTLY this format — no markdown, no extra text, no explanation, nothing else:
VALID: YES
ISSUES: NONE

or

VALID: NO
ISSUES: <comma-separated list of issues>

Do NOT include any other words, punctuation, or formatting outside these two lines."""


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

        # Strip markdown bold/italic formatting that some LLMs add
        clean_text = re.sub(r"[*_`]{1,3}", "", text)

        # Parse structured response
        valid_match = re.search(r"VALID:\s*(YES|NO)", clean_text, re.IGNORECASE)
        issues_match = re.search(r"ISSUES:\s*(.+)", clean_text, re.IGNORECASE | re.DOTALL)

        if valid_match:
            passed = valid_match.group(1).upper() == "YES"
            issues_raw = issues_match.group(1).strip().split("\n")[0] if issues_match else ""
            issues = (
                []
                if issues_raw.upper() in ("NONE", "", "N/A")
                else [i.strip() for i in issues_raw.split(",") if i.strip()]
            )
            return passed, issues

        # Secondary fallback: look for standalone YES / NO anywhere in response
        yes_match = re.search(r"\byes\b", clean_text, re.IGNORECASE)
        no_match = re.search(r"\bno\b", clean_text, re.IGNORECASE)
        if no_match and not yes_match:
            logger.debug(f"SQL Validator: parsed fallback NO from response: {text!r}")
            return False, ["SQL may not correctly answer the question (ambiguous LLM response)"]
        if yes_match and not no_match:
            logger.debug(f"SQL Validator: parsed fallback YES from response: {text!r}")
            return True, []

        # Last resort: log at DEBUG (not WARNING) and default to VALID
        logger.debug(f"SQL Validator: response unparseable, defaulting to VALID. Raw: {text!r}")
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

        # ── Step 2: LLM semantic check ────────────────────────────────────────
        llm_passed, llm_issues = _validate_with_llm(question, sql)

        if not llm_passed:
            logger.warning(f"SQL Validator: LLM check FAILED: {llm_issues}")
            return {
                "validation_passed": False,
                "validation_issues": llm_issues,
                "sql_validation_attempts": attempts + 1,
            }

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
