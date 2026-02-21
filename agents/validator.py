"""
SQL Validation Agent for the DRGC Text-to-SQL pipeline.

Validates that the generated (or cache-retrieved) SQL correctly answers
the user's question along multiple semantic dimensions:
  - Time period alignment  (year / month in question must appear in SQL)
  - Named-entity alignment (ALL-CAPS brands/markets must appear in SQL)
  - Metric alignment       (market_share, TDP, velocity patterns)

If validation fails the node signals: regenerate (fresh SQL) or bypass_cache
(cache hit that was wrong → full re-plan).  Includes an attempt counter to
prevent infinite loops – after max_validation_attempts the SQL is forwarded
to the executor unchanged.
"""

import re
from typing import Optional, List, Tuple

from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from core.state import AgentState
from config import settings


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_VALIDATION_ATTEMPTS = 2   # after this many failures, forward SQL anyway

# Metric keywords and the SQL patterns that should appear when they are asked
METRIC_PATTERNS = {
    "market share": [r"sum\s*\(", r"/\s*sum\s*\(", r"total_sales", r"market_share"],
    "market_share": [r"sum\s*\(", r"/\s*sum\s*\(", r"total_sales", r"market_share"],
    "tdp":          [r"tdp", r"total distribution points", r"avg\s*\("],
    "velocity":     [r"velocity", r"sales.*per.*tdp", r"/\s*tdp"],
    "distribution": [r"distribution", r"tdp", r"count\s*\("],
    "sales":        [r"sales", r"sum\s*\("],
    "volume":       [r"volume", r"units", r"sum\s*\("],
    "revenue":      [r"revenue", r"sales", r"sum\s*\("],
}

# Month name → numeric string
MONTH_MAP = {
    "january": "01", "february": "02", "march": "03",
    "april": "04",   "may": "05",      "june": "06",
    "july": "07",    "august": "08",   "september": "09",
    "october": "10", "november": "11", "december": "12",
    "jan": "01",     "feb": "02",      "mar": "03",
    "apr": "04",                        "jun": "06",
    "jul": "07",     "aug": "08",      "sep": "09",
    "oct": "10",     "nov": "11",      "dec": "12",
}


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_years(text: str) -> List[str]:
    """Return 4-digit years (2000-2030) found in *text*."""
    return re.findall(r"\b(20[0-2][0-9])\b", text)


def _extract_months(text: str) -> List[str]:
    """Return 2-digit month strings (e.g. '03') found by name or number."""
    text_lower = text.lower()
    months: List[str] = []
    for name, num in MONTH_MAP.items():
        if re.search(r"\b" + name + r"\b", text_lower):
            months.append(num)
    # bare numeric months like "month 3" or "Q1/2/3/4 → quarters not months"
    for m in re.findall(r"\bmonth\s+(\d{1,2})\b", text_lower):
        months.append(m.zfill(2))
    return list(set(months))


def _extract_capswords(text: str) -> List[str]:
    """
    Extract ALL-CAPS tokens (≥3 chars) – typically brand/market/category names
    in Nielsen data.  Ignore pure numbers and common SQL keywords.
    """
    SQL_KEYWORDS = {
        "SELECT", "FROM", "WHERE", "WITH", "AS", "ON", "AND", "OR",
        "NOT", "IN", "IS", "NULL", "BY", "ORDER", "GROUP", "HAVING",
        "LEFT", "RIGHT", "JOIN", "INNER", "OUTER", "UNION", "ALL",
        "CASE", "WHEN", "THEN", "ELSE", "END", "LIKE", "SUM", "AVG",
        "COUNT", "MIN", "MAX", "OVER", "PARTITION", "BETWEEN", "CTE",
        "DESC", "ASC", "LIMIT", "DISTINCT", "TOP", "INTO", "SET",
    }
    words = re.findall(r"\b([A-Z][A-Z0-9&_\-]{2,})\b", text)
    return [w for w in words if w not in SQL_KEYWORDS and not w.isdigit()]


def _extract_question_entities(question: str) -> dict:
    """
    Pull discriminating tokens from the question string.
    Returns a dict with keys: years, months, caps_entities.
    """
    return {
        "years":        _extract_years(question),
        "months":       _extract_months(question),
        "caps_entities": list(set(_extract_capswords(question))),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based SQL validation
# ──────────────────────────────────────────────────────────────────────────────

def _validate_rule_based(
    question: str,
    sql: str,
) -> Tuple[bool, List[str]]:
    """
    Fast, deterministic validation – no LLM needed.

    Checks:
      1. Every year mentioned in the question appears somewhere in the SQL.
      2. Every month mentioned in the question appears somewhere in the SQL.
      3. Every ALL-CAPS entity in the question appears somewhere in the SQL.
      4. Metric type requested is reflected in the SQL.

    Returns (passed: bool, issues: List[str]).
    """
    issues: List[str] = []
    sql_upper = sql.upper()
    sql_lower = sql.lower()
    entities = _extract_question_entities(question)

    # ── 1. Year alignment ────────────────────────────────────────────────────
    for year in entities["years"]:
        if year not in sql:
            issues.append(
                f"Year '{year}' mentioned in the question is missing from the SQL. "
                f"The SQL may be filtering on a different year."
            )

    # ── 2. Month alignment ───────────────────────────────────────────────────
    for month in entities["months"]:
        # month string e.g. "03" should appear in year_month like '202303'
        if month not in sql:
            issues.append(
                f"Month '{month}' mentioned in the question is missing from the SQL."
            )

    # ── 3. Named-entity alignment ─────────────────────────────────────────────
    q_lower = question.lower()
    for entity in entities["caps_entities"]:
        # Skip very generic tokens that legitimately may not appear in WHERE clause
        if len(entity) < 4:
            continue
        # The entity should appear in the SQL (case-insensitive)
        if entity.upper() not in sql_upper:
            # Secondary: if entity appears in question as a proper noun → required
            # Only flag if it's part of a filter-like context (brand, market, etc.)
            entity_lower = entity.lower()
            filter_context_keywords = [
                "market", "brand", "manufacturer", "category", "segment",
                "customer", "division", "channel", "product", "flavor",
            ]
            in_filter_context = any(
                kw in q_lower for kw in filter_context_keywords
            )
            if in_filter_context:
                issues.append(
                    f"Entity '{entity}' from the question is not referenced in the SQL. "
                    f"The SQL may be using a different or incorrect filter value."
                )

    # ── 4. Metric alignment ───────────────────────────────────────────────────
    q_lower = question.lower()
    for metric_kw, sql_patterns in METRIC_PATTERNS.items():
        if metric_kw in q_lower:
            matched = any(re.search(p, sql_lower) for p in sql_patterns)
            if not matched:
                issues.append(
                    f"Question asks for '{metric_kw}' but the SQL does not appear "
                    f"to compute it correctly (no matching aggregation found)."
                )
            break  # only check first matching metric

    passed = len(issues) == 0
    return passed, issues


# ──────────────────────────────────────────────────────────────────────────────
# LLM-based validation (semantic check)
# ──────────────────────────────────────────────────────────────────────────────

def _build_llm():
    """Build a fast, cheap LLM for validation (uses the 'fast' model slot)."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.fast_model,
        openai_api_base=settings.llm_api_base,
        openai_api_key=settings.llm_api_key,
        temperature=0,
        max_tokens=256,
    )


_SYSTEM_PROMPT = """You are a SQL correctness auditor for a Nielsen POS (point-of-sale) analytics database.
The database has a single table: nielsen_pos.
Important columns: year_month (YYYYMM format), manufacturer, brand, market, segment, category, customer, tdp, dollar_sales.

Your job: given a natural language question and a SQL query, decide if the SQL correctly answers the question.

Focus ONLY on these checks:
1. Time period: if the question specifies a year or month, the SQL must filter on it.
2. Entity filters: if the question names a brand/market/manufacturer/category, the SQL must filter on it.
3. Metric: if the question asks for market share, the SQL must compute a ratio (e.g. SUM/SUM).

Reply in this EXACT format – nothing else:
VALID: <YES or NO>
ISSUES: <comma-separated list of issues, or NONE>"""


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
            logger.warning("SQL Validator: no SQL in state – skipping validation")
            return {
                "validation_passed": False,
                "validation_issues": ["No SQL query was generated."],
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
