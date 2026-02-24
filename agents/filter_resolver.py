"""
Filter Value Resolver — runs AFTER the generator, BEFORE the validator.

Problem:
    LLMs often invent filter values that don't match the actual data in the DB.
    e.g. the question says "MDLZ SINGLES" but the DB stores "MONDELEZ SINGLES PPG".
    This causes valid SQL to return zero rows silently.

Solution:
    1. Parse the generated SQL with sqlglot and extract every  col = 'literal'
       predicate from WHERE clauses (skip numeric literals).
    2. For each (column, guessed_value) pair, fetch DISTINCT values from the DB.
    3. Ask the LLM (in parallel threads) to pick the best match from the real
       values.  The LLM responds with:
           {"match": "<best DB value>", "confidence": 0.0–1.0, "clarify": false}
       or, if it can't find anything sensible:
           {"match": null, "confidence": <low>, "clarify": true,
            "question": "Which <col> did you mean: A, B or C?"}
    4. Auto-correct if confidence ≥ threshold (default 0.55).
       Collect clarification questions for anything below the threshold.
    5. Return corrected SQL.  If clarification is needed, set the state's
       `direct_response` field — the graph routes to `direct_respond` and
       asks the user before running SQL.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import sqlglot
import sqlglot.expressions as exp
from core.llm_factory import create_llm
from loguru import logger

from core.database import db_manager
from core.state import AgentState

# ── tunables ──────────────────────────────────────────────────────────────────
# Auto-correct when LLM confidence is at or above this threshold; otherwise ask.
_AUTO_CORRECT_THRESHOLD = 0.55

# Maximum distinct values fed to the LLM per column (keeps prompt manageable).
_MAX_DISTINCT = 300

# System columns that are always numeric / IDs — never string-matched.
_SKIP_COLUMNS = {
    "year_nielsen", "year", "period_num", "week_num", "month_num",
    "id", "row_id", "record_id",
}

# ─────────────────────────────────────────────────────────────────────────────
# SQL parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_string_filters(sql: str) -> List[Tuple[str, str]]:
    """
    Return a list of (column_name, literal_value) pairs for every
    ``col = 'string'``  (or  col IN ('a','b','c'))  predicate that appears in
    the WHERE clauses of the query.  Numeric literals are ignored.
    """
    results: List[Tuple[str, str]] = []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception as exc:
        logger.warning(f"FilterResolver: could not parse SQL — {exc}")
        return results

    # Walk every EQ node: col = 'value'
    for node in tree.find_all(exp.EQ):
        left, right = node.left, node.right
        col_name = None
        lit_val = None

        if isinstance(left, exp.Column):
            col_name = left.name.lower()
        if isinstance(right, exp.Literal) and right.is_string:
            lit_val = right.this

        if col_name and lit_val and col_name not in _SKIP_COLUMNS:
            results.append((col_name, lit_val))

    # Walk IN lists: col IN ('a', 'b')
    for node in tree.find_all(exp.In):
        col_node = node.this
        if not isinstance(col_node, exp.Column):
            continue
        col_name = col_node.name.lower()
        if col_name in _SKIP_COLUMNS:
            continue
        for expr in node.expressions:
            if isinstance(expr, exp.Literal) and expr.is_string:
                results.append((col_name, expr.this))

    # De-duplicate while preserving order
    seen: set = set()
    unique = []
    for item in results:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _rewrite_sql(sql: str, corrections: Dict[Tuple[str, str], str]) -> str:
    """
    Replace filter literals in the SQL string.
    ``corrections`` maps (column, old_value) → new_value.
    We do a careful string replacement that respects SQL quoting.
    """
    if not corrections:
        return sql

    result = sql
    for (col, old_val), new_val in corrections.items():
        if old_val == new_val:
            continue
        # Replace  col = 'old'  →  col = 'new'  (case-insensitive column match)
        pattern = re.compile(
            r"(?i)(\b" + re.escape(col) + r"\s*=\s*')(" + re.escape(old_val) + r")(')",
        )
        result = pattern.sub(lambda m: m.group(1) + new_val + m.group(3), result)

        # Also handle IN lists:  'old_val'  →  'new_val'
        result = result.replace(f"'{old_val}'", f"'{new_val}'")

    return result


# ── Column synonym map ───────────────────────────────────────────────────────
# Maps common invented column names → the real DB column they likely represent.
# Used to generate intelligent suggestions when the LLM hallucinates a column.
_COLUMN_SYNONYMS: Dict[str, str] = {
    # ── segment / category family ──────────────────────────────────────────
    "segment":           "category",
    "product_segment":   "category",
    "segments":          "category",
    "segment_name":      "category",
    "product_category":  "category",
    "item_category":     "category",
    "category_name":     "category",
    # ── sub_category ──────────────────────────────────────────────────────
    "sub_segment":       "sub_category",
    "subsegment":        "sub_category",
    "subcategory":       "sub_category",
    # ── manufacturer ──────────────────────────────────────────────────────
    "company":           "manufacturer",
    "manufacturer_name": "manufacturer",
    "mfr":               "manufacturer",
    "supplier":          "manufacturer",
    "vendor":            "manufacturer",
    "maker":             "manufacturer",
    # ── brand ─────────────────────────────────────────────────────────────
    "product":           "brand",
    "brand_name":        "brand",
    "item":              "brand",
    "sku":               "brand",
    "product_name":      "brand",
    # ── subbrand ──────────────────────────────────────────────────────────
    "sub_brand":         "subbrand",
    "subbrand_name":     "subbrand",
    "variant":           "subbrand",
    "sub_brand_name":    "subbrand",
    # ── market / geography ────────────────────────────────────────────────
    "region":            "market",
    "geography":         "market",
    "geo":               "market",
    "market_name":       "market",
    "area":              "market",
    "territory":         "market",
    # ── customer / retailer ───────────────────────────────────────────────
    "retailer":          "customer",
    "store":             "customer",
    "account":           "customer",
    "channel":           "customer",
    "customer_name":     "customer",
    "outlet":            "customer",
    # ── mega_category ─────────────────────────────────────────────────────
    "department":        "mega_category",
    "mega_cat":          "mega_category",
    "super_category":    "mega_category",
    "aisle":             "mega_category",
    # ── division ──────────────────────────────────────────────────────────
    "division_name":     "division",
    # ── ppg ───────────────────────────────────────────────────────────────
    "ppg_name":          "ppg",
    "product_group":     "ppg",
    "planning_group":    "ppg",
}

_KNOWN_DB_COLUMNS: set = {
    "mega_category", "manufacturer", "category", "sub_category",
    "brand", "subbrand", "ppg", "market", "customer", "division",
}


def _suggest_column(col_name: str) -> Optional[str]:
    """
    Given an invented / hallucinated column name, return the most likely real
    DB column.  Checks the synonym table first, then falls back to substring
    matching against known columns.
    """
    key = col_name.lower().strip()

    # 1. Direct synonym lookup
    if key in _COLUMN_SYNONYMS:
        return _COLUMN_SYNONYMS[key]

    # 2. Alias appears inside the invented name  (e.g. "product_segment" ∋ "segment")
    for alias, real in _COLUMN_SYNONYMS.items():
        if alias in key:
            return real

    # 3. Known DB column name appears inside the invented name
    for real_col in _KNOWN_DB_COLUMNS:
        if real_col in key:
            return real_col

    return None


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_distinct(column: str, table: str = "nielsen_pos") -> List[str]:
    """Fetch up to _MAX_DISTINCT unique non-null values for a column."""
    sql = (
        f"SELECT DISTINCT {column} FROM {table} "
        f"WHERE {column} IS NOT NULL AND {column} != '' "
        f"LIMIT {_MAX_DISTINCT}"
    )
    rows, err, _ = db_manager.execute_query(sql)
    if err or not rows:
        return []
    return [str(r[column]) for r in rows if r.get(column)]


# ─────────────────────────────────────────────────────────────────────────────
# LLM resolver
# ─────────────────────────────────────────────────────────────────────────────

_RESOLVER_SYSTEM = """\
You are a database filter-value matcher for a Nielsen retail analytics system.

Given:
- The column name
- The value the user/LLM wrote in the SQL query ("guessed value")
- A list of REAL values that exist in the database for that column

Your task: identify which real database value best matches the guessed value.

Rules:
- Prioritise semantic / abbreviation matches (e.g. "MDLZ" → "MONDELEZ",
  "Northeast" → "NE", "Total Bars" → "TOTAL CHOCOLATE BARS", etc.).
- Consider common Nielsen abbreviations and category naming conventions.
- If two or more values are equally plausible AND the distinction is business-critical,
  set "clarify": true and compose a short clarifying question.
- If there is simply no reasonable match (nothing close), set "clarify": true.
- Return ONLY a JSON object — no prose, no markdown.

JSON schema:
{
  "match": "<best matching DB value or null>",
  "confidence": <0.0 – 1.0>,
  "clarify": <true | false>,
  "question": "<clarification question if clarify=true, else null>"
}
"""

_RESOLVER_USER = """\
Column     : {column}
SQL value  : {guessed}
DB values  : {db_values}

Return the JSON object now.
"""


def _resolve_one(
    llm,
    column: str,
    guessed: str,
    db_values: List[str],
) -> dict:
    """Ask the LLM to resolve a single filter value.  Returns parsed JSON dict."""
    if not db_values:
        # Column doesn't exist — try to suggest the closest real column.
        suggested = _suggest_column(column)
        if suggested:
            suggested_values = _fetch_distinct(suggested)
            if suggested_values:
                # Try exact match first
                lower_map = {v.lower(): v for v in suggested_values}
                if guessed.lower() in lower_map:
                    best = lower_map[guessed.lower()]
                    return {
                        "match": guessed, "confidence": 0.0, "clarify": True,
                        "question": (
                            f"There's no column named **{column}** in the database — "
                            f"did you mean **{suggested}** = '{best}'?"
                        ),
                    }
                # Ask LLM to find best match in the suggested column
                values_str = "\n".join(f"  - {v}" for v in suggested_values[:_MAX_DISTINCT])
                try:
                    resp = llm.invoke([
                        {"role": "system", "content": _RESOLVER_SYSTEM},
                        {"role": "user",   "content": _RESOLVER_USER.format(
                            column=suggested, guessed=guessed, db_values=values_str
                        )},
                    ])
                    raw = resp.content.strip()
                    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                    raw = raw.replace("```", "").strip()
                    result = json.loads(raw)
                    best  = result.get("match") or guessed
                    conf  = float(result.get("confidence", 0.0))
                    question = (
                        f"There's no column named **{column}** in the database. "
                        f"Did you mean **{suggested}** = '{best}'?"
                        + (f" (confidence: {conf:.0%})" if conf < 0.9 else "")
                    )
                    return {
                        "match": guessed, "confidence": 0.0, "clarify": True,
                        "question": question,
                    }
                except Exception as exc:
                    logger.warning(f"FilterResolver: suggestion LLM failed — {exc}")
                    return {
                        "match": guessed, "confidence": 0.0, "clarify": True,
                        "question": (
                            f"There's no column named **{column}** in the database — "
                            f"did you mean **{suggested}**?"
                        ),
                    }
        # No suggestion found — list available columns
        cols = ", ".join(sorted(_KNOWN_DB_COLUMNS))
        return {
            "match": guessed, "confidence": 0.0, "clarify": True,
            "question": (
                f"I couldn't find a column named **{column}** in the database. "
                f"Available filter columns are: {cols}."
            ),
        }

    # If guessed already in db_values (exact, case-insensitive) → skip LLM
    lower_map = {v.lower(): v for v in db_values}
    if guessed.lower() in lower_map:
        exact = lower_map[guessed.lower()]
        logger.debug(f"FilterResolver: '{column}'='{guessed}' exact match → '{exact}'")
        return {"match": exact, "confidence": 1.0, "clarify": False, "question": None}

    values_str = "\n".join(f"  - {v}" for v in db_values[:_MAX_DISTINCT])
    try:
        resp = llm.invoke([
            {"role": "system", "content": _RESOLVER_SYSTEM},
            {"role": "user",   "content": _RESOLVER_USER.format(
                column=column, guessed=guessed, db_values=values_str
            )},
        ])
        raw = resp.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = raw.replace("```", "").strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning(f"FilterResolver: LLM call failed for {column}='{guessed}' — {exc}")
        return {"match": guessed, "confidence": 0.0, "clarify": True,
                "question": f"I couldn't verify the value '{guessed}' for '{column}'."}


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

class FilterResolverAgent:
    """
    Resolves SQL filter literals against actual DB values using parallel LLM calls.
    """

    def __init__(self):
        self.llm = create_llm("reasoning")

    def resolve(self, state: AgentState) -> dict:
        sql = state.get("sql_query", "")
        if not sql:
            return {}

        logger.info("FilterResolver: scanning SQL for filter values to verify")

        filters = _extract_string_filters(sql)
        if not filters:
            logger.info("FilterResolver: no string filter literals found — skipping")
            return {}

        logger.info(f"FilterResolver: {len(filters)} filter(s) to verify: "
                    + ", ".join(f"{c}='{v}'" for c, v in filters))

        # Fetch DB distinct values for unique columns (batch)
        unique_cols = list({col for col, _ in filters})
        db_map: Dict[str, List[str]] = {}
        for col in unique_cols:
            db_map[col] = _fetch_distinct(col)
            logger.debug(f"FilterResolver: {col} → {len(db_map[col])} distinct values")

        # Resolve all filters in parallel
        corrections: Dict[Tuple[str, str], str] = {}
        clarifications: List[str] = []
        filter_log: List[Dict] = []

        with ThreadPoolExecutor(max_workers=min(len(filters), 6)) as pool:
            futures = {
                pool.submit(
                    _resolve_one, self.llm, col, val, db_map.get(col, [])
                ): (col, val)
                for col, val in filters
            }
            for future in as_completed(futures):
                col, val = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error(f"FilterResolver: resolution failed for {col}='{val}': {exc}")
                    result = {"match": val, "confidence": 0.0, "clarify": False, "question": None}

                match      = result.get("match") or val
                confidence = float(result.get("confidence", 0.0))
                clarify    = result.get("clarify", False)
                question   = result.get("question")

                log_entry = {
                    "column": col, "sql_value": val,
                    "db_match": match, "confidence": confidence,
                }

                if clarify or confidence < _AUTO_CORRECT_THRESHOLD:
                    logger.warning(
                        f"FilterResolver: '{col}'='{val}' — "
                        f"low confidence ({confidence:.2f}), needs clarification"
                    )
                    if question:
                        clarifications.append(question)
                    log_entry["action"] = "clarify"
                else:
                    if match != val:
                        logger.info(
                            f"FilterResolver: '{col}' '{val}' → '{match}' "
                            f"(confidence {confidence:.2f}) — auto-corrected"
                        )
                        corrections[(col, val)] = match
                        log_entry["action"] = "corrected"
                    else:
                        log_entry["action"] = "unchanged"

                filter_log.append(log_entry)

        # Apply corrections to SQL
        corrected_sql = _rewrite_sql(sql, corrections) if corrections else sql

        # Log summary
        n_corrected  = sum(1 for e in filter_log if e["action"] == "corrected")
        n_unchanged  = sum(1 for e in filter_log if e["action"] == "unchanged")
        n_clarify    = sum(1 for e in filter_log if e["action"] == "clarify")
        logger.info(
            f"FilterResolver done: {n_corrected} corrected, "
            f"{n_unchanged} unchanged, {n_clarify} need clarification"
        )

        out: dict = {"filter_log": filter_log}

        if corrections:
            out["sql_query"] = corrected_sql

        if clarifications:
            # Build a single readable clarification message
            intro = (
                "Before I run the query, I need a quick clarification:\n\n"
                + "\n".join(f"• {q}" for q in clarifications)
            )
            out["direct_response"] = intro
            out["needs_clarification"] = True
            # Store context so interaction_node on the NEXT turn can auto-route
            # to correction without re-classifying intent.
            out["pending_filter_clarification"] = [
                {
                    "column":         e["column"],
                    "sql_value":      e["sql_value"],
                    "db_match":       e.get("db_match"),
                    "clarification_q": clarifications[
                        sum(1 for x in filter_log[:filter_log.index(e)]
                            if x.get("action") == "clarify")
                    ] if e.get("action") == "clarify" else None,
                }
                for e in filter_log if e.get("action") == "clarify"
            ]

        return out


# ── LangGraph node ────────────────────────────────────────────────────────────

def filter_resolver_node(state: AgentState) -> dict:
    """LangGraph node wrapper for FilterResolverAgent."""
    return FilterResolverAgent().resolve(state)
