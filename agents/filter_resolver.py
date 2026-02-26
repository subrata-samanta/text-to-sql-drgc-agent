"""
Filter Value Resolver -- runs AFTER the generator, BEFORE the validator.

Problem
-------
LLMs often invent filter values that don't match actual data in the DB.
More subtly, a query for "candy segment" may produce `category = 'candy'`
when the correct answer is `mega_category = 'NON CHOCOLATE CANDY'`.

Solution -- Intent-First, Hierarchy-Aware Resolution
-----------------------------------------------------
For product-hierarchy columns (mega_category, manufacturer, category,
sub_category, brand, subbrand, ppg) the resolver runs 4 phases:

  Phase 1 -- Granularity Intent Classification
    Determine the intended hierarchy level from linguistic signals in the
    user's question ("segment" -> mega_category; "brand" -> brand; etc.).
    A fast keyword scan runs first; the LLM is called as a fallback.

  Phase 2 -- Hierarchy-Aware Scoring
    Fetch distinct DB values for EVERY hierarchy level. Score each
    candidate as:
        combined = 0.4 x string_similarity + 0.6 x level_alignment
    Level-alignment weight dominates so we don't always fall to the lowest
    textually-matching granularity.

  Phase 3 -- Consistency & Contextual Override
    If the top string match is at a finer level than the intended level,
    but a coarser-level value exceeds the minimum string-similarity
    threshold, prefer the coarser level (e.g. "candy segment" -> prefer
    mega_category over category even if "candy" scores higher than
    "non chocolate candy").

  Phase 4 -- Ambiguity Fallback
    When the top-2 candidates are within 0.10 of each other and are at
    different levels, default to the coarser-level candidate and surface
    the interpretation to the user ("Interpreted 'candy' as
    mega_category='NON CHOCOLATE CANDY'. Let me know if you meant a more
    specific level.").

For non-hierarchy columns (market, customer, division, total, etc.) the
original single-level LLM fuzzy matcher is used unchanged.

SQL rewriting
-------------
The rewriter handles both value swaps (same column) and cross-column
corrections (column rename + value swap) when the best match is found in
a different hierarchy level than the one the SQL was written against.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import sqlglot
import sqlglot.expressions as exp
from core.llm_factory import create_llm
from loguru import logger

from config import settings
from core.database import db_manager
from core.state import AgentState

# ── tunables ──────────────────────────────────────────────────────────────────
_AUTO_CORRECT_THRESHOLD       = 0.50   # minimum combined score to auto-correct
_HIERARCHY_AMBIGUITY_MARGIN   = 0.10   # top-2 within this -> pick coarser level
_MIN_STRING_SCORE_FOR_COARSER = 0.30   # Phase 3 override: coarser beats finer above this
_MAX_DISTINCT                 = 1000   # max DB values per column
# String-similarity weight dominates so exact/near-exact matches at a different
# hierarchy level win over poor-string-match same-level candidates.
_STRING_WEIGHT                = 0.65   # weight for string-similarity in combined score
_LEVEL_WEIGHT                 = 0.35   # weight for level-alignment in combined score
# Minimum string similarity for a cross-level candidate to be auto-promoted
# (avoids promoting weak fuzzy matches just because they sit at the right level).
_CROSS_LEVEL_EXACT_THRESHOLD  = 0.88   # used in the cross-level exact-match guard
_SKIP_COLUMNS: set = {
    "year_nielsen", "year", "period_num", "week_num", "month_num",
    "id", "row_id", "record_id",
}

# ── Hard-coded business-logic filter overrides ────────────────────────────────
# When the user's question contains ANY of the ``question_keywords`` AND a SQL
# filter value contains ANY of the ``value_keywords``, the filter is
# immediately rewritten to (``force_column``, ``force_value``) with no LLM
# call.  Add new rules here to encode additional business knowledge.
#
# Matching is always case-insensitive.  The check on ``value_keywords`` avoids
# triggering the rule when the SQL already contains the correct canonical value.
#
# Rule fields:
#   question_keywords  — at least one must appear in the user question
#   value_keywords     — at least one must appear in the SQL filter literal
#                        (use [""] to match ANY non-empty value)
#   force_column       — canonical DB column to write
#   force_value        — canonical DB value to write (exact, case-sensitive)
_BUSINESS_RULE_OVERRIDES: List[Dict] = [
    # ── Keep ONLY rules where the correct DB value is COUNTER-INTUITIVE even
    # to a CPG-savvy LLM (i.e. a perfect expansion still can't predict the
    # exact Nielsen canonical string or the target column).
    #
    # Everything else (abbreviations, misspellings, wrong-column placements
    # that a CPG expert would recognise) is now handled dynamically by
    # _llm_expand_entity — no code changes needed for new product names.
    {
        # "candy" almost always means the NON CHOCOLATE *category* in Nielsen
        # nomenclature.  LLMs consistently hallucinate the value as 'CANDY',
        # 'CANDY BARS', 'CONFECTIONERY', etc.  This mapping is genuinely
        # counter-intuitive without deep Nielsen schema knowledge.
        "question_keywords": ["candy"],
        "value_keywords":    ["candy", "non chocolate", "nonchocolate"],
        "force_column":      "category",
        "force_value":       "NON CHOCOLATE",
    },
]

# ── Known Nielsen abbreviation / misspelling seed cache ──────────────────────
# Pre-seeds the LLM expansion cache so frequently-seen abbreviations cost zero
# LLM calls.  The LLM handles anything NOT present here automatically.
#
# Format — lowercase key : (canonical_expansion, preferred_column_hint | None)
# The preferred_column_hint is propagated to the hierarchy resolver so it
# searches the right axis without relying on the (often wrong) SQL column.
#
# Add new entries here only for values the LLM consistently gets wrong due to
# extremely terse abbreviations or Nielsen-specific naming quirks.  For the
# vast majority of new CPG names, the LLM will resolve them correctly on its
# own without any code changes.
_ABBREVIATION_EXPANSIONS: Dict[str, Tuple[str, Optional[str]]] = {
    # value (lower)              : (expanded_canonical,    preferred_column_hint)
    "spk"                       : ("SOUR PATCH KIDS",         "subbrand"),
    "swf"                       : ("SWEDISH FISH",             "subbrand"),
    "swf tropical"              : ("SWEDISH FISH TROPICAL",    "subbrand"),
    "mdlz"                      : ("MONDELEZ",                 "manufacturer"),
    "nat valley"                : ("NATURE VALLEY",            "brand"),
    "nat. valley"               : ("NATURE VALLEY",            "brand"),
    "nature valley"             : ("NATURE VALLEY",            "brand"),
    "conify"                    : ("CONFECTIONARY",            "mega_category"),
    "confectionery"             : ("CONFECTIONARY",            "mega_category"),
    "nutter butter"             : ("NAB NUTTER BUTTER",        "brand"),
    "wheat thins"               : ("NAB WHEAT THINS",          "brand"),
    "nabisco wheat thins"       : ("NAB WHEAT THINS",          "brand"),
    "halls cough drops"         : ("HALLS",                    "brand"),
    "halls cough drop"          : ("HALLS",                    "brand"),
    "mondelez small sub"        : ("MDLZ SMALL SUB",           "ppg"),
    "mdlz small sub"            : ("MDLZ SMALL SUB",           "ppg"),
    # ── Additional seeds for frequently misclassified CPG entities ─────────
    "chips ahoy"                : ("NAB CHIPS AHOY",           "brand"),
    "nab chips ahoy"            : ("NAB CHIPS AHOY",           "brand"),
    "tates"                     : ("TATES",                    "brand"),
    "tate's"                    : ("TATES",                    "brand"),
    "tate s"                    : ("TATES",                    "brand"),
    "p&g"                       : ("P&G",                      "manufacturer"),
    "procter & gamble"          : ("P&G",                      "manufacturer"),
    "procter and gamble"        : ("P&G",                      "manufacturer"),
    "firehook"                  : ("FIREHOOK",                 "manufacturer"),
    "firehoo"                   : ("FIREHOOK",                 "manufacturer"),
    "fireho0e"                  : ("FIREHOOK",                 "manufacturer"),
    # ── Sub-category Nielsen code seeds ───────────────────────────────────
    # These Nielsen codes are cryptic enough that the LLM cannot infer them;
    # seed them here so no LLM call is needed for the fuzzy phase.
    "butter cookie"             : ("CKY-BUTTER",               "sub_category"),
    "butter cookies"            : ("CKY-BUTTER",               "sub_category"),
}

# In-process LRU-style expansion cache, pre-seeded from _ABBREVIATION_EXPANSIONS.
# Populated on-demand by _llm_expand_entity for new/unseen names.
# Thread-safe via _ENTITY_EXPANSION_LOCK.
_ENTITY_EXPANSION_CACHE: Dict[str, Tuple[str, Optional[str]]] = dict(_ABBREVIATION_EXPANSIONS)
_ENTITY_EXPANSION_LOCK  = threading.Lock()

# ── LLM prompts for dynamic entity expansion ──────────────────────────────────
_ENTITY_EXPAND_SYSTEM = """\
You are a CPG (Consumer Packaged Goods) retail expert specializing in Nielsen
IQ retail measurement data.

Your task: given a product / brand / manufacturer / category name as typed by
a business user (which may be an abbreviation, nickname, misspelling, or
informal form), expand it to the most likely CANONICAL retail name as it would
appear in a Nielsen database.

Nielsen-specific conventions to apply:
  • Nabisco brands carry the "NAB" prefix in Nielsen:
      Wheat Thins → NAB WHEAT THINS,  Nutter Butter → NAB NUTTER BUTTER,
      Oreo → NAB OREO,  Chips Ahoy → NAB CHIPS AHOY, etc.
  • MDLZ / MNZ / Mondelez abbreviations → MONDELEZ (manufacturer)
  • Subbrand short-hands:  SPK → SOUR PATCH KIDS (subbrand),
      SWF → SWEDISH FISH (subbrand),  SWF Tropical → SWEDISH FISH TROPICAL
  • Category misspellings:  Conify / Confify → CONFECTIONARY (mega_category)
  • Manufacturer abbreviations:  PVM → PERFETTI VAN MELLE,
      MARS WM → MARS WRIGLEY,  UNTDBISC → UNITED BISCUITS,
      GEN MILLS → GENERAL MILLS,  KFT → KRAFT, etc.
  • Drug / throat brands: "Halls Cough Drops" is a brand (not a manufacturer);
      canonical Nielsen brand = HALLS
  • Preserve prefixes that are already present (NAB, MDLZ, AO, etc.)

Also identify the most likely Nielsen hierarchy column for this entity
(one of: mega_category, category, sub_category, manufacturer, brand,
subbrand, ppg, market, customer, division, total).
Only set preferred_column when you are highly confident; otherwise null.

If the name is already in standard canonical form return it UNCHANGED with
confidence ≤ 0.40 so the caller knows no expansion was performed.

Return ONLY a JSON object — no prose, no markdown fence:
{
  "expanded":         "<canonical name in UPPER CASE>",
  "preferred_column": "<column name or null>",
  "confidence":       <0.0-1.0>,
  "reasoning":        "<one sentence>"
}
"""

_ENTITY_EXPAND_USER = """\
User question (for context): {question}
Entity as written in the SQL : {guessed}

Return the JSON object now.
"""


def _llm_expand_entity(
    llm,
    guessed: str,
    question: str = "",
) -> Tuple[str, Optional[str]]:
    """
    Expand an abbreviated / misspelled CPG entity name to its canonical form.

    Resolution order (fastest to slowest)
    --------------------------------------
    1. Exact lookup in ``_ENTITY_EXPANSION_CACHE`` — pre-seeded from
       ``_ABBREVIATION_EXPANSIONS`` and populated on demand for new values.
    2. Partial-prefix scan against the seed keys (e.g. ``"SWF Berry"``
       matches the ``"swf"`` seed → ``"SWEDISH FISH BERRY"``).
    3. LLM call — the result is cached so a second identical request is free.

    Returns ``(expanded_value, preferred_column_hint)``.
    Returns ``(guessed, None)`` unchanged when:
      - the LLM confidence is < 0.65, or
      - the LLM fails (network error, JSON parse error, etc.)
    In those cases the original text is passed to the fuzzy scorer as-is.
    """
    key = guessed.lower().strip()

    # 1. Exact cache hit
    with _ENTITY_EXPANSION_LOCK:
        if key in _ENTITY_EXPANSION_CACHE:
            cached = _ENTITY_EXPANSION_CACHE[key]
            if cached[0].upper() != guessed.upper():
                logger.info(
                    f"FilterResolver: entity cache hit '{guessed}' -> '{cached[0]}'"
                    + (f" (col: {cached[1]})" if cached[1] else "")
                )
            return cached

    # 2. Partial-prefix scan (e.g. "SWF Blind Melon" → "SWEDISH FISH BLIND MELON")
    for abbr, (expanded, col_hint) in _ABBREVIATION_EXPANSIONS.items():
        if key.startswith(abbr + " ") or key.endswith(" " + abbr):
            suffix = (
                key[len(abbr):].strip() if key.startswith(abbr)
                else key[: -len(abbr)].strip()
            )
            full_expansion = (expanded + " " + suffix).strip().upper()
            outcome: Tuple[str, Optional[str]] = (full_expansion, col_hint)
            logger.info(
                f"FilterResolver: partial-seed expansion '{guessed}' -> '{full_expansion}'"
            )
            with _ENTITY_EXPANSION_LOCK:
                _ENTITY_EXPANSION_CACHE[key] = outcome
            return outcome

    # 3. LLM expansion (handles any unseen CPG name)
    try:
        resp = llm.invoke([
            {"role": "system", "content": _ENTITY_EXPAND_SYSTEM},
            {"role": "user",   "content": _ENTITY_EXPAND_USER.format(
                question=question, guessed=guessed,
            )},
        ])
        raw = resp.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = raw.replace("```", "").strip()
        data       = json.loads(raw)
        expanded   = (data.get("expanded") or guessed).strip().upper()
        col_hint   = data.get("preferred_column") or None
        confidence = float(data.get("confidence", 0.0))

        if confidence >= 0.65 and expanded != guessed.upper():
            outcome = (expanded, col_hint)
            logger.info(
                f"FilterResolver: LLM entity expansion '{guessed}' -> '{expanded}' "
                f"(conf={confidence:.2f}"
                + (f", col={col_hint}" if col_hint else "") + ") "
                f"— {data.get('reasoning', '')}"
            )
        else:
            # Not confident — return original; fuzzy scorer will try its best
            outcome = (guessed, None)
            logger.debug(
                f"FilterResolver: LLM expansion '{guessed}' unchanged "
                f"(conf={confidence:.2f})"
            )
    except Exception as exc:
        logger.warning(f"FilterResolver: entity expansion LLM failed for '{guessed}' — {exc}")
        outcome = (guessed, None)

    # Cache result (including negative/unchanged results) to avoid repeated LLM calls
    with _ENTITY_EXPANSION_LOCK:
        _ENTITY_EXPANSION_CACHE[key] = outcome
    return outcome


# ── Three separate hierarchy groups (each is an independent coarsest→finest list) ──
#
# Category taxonomy: a product belongs to one mega_category > category > sub_category.
_CATEGORY_HIERARCHY: List[str] = ["mega_category", "category", "sub_category"]
#
# Brand/product: a product has one manufacturer > brand > subbrand > ppg.
_PRODUCT_HIERARCHY: List[str] = ["manufacturer", "brand", "subbrand", "ppg"]
#
# Geographic: data is sliced at one of four geographic granularities.
#   total (national) is the coarsest; market (metro/SMM) is the finest.
_GEO_HIERARCHY: List[str] = ["total", "division", "customer", "market"]

# Quick set for "is this a hierarchy column at all?"
_ALL_HIERARCHY_COLUMNS: set = set(
    _CATEGORY_HIERARCHY + _PRODUCT_HIERARCHY + _GEO_HIERARCHY
)

# Map each column to the hierarchy list it belongs to.
# The resolver uses this to restrict candidate comparison to the *correct* axis
# (e.g. resolving 'brand' never looks at mega_category values).
_COLUMN_TO_HIERARCHY: Dict[str, List[str]] = (
    {col: _CATEGORY_HIERARCHY for col in _CATEGORY_HIERARCHY}
    | {col: _PRODUCT_HIERARCHY for col in _PRODUCT_HIERARCHY}
    | {col: _GEO_HIERARCHY     for col in _GEO_HIERARCHY}
)

# ── Linguistic signals for granularity intent (per hierarchy level) ────────────
_INTENT_LEVEL_KEYWORDS: Dict[str, List[str]] = {
    # --- Category taxonomy ---
    "mega_category": [
        "segment", "space", "universe", "macro", "aisle",
        "sector", "super category", "mega", "master category",
    ],
    "category": [
        "category", "type", "kind",
    ],
    "sub_category": [
        "sub category", "subcategory", "sub segment",
        "subsegment", "sub-category",
    ],
    # --- Brand/product hierarchy ---
    "manufacturer": [
        "manufacturer", "company", "player", "competitor",
        "maker", "corp", "mfr", "who makes",
    ],
    "brand": [
        "brand", "label", "banner",
    ],
    "subbrand": [
        "variant", "subbrand", "sub brand", "sub-brand",
    ],
    "ppg": [
        "sku", "ppg", "product group", "planning group", "upc",
    ],
    # --- Geographic hierarchy ---
    "total": [
        "total us", "national", "all us", "us total", "total market", "country",
    ],
    "division": [
        "division", "region", "census region", "geographic division",
    ],
    "customer": [
        "retailer", "store", "account", "customer", "chain", "outlet", "channel",
    ],
    "market": [
        "market", "metro", "city", "dma", "smm", "local market", "metro area",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# SQL parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Use the dialect that matches the active provider so sqlglot can correctly
# handle provider-specific syntax (e.g. backtick-quoted multi-part identifiers
# with hyphens in Databricks vs. SQLite's simpler identifier rules).
_SQLGLOT_DIALECT: str = "databricks" if settings.llm_provider.lower() == "dbrx" else "sqlite"

# Regex: quarter_nielsen = <bare integer 1-4>  (no quotes — invisible to string walker)
_QUARTER_INT_RE = re.compile(
    r"(?i)\bquarter_nielsen\s*=\s*([1-4])(?!\d)(?!\s*['\"])"
)


def _fix_quarter_integer(sql: str) -> str:
    """Rewrite  quarter_nielsen = <1|2|3|4>  →  quarter_nielsen = 'Q<N>'.

    The LLM sometimes emits a bare integer despite the schema description;
    since it is a numeric literal (not a string), the filter resolver's
    string-literal walker cannot see or correct it.  This function catches
    it as a raw regex substitution before the SQL reaches the database.
    """
    fixed = _QUARTER_INT_RE.sub(
        lambda m: f"quarter_nielsen = 'Q{m.group(1)}'", sql
    )
    if fixed != sql:
        logger.info(
            "FilterResolver: rewrote bare-integer quarter_nielsen filter "
            "(e.g. = 1 → = 'Q1')"
        )
    return fixed


def _extract_string_filters(sql: str) -> List[Tuple[str, str]]:
    """
    Return a list of (column_name, literal_value) pairs for every
    ``col = 'string'``  (or  col IN ('a','b','c'))  predicate that appears in
    the WHERE clauses of the query.  Numeric literals are ignored.
    """
    results: List[Tuple[str, str]] = []
    try:
        # Use IGNORE so sqlglot returns a partial AST rather than raising on
        # Databricks-specific syntax it doesn't fully understand (backtick-quoted
        # multi-part identifiers with hyphens, SparkSQL extensions, etc.).
        # We only need the WHERE-clause predicate nodes, which always parse fine.
        tree = sqlglot.parse_one(
            sql,
            read=_SQLGLOT_DIALECT,
            error_level=sqlglot.errors.ErrorLevel.IGNORE,
        )
    except Exception as exc:
        logger.warning(f"FilterResolver: could not parse SQL — {exc}")
        return results

    def _col_from_expr(node) -> Optional[str]:
        """Extract a column name from a direct Column node or a single-arg
        function wrapper such as UPPER(col) / LOWER(col) / TRIM(col).
        Returns None when no column can be determined."""
        if isinstance(node, exp.Column):
            return node.name.lower()
        # Function wrapping a column: UPPER(t.col), LOWER(col), etc.
        inner = getattr(node, "this", None)
        if inner is not None and isinstance(inner, exp.Column):
            return inner.name.lower()
        return None

    # Walk every EQ node: col = 'value'  OR  UPPER(col) = 'VALUE'
    for node in tree.find_all(exp.EQ):
        left, right = node.left, node.right
        col_name = _col_from_expr(left)
        lit_val = None

        if isinstance(right, exp.Literal) and right.is_string:
            lit_val = right.this

        if col_name and lit_val and col_name not in _SKIP_COLUMNS:
            results.append((col_name, lit_val))

    # Walk IN lists: col IN ('a', 'b')  OR  UPPER(col) IN ('A', 'B')
    for node in tree.find_all(exp.In):
        col_name = _col_from_expr(node.this)
        if not col_name or col_name in _SKIP_COLUMNS:
            continue
        for expr in node.expressions:
            if isinstance(expr, exp.Literal) and expr.is_string:
                results.append((col_name, expr.this))

    # Walk every LIKE predicate: col LIKE '%pattern%'
    # Extract the text between % wildcards so it can be entity-expanded and
    # fuzzy-matched against DB values (e.g. '%Butter Cookie%' → 'Butter Cookie').
    for node in tree.find_all(exp.Like):
        col_node = node.this
        if not isinstance(col_node, exp.Column):
            continue
        col_name = col_node.name.lower()
        if col_name in _SKIP_COLUMNS:
            continue
        pattern_node = node.expression
        if isinstance(pattern_node, exp.Literal) and pattern_node.is_string:
            # Strip leading/trailing % wildcards to get the bare search term
            raw_pattern = pattern_node.this.strip("%").strip()
            if raw_pattern:
                results.append((col_name, raw_pattern))

    # De-duplicate while preserving order
    seen: set = set()
    unique = []
    for item in results:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _rewrite_sql(
    sql: str,
    value_corrections: Dict[Tuple[str, str], str],
    column_corrections: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
) -> str:
    """
    Apply corrections to the SQL string.

    value_corrections:  (col, old_val) -> new_val
        Same column, different value.
    column_corrections: (col, old_val) -> (new_col, new_val)
        Different hierarchy level was chosen — rename the column too.
    """
    result = sql

    # Collect ALL (old_literal → new_literal) pairs from both correction dicts
    # so we can run a final broad sweep at the end that catches any form the
    # structured patterns miss (UPPER(), LIKE, unusual apostrophe encoding, etc.)
    _all_literal_swaps: List[Tuple[str, str]] = []

    # 1. Cross-column corrections first (so value replacement doesn't interfere)
    if column_corrections:
        for (old_col, old_val), (new_col, new_val) in column_corrections.items():
            if old_col == new_col:
                # Same-column rule (e.g. candy → NON CHOCOLATE in same column).
                # Treat as a value correction so the replacement actually fires.
                value_corrections = dict(value_corrections)  # make mutable copy
                value_corrections[(old_col, old_val)] = new_val
                continue

            # a) plain EQ:  old_col = 'old_val'  ->  new_col = 'new_val'
            eq_pattern = re.compile(
                r"(?i)\b" + re.escape(old_col)
                + r"(\s*=\s*')" + re.escape(old_val) + r"(')",
            )
            result = eq_pattern.sub(
                lambda m, nc=new_col, nv=new_val: f"{nc}{m.group(1)}{nv}{m.group(2)}",
                result,
            )

            # b) LIKE predicate:  old_col LIKE '%old_val%'  ->  new_col = 'new_val'
            #    Also handles UPPER(old_col) LIKE and variations.
            like_pattern = re.compile(
                r"(?i)(?:UPPER\s*\([^)]*\b" + re.escape(old_col) + r"[^)]*\)|\b"
                + re.escape(old_col) + r")\s+LIKE\s+'[^']*"
                + re.escape(old_val) + r"[^']*'",
            )
            result = like_pattern.sub(f"{new_col} = '{new_val}'", result)

            # c) Rename the column in any IN clause heading
            in_pattern = re.compile(r"(?i)\b" + re.escape(old_col) + r"(\s+IN\s*\()")
            result = in_pattern.sub(lambda m, nc=new_col: f"{nc}{m.group(1)}", result)

            _all_literal_swaps.append((old_val, new_val))

    # 2. Same-column value corrections
    for (col, old_val), new_val in value_corrections.items():
        if old_val == new_val:
            continue
        # a) col = 'old_val' (plain EQ)
        eq_pattern = re.compile(
            r"(?i)(\b" + re.escape(col) + r"\s*=\s*')(" + re.escape(old_val) + r")(')",
        )
        result = eq_pattern.sub(lambda m, nv=new_val: m.group(1) + nv + m.group(3), result)
        # b) UPPER(col) = 'OLD_VAL'
        upper_pattern = re.compile(
            r"(?i)(UPPER\s*\([^)]*\b" + re.escape(col) + r"[^)]*\)\s*=\s*')("
            + re.escape(old_val.upper()) + r"|" + re.escape(old_val) + r")'",
        )
        result = upper_pattern.sub(lambda m, nv=new_val.upper(): m.group(1) + nv + "'", result)

        _all_literal_swaps.append((old_val, new_val))

    # 3. Final broad sweep — replaces 'old_val' wherever it appears between
    #    single quotes, regardless of surrounding SQL syntax.  This is the
    #    safety net for UPPER() wrappers, IN lists, and apostrophe-encoding
    #    variations (doubled '' vs raw ' in LLM-generated SQL).
    #    Values are specific entity names so false positives are negligible.
    for old_val, new_val in _all_literal_swaps:
        if old_val == new_val:
            continue
        # Try SQL-escaped form first ('' for embedded apostrophes)
        sql_esc_old = old_val.replace("'", "''")
        sql_esc_new = new_val.replace("'", "''")
        result = result.replace(f"'{sql_esc_old}'", f"'{sql_esc_new}'")
        # Then the raw form (LLM sometimes omits proper escaping)
        result = result.replace(f"'{old_val}'", f"'{new_val}'")
        # Case-insensitive match for UPPER() contexts where LLM uppercased the value
        if old_val.upper() != old_val:
            result = result.replace(f"'{old_val.upper()}'", f"'{new_val.upper()}'")

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
    # NOTE: product_name is a real DB column — do NOT alias it to brand.
    #       It is in _KNOWN_DB_COLUMNS and handled by _resolve_simple.
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
    # real product-level columns — must NOT be aliased to other columns
    "product_name", "upc", "pack", "pack_type",
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

# Provider-aware default table for distinct-value fetches.
_FETCH_TABLE: str = (
    settings.dbx_full_table
    if settings.llm_provider.lower() == "dbrx"
    else "nielsen_pos"
)


def _get_col_value(row: dict, column: str):
    """Return the value for *column* from *row* using a case-insensitive key lookup.

    Databricks may return column names in a different case from the one we used
    in the SELECT clause (e.g. 'BRAND' instead of 'brand').  This helper finds
    the matching key regardless of casing so we never silently miss values.
    """
    if column in row:
        return row[column]
    lower = column.lower()
    for k, v in row.items():
        if k.lower() == lower:
            return v
    return None


def _fetch_distinct(column: str, table: str = _FETCH_TABLE) -> List[str]:
    """Fetch up to _MAX_DISTINCT unique non-null values for a column."""
    # For Databricks/SparkSQL we skip the empty-string filter because the
    # column may be a non-varchar type; IS NOT NULL is sufficient.
    if settings.llm_provider.lower() == "dbrx":
        sql = (
            f"SELECT DISTINCT {column} FROM {table} "
            f"WHERE {column} IS NOT NULL "
            f"LIMIT {_MAX_DISTINCT}"
        )
    else:
        sql = (
            f"SELECT DISTINCT {column} FROM {table} "
            f"WHERE {column} IS NOT NULL AND {column} != '' "
            f"LIMIT {_MAX_DISTINCT}"
        )
    rows, err, _ = db_manager.execute_query(sql)
    if err:
        logger.error(
            f"FilterResolver: _fetch_distinct DB error for column '{column}': {err}"
        )
        return []
    if not rows:
        logger.warning(f"FilterResolver: _fetch_distinct returned 0 rows for '{column}'")
        return []
    values = [str(_get_col_value(r, column)) for r in rows
              if _get_col_value(r, column) is not None]
    logger.debug(f"FilterResolver: '{column}' → {len(values)} distinct values fetched")
    return values


def _fetch_all_hierarchy_values() -> Dict[str, List[str]]:
    """DEPRECATED — retained for backward compat; use _build_value_cache() instead."""
    return _build_value_cache()


# ─────────────────────────────────────────────────────────────────────────────
# JSON value cache — avoids repeated DB hits across requests
# ─────────────────────────────────────────────────────────────────────────────


def _discover_valid_cache_columns() -> List[str]:
    """Return the subset of _ALL_HIERARCHY_COLUMNS that actually exist in the DB table.

    For the ``dbrx`` provider we run ``SELECT * … LIMIT 0`` against the
    Databricks table to discover the real column names and intersect them with
    the hardcoded hierarchy set.  This prevents ``UNRESOLVED_COLUMN`` SQL
    errors during cache warmup when the Databricks table has a different schema
    than the SQLite ``nielsen_pos`` table used by the ``groq`` provider.

    For ``groq`` (or any other provider) we trust the hardcoded list as-is.
    """
    if settings.llm_provider.lower() != "dbrx":
        return list(_ALL_HIERARCHY_COLUMNS)
    try:
        from dbx_connection import get_table_columns  # lazy import — dbrx only
        actual_cols = set(get_table_columns())
        valid = [c for c in _ALL_HIERARCHY_COLUMNS if c in actual_cols]
        if not valid:
            logger.warning(
                "FilterResolver: none of the expected hierarchy columns "
                f"({sorted(_ALL_HIERARCHY_COLUMNS)}) exist in the Databricks "
                f"table.  Actual columns found: {sorted(actual_cols)}.  "
                "The value cache will be empty — update the hierarchy column "
                "constants to match the real schema."
            )
        else:
            logger.info(
                f"FilterResolver: {len(valid)}/{len(_ALL_HIERARCHY_COLUMNS)} "
                f"hierarchy columns present in Databricks table: {valid}"
            )
        return valid
    except Exception as exc:
        logger.warning(
            f"FilterResolver: could not query Databricks table columns "
            f"({exc}); falling back to full hardcoded list.  Cache warmup "
            "may produce UNRESOLVED_COLUMN errors for non-existent columns."
        )
        return list(_ALL_HIERARCHY_COLUMNS)


# All columns whose distinct values we cache.
# For dbrx this is narrowed at startup to only columns that actually exist in
# the Databricks table (avoiding UNRESOLVED_COLUMN errors during warmup).
_CACHED_COLUMNS: List[str] = _discover_valid_cache_columns()

# Cache file is scoped to the active provider so groq and dbrx never share
# or overwrite each other's distinct-value snapshots.
_CACHE_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache", f"db_values_cache_{settings.llm_provider.lower()}.json"
)

# In-memory cache — populated once per process lifetime by the background
# warmup thread.  _get_all_cached_values() waits on _CACHE_READY_EVENT so
# queries never see a partially-built cache.
_VALUE_CACHE_MEMORY: Optional[Dict[str, List[str]]] = None
_CACHE_READY_EVENT  = threading.Event()    # set when cache is ready
_CACHE_WARM_THREAD: Optional[threading.Thread] = None  # background builder


def _load_value_cache() -> Optional[Dict[str, List[str]]]:
    """Load the JSON cache from disk.  Returns the values dict or None on miss."""
    try:
        if not os.path.exists(_CACHE_PATH):
            return None
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("values")
    except Exception as exc:
        logger.warning(f"FilterResolver: could not load value cache: {exc}")
        return None


def _save_value_cache(values: Dict[str, List[str]]) -> None:
    """Persist distinct values to cache/db_values_cache_{provider}.json."""
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        payload = {"cached_at": datetime.utcnow().isoformat(), "values": values}
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info(f"FilterResolver: value cache saved to {_CACHE_PATH}")
    except Exception as exc:
        logger.warning(f"FilterResolver: could not save value cache: {exc}")


def _build_value_cache() -> Dict[str, List[str]]:
    """Fetch distinct values for every cached column in parallel and persist."""
    result: Dict[str, List[str]] = {}
    with ThreadPoolExecutor(max_workers=min(len(_CACHED_COLUMNS), 8)) as pool:
        futures = {pool.submit(_fetch_distinct, col): col for col in _CACHED_COLUMNS}
        for future in as_completed(futures):
            col = futures[future]
            try:
                vals = future.result()
                result[col] = vals
                logger.info(
                    f"FilterResolver: cached {len(vals)} distinct values for '{col}'"
                )
            except Exception as exc:
                logger.error(
                    f"FilterResolver: failed to fetch distinct for '{col}': {exc}"
                )
                result[col] = []

    total_values = sum(len(v) for v in result.values())
    if total_values > 0:
        _save_value_cache(result)
    else:
        logger.error(
            "FilterResolver: ALL columns returned 0 values — cache NOT written. "
            "Check that column names in _CACHED_COLUMNS match the Databricks table "
            f"'{_FETCH_TABLE}' and that the DB connection is working."
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Background warmup
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_worker() -> None:
    """Runs in a daemon thread: load from disk or fetch from DB, then signal ready."""
    global _VALUE_CACHE_MEMORY
    try:
        disk = _load_value_cache()
        disk_has_data = disk is not None and sum(len(v) for v in disk.values()) > 0
        if disk_has_data:
            logger.info("FilterResolver: value cache loaded from disk.")
            _VALUE_CACHE_MEMORY = disk
        else:
            if disk is not None:
                logger.warning(
                    "FilterResolver: disk cache exists but contains no values "
                    "(previous build may have failed) — rebuilding from DB."
                )
            else:
                logger.info(
                    "FilterResolver: cache file absent — fetching distinct values from DB "
                    "(one-time cost; subsequent runs load instantly from disk)."
                )
            _VALUE_CACHE_MEMORY = _build_value_cache()
    except Exception as exc:
        logger.error(f"FilterResolver: cache warmup failed: {exc}")
        _VALUE_CACHE_MEMORY = {}
    finally:
        _CACHE_READY_EVENT.set()
        logger.info("FilterResolver: value cache ready.")


def initialize_value_cache() -> None:
    """
    Start the background cache warmup (idempotent — safe to call multiple times).

    Call this once at **application startup** (before any query is served) so the
    DB fetch overlaps with the rest of the startup sequence.  By the time the
    first real query arrives the cache will already be warm.

    If the JSON cache file exists the warmup is essentially instant (one file
    read). The file is created automatically on the very first run.
    """
    global _CACHE_WARM_THREAD
    if _CACHE_READY_EVENT.is_set():
        return  # already warm
    if _CACHE_WARM_THREAD is not None and _CACHE_WARM_THREAD.is_alive():
        return  # already running
    _CACHE_WARM_THREAD = threading.Thread(
        target=_warmup_worker,
        name="FilterResolver-CacheWarmup",
        daemon=True,
    )
    _CACHE_WARM_THREAD.start()
    logger.info("FilterResolver: cache warmup thread started.")


def _get_all_cached_values() -> Dict[str, List[str]]:
    """
    Return distinct DB values for all hierarchy columns.

    Blocks only if the background warmup thread has not finished yet
    (which only happens when a query arrives faster than the DB fetch
    completes — extremely unlikely in any real startup sequence).
    """
    if not _CACHE_READY_EVENT.is_set():
        logger.debug("FilterResolver: waiting for cache warmup thread...")
        _CACHE_READY_EVENT.wait()  # unblocks the moment the thread sets the event
    return _VALUE_CACHE_MEMORY or {}


def refresh_value_cache() -> Dict[str, List[str]]:
    """Force-rebuild the value cache from the DB and reset the in-memory copy."""
    global _VALUE_CACHE_MEMORY
    _CACHE_READY_EVENT.clear()
    logger.info("FilterResolver: refreshing value cache from DB.")
    _VALUE_CACHE_MEMORY = _build_value_cache()
    _CACHE_READY_EVENT.set()
    return _VALUE_CACHE_MEMORY


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy-aware scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fuzzy_score(a: str, b: str) -> float:
    """Token-aware fuzzy similarity between two strings, 0-1."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if a in b or b in a:
        ratio = max(ratio, 0.75)
    return ratio


def _level_alignment_score(candidate_level: str, intended_level: str, hierarchy: List[str]) -> float:
    """
    Score how well the candidate's hierarchy level matches the intended level.
    Same level -> 1.0; decreases with distance; coarser levels get a small
    bonus over finer levels to implement the "default up" principle.

    ``hierarchy`` must be the specific hierarchy list (category / product / geo)
    that the column being resolved belongs to — never the full merged union.
    """
    try:
        cand_idx     = hierarchy.index(candidate_level)
        intended_idx = hierarchy.index(intended_level)
    except ValueError:
        return 0.5  # unknown level — neutral
    distance = abs(cand_idx - intended_idx)
    base_score    = max(0.0, 1.0 - distance * 0.25)
    coarser_bonus = 0.05 if cand_idx < intended_idx else 0.0
    return min(1.0, base_score + coarser_bonus)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Granularity intent classification
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """\
You are a Nielsen retail analytics expert. Given a user question and an entity
mention from that question, identify the intended hierarchy level for that entity.

The Nielsen hierarchies (each is independent — coarsest -> finest):

  CATEGORY  : mega_category  > category       > sub_category
  PRODUCT   : manufacturer   > brand          > subbrand       > ppg
  GEOGRAPHIC: total          > division       > customer       > market

Linguistic signals:
  mega_category  — "segment", "space", "universe", "macro", "aisle", "sector"
  category       — "category", "type", "kind"
  sub_category   — "sub category", "subcategory", "sub segment"
  manufacturer   — "manufacturer", "company", "player", "competitor", "maker"
  brand          — "brand", "label", "banner"
  subbrand       — "variant", "subbrand", "sub brand"
  ppg            — "sku", "ppg", "product group", "planning group"
  total          — "total us", "national", "all us", "country"
  division       — "division", "region", "census region"
  customer       — "retailer", "store", "account", "chain", "outlet", "channel"
  market         — "market", "metro", "city", "dma", "smm", "local market"

Return ONLY a JSON object — no prose, no markdown:
{
  "intended_level": "<one of the 11 levels above>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>"
}
"""

_CLASSIFY_USER = """\
User question : {question}
Entity mention: {mention}
SQL column used by the model: {sql_column}
Valid levels for this hierarchy: {valid_levels}

Restrict your answer to one of the valid levels listed above.
Return the JSON object now.
"""


def _fast_keyword_classify(
    question: str, mention: str, valid_levels: List[str]
) -> Optional[str]:
    """
    Fast keyword scan — no LLM.
    Returns the intended hierarchy level (restricted to valid_levels) or None.
    """
    combined = (question + " " + mention).lower()
    scores: Dict[str, int] = {level: 0 for level in valid_levels}
    for level in valid_levels:
        keywords = _INTENT_LEVEL_KEYWORDS.get(level, [])
        for kw in keywords:
            weight = 2 if " " in kw else 1
            if kw in combined:
                scores[level] += weight
    best_level = max(scores, key=lambda l: scores[l])
    if scores[best_level] > 0:
        logger.debug(f"FilterResolver: fast keyword classify -> {best_level} (scores={scores})")
        return best_level
    return None


def _classify_intent_level(
    llm,
    question: str,
    mention: str,
    sql_column: str,
    valid_levels: List[str],
) -> Tuple[str, float]:
    """
    Return (intended_level, confidence) restricted to the given hierarchy's levels.
    Tries fast keyword scan first; falls back to LLM if no strong signal.
    """
    fast = _fast_keyword_classify(question, mention, valid_levels)
    if fast:
        return fast, 0.80
    try:
        resp = llm.invoke([
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user",   "content": _CLASSIFY_USER.format(
                question=question, mention=mention,
                sql_column=sql_column,
                valid_levels=", ".join(valid_levels),
            )},
        ])
        raw = resp.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = raw.replace("```", "").strip()
        data = json.loads(raw)
        level = data.get("intended_level", sql_column)
        if level not in valid_levels:
            level = sql_column if sql_column in valid_levels else valid_levels[0]
        conf  = float(data.get("confidence", 0.5))
        logger.debug(
            f"FilterResolver: LLM classify '{mention}' -> {level} "
            f"(conf={conf:.2f}) — {data.get('reasoning', '')}"
        )
        return level, conf
    except Exception as exc:
        logger.warning(f"FilterResolver: intent classification failed — {exc}")
        return sql_column if sql_column in valid_levels else valid_levels[0], 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2-4 — Hierarchy-aware candidate resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_hierarchy_entity(
    llm,
    question: str,
    sql_column: str,
    guessed: str,
    all_hierarchy_values: Dict[str, List[str]],
    hierarchy: List[str],
) -> dict:
    """
    4-phase intent-first, hierarchy-aware resolution for all three hierarchy types
    (category, product, geographic).

    ``hierarchy`` is the specific ordered list for this axis
    (e.g. _PRODUCT_HIERARCHY); all scoring is restricted to that list so that
    brand values are never conflated with mega_category or market values.

    Returns dict with keys: match, matched_column, confidence, clarify,
    question (clarification text), interpretation (user-facing note).
    """
    # Phase 1: classify intended granularity — restricted to this hierarchy's levels
    intended_level, intent_conf = _classify_intent_level(
        llm, question, guessed, sql_column, valid_levels=hierarchy
    )
    logger.debug(
        f"FilterResolver: '{guessed}' in '{sql_column}' -> "
        f"intended level = {intended_level} (conf={intent_conf:.2f})"
    )

    # Phase 2: score every DB value across all levels of *this* hierarchy
    candidates = []
    for level, values in all_hierarchy_values.items():
        for val in values:
            s_score  = _fuzzy_score(guessed, val)
            l_score  = _level_alignment_score(level, intended_level, hierarchy)
            combined = _STRING_WEIGHT * s_score + _LEVEL_WEIGHT * l_score
            candidates.append({
                "value": val, "level": level,
                "s_score": s_score, "l_score": l_score, "combined": combined,
            })

    if not candidates:
        return {
            "match": guessed, "matched_column": sql_column,
            "confidence": 0.0, "clarify": True,
            "question": "I couldn't find any matching values in the database.",
            "interpretation": None,
        }

    candidates.sort(key=lambda x: x["combined"], reverse=True)
    top = candidates[0]

    # ── Cross-level exact-match guard ──────────────────────────────────────
    # If ANY candidate (regardless of level) has a very high string similarity
    # (>= _CROSS_LEVEL_EXACT_THRESHOLD) it almost certainly IS the right entity
    # even if it sits at a different hierarchy level than the SQL column.
    # Return it immediately before any level-alignment logic can override it.
    # This fixes cases like: manufacturer='NATURE VALLEY' when the DB only has
    # brand='NATURE VALLEY' — the string score 1.0 at brand should always win.
    cross_exact: Optional[dict] = None
    for cand in candidates:
        if cand["s_score"] >= _CROSS_LEVEL_EXACT_THRESHOLD:
            if cross_exact is None or cand["s_score"] > cross_exact["s_score"]:
                cross_exact = cand

    if cross_exact is not None and cross_exact["level"] != sql_column:
        corrected_val = cross_exact["value"]
        interp = (
            f"Interpreted **'{guessed}'** as **{cross_exact['level']}** = "
            f"'**{corrected_val}**' (exact match at different hierarchy level)"
        )
        logger.info(
            f"FilterResolver: cross-level exact-match guard — promoting "
            f"'{guessed}' from '{sql_column}' to "
            f"'{cross_exact['level']}'='{corrected_val}' "
            f"(s_score={cross_exact['s_score']:.2f})"
        )
        return {
            "match":          corrected_val,
            "matched_column": cross_exact["level"],
            "confidence":     cross_exact["combined"],
            "clarify":        False,
            "question":       None,
            "interpretation": interp,
        }

    # ── Perfect-match guard ────────────────────────────────────────────────
    # If a candidate at the SAME level as sql_column has an exact or near-exact
    # string match (s_score >= 0.90, i.e. identical casing-normalised or a
    # single-character typo), use it immediately — do NOT attempt any
    # cross-level promotion via Phase 3 or Phase 4.
    #
    # This prevents a correctly-placed filter (e.g. mega_category='CHOCOLATE')
    # from being demoted to a finer level (category/sub_category/brand) purely
    # because the intent classifier inferred a finer granularity and boosted
    # their Level-alignment score above the exact mega_category hit.
    sql_col_best: Optional[dict] = None
    for cand in candidates:
        if cand["level"] == sql_column and cand["s_score"] >= 0.90:
            if sql_col_best is None or cand["s_score"] > sql_col_best["s_score"]:
                sql_col_best = cand

    if sql_col_best is not None:
        corrected_val  = sql_col_best["value"]
        interp = (
            f"Interpreted **'{guessed}'** as **{sql_column}** = '**{corrected_val}**'"
            if corrected_val.upper() != guessed.upper() else None
        )
        logger.info(
            f"FilterResolver: perfect-match guard — keeping '{guessed}' at "
            f"'{sql_column}' -> '{corrected_val}' "
            f"(s_score={sql_col_best['s_score']:.2f}); skipping cross-level phases"
        )
        return {
            "match":          corrected_val,
            "matched_column": sql_column,
            "confidence":     sql_col_best["combined"],
            "clarify":        False,
            "question":       None,
            "interpretation": interp,
        }

    # Phase 3: hierarchy consistency — prefer coarser if top is finer than intended
    top_idx      = hierarchy.index(top["level"]) if top["level"] in hierarchy else 0
    intended_idx = hierarchy.index(intended_level) if intended_level in hierarchy else 0

    if top_idx > intended_idx:
        for cand in candidates:
            cand_idx = hierarchy.index(cand["level"]) if cand["level"] in hierarchy else len(hierarchy)
            if cand_idx <= intended_idx and cand["s_score"] >= _MIN_STRING_SCORE_FOR_COARSER:
                logger.info(
                    f"FilterResolver: Phase 3 override — promoted "
                    f"'{cand['value']}' ({cand['level']}) over "
                    f"'{top['value']}' ({top['level']}) "
                    f"because intended level={intended_level}"
                )
                top = cand
                break

    # Phase 4: ambiguity — default to coarser when top-2 are close
    second = candidates[1] if len(candidates) > 1 else None
    ambiguous = (
        second is not None
        and top["combined"] - second["combined"] <= _HIERARCHY_AMBIGUITY_MARGIN
        and top["level"] != second["level"]
    )

    if ambiguous:
        top_idx2    = hierarchy.index(top["level"])    if top["level"]    in hierarchy else 0
        second_idx2 = hierarchy.index(second["level"]) if second["level"] in hierarchy else 0
        coarser = top if top_idx2 <= second_idx2 else second
        finer   = second if top_idx2 <= second_idx2 else top
        logger.info(
            f"FilterResolver: ambiguous — '{top['value']}' ({top['level']}) vs "
            f"'{second['value']}' ({second['level']}) — defaulting to coarser: "
            f"'{coarser['value']}' ({coarser['level']})"
        )
        interpretation = (
            f"Interpreted **'{guessed}'** as **{coarser['level']}** = "
            f"'**{coarser['value']}**'. "
            f"(Also matched '{finer['value']}' at {finer['level']} level — "
            f"defaulted to broader level for analytics.)"
        )
        return {
            "match":          coarser["value"],
            "matched_column": coarser["level"],
            "confidence":     coarser["combined"],
            "clarify":        False,
            "question":       None,
            "interpretation": interpretation,
        }

    # Normal resolution
    confidence     = top["combined"]
    level_note     = f" (intended level: {intended_level})" if top["level"] != intended_level else ""
    interpretation = (
        f"Interpreted **'{guessed}'** as **{top['level']}** = '**{top['value']}**'{level_note}"
    )

    if confidence < _AUTO_CORRECT_THRESHOLD:
        top_at_level = [c for c in candidates[:5] if c["level"] == intended_level] or candidates[:3]
        options_str  = ", ".join(f"'{c['value']}' ({c['level']})" for c in top_at_level)
        return {
            "match":          top["value"],
            "matched_column": top["level"],
            "confidence":     confidence,
            "clarify":        True,
            "question": (
                f"Could you clarify what **'{guessed}'** refers to? "
                f"Possible matches: {options_str}."
            ),
            "interpretation": interpretation,
        }

    logger.info(
        f"FilterResolver: hierarchy resolved '{guessed}' -> "
        f"{top['level']}='{top['value']}' (score={confidence:.2f})"
    )
    return {
        "match":          top["value"],
        "matched_column": top["level"],
        "confidence":     confidence,
        "clarify":        False,
        "question":       None,
        "interpretation": interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simple LLM resolver (non-hierarchy columns)
# ─────────────────────────────────────────────────────────────────────────────

_SIMPLE_SYSTEM = """\
You are a database filter-value matcher for a Nielsen retail analytics system.

Given:
- The column name
- The value the user/LLM wrote in the SQL query ("guessed value")
- A list of REAL values that exist in the database for that column

Your task: identify which real database value best matches the guessed value.

Rules:
- Prioritise semantic / abbreviation matches (e.g. "MDLZ" -> "MONDELEZ",
  "Northeast" -> "NE", "Total Bars" -> "TOTAL CHOCOLATE BARS", etc.).
- Consider common Nielsen abbreviations and category naming conventions.
- If two or more values are equally plausible AND the distinction matters,
  set "clarify": true and compose a short clarifying question.
- If there is simply no reasonable match, set "clarify": true.
- Return ONLY a JSON object — no prose, no markdown.

JSON schema:
{
  "match": "<best matching DB value or null>",
  "confidence": <0.0 - 1.0>,
  "clarify": <true | false>,
  "question": "<clarification question if clarify=true, else null>"
}
"""

_SIMPLE_USER = """\
Column     : {column}
SQL value  : {guessed}
DB values  : {db_values}

Return the JSON object now.
"""


def _resolve_simple(
    llm,
    column: str,
    guessed: str,
    db_values: List[str],
) -> dict:
    """
    Simple LLM-based resolver for non-hierarchy columns (market, customer,
    division, total, etc.).  Returns the same dict shape as
    _resolve_hierarchy_entity with matched_column = column (unchanged).
    """
    base = {"matched_column": column, "interpretation": None}

    if not db_values:
        suggested = _suggest_column(column)
        if suggested:
            suggested_values = _fetch_distinct(suggested)
            if suggested_values:
                lower_map = {v.lower(): v for v in suggested_values}
                if guessed.lower() in lower_map:
                    best = lower_map[guessed.lower()]
                    return {**base, "match": guessed, "confidence": 0.0, "clarify": True,
                            "question": (
                                f"There's no column named **{column}** in the database — "
                                f"did you mean **{suggested}** = '{best}'?"
                            )}
                values_str = "\n".join(f"  - {v}" for v in suggested_values[:_MAX_DISTINCT])
                try:
                    resp = llm.invoke([
                        {"role": "system", "content": _SIMPLE_SYSTEM},
                        {"role": "user",   "content": _SIMPLE_USER.format(
                            column=suggested, guessed=guessed, db_values=values_str
                        )},
                    ])
                    raw = resp.content.strip()
                    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                    raw = raw.replace("```", "").strip()
                    result = json.loads(raw)
                    best = result.get("match") or guessed
                    conf = float(result.get("confidence", 0.0))
                    return {**base, "match": guessed, "confidence": 0.0, "clarify": True,
                            "question": (
                                f"There's no column named **{column}** in the database. "
                                f"Did you mean **{suggested}** = '{best}'?"
                                + (f" (confidence: {conf:.0%})" if conf < 0.9 else "")
                            )}
                except Exception as exc:
                    logger.warning(f"FilterResolver: suggestion LLM failed — {exc}")
                    return {**base, "match": guessed, "confidence": 0.0, "clarify": True,
                            "question": (
                                f"There's no column named **{column}** — "
                                f"did you mean **{suggested}**?"
                            )}
        cols = ", ".join(sorted(_KNOWN_DB_COLUMNS))
        return {**base, "match": guessed, "confidence": 0.0, "clarify": True,
                "question": (
                    f"I couldn't find a column named **{column}** in the database. "
                    f"Available filter columns are: {cols}."
                )}

    # Exact case-insensitive match — skip LLM
    lower_map = {v.lower(): v for v in db_values}
    if guessed.lower() in lower_map:
        exact = lower_map[guessed.lower()]
        logger.debug(f"FilterResolver: '{column}'='{guessed}' exact match -> '{exact}'")
        return {**base, "match": exact, "confidence": 1.0, "clarify": False, "question": None}

    values_str = "\n".join(f"  - {v}" for v in db_values[:_MAX_DISTINCT])
    try:
        resp = llm.invoke([
            {"role": "system", "content": _SIMPLE_SYSTEM},
            {"role": "user",   "content": _SIMPLE_USER.format(
                column=column, guessed=guessed, db_values=values_str
            )},
        ])
        raw = resp.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = raw.replace("```", "").strip()
        result = json.loads(raw)
        return {**base, **result}
    except Exception as exc:
        logger.warning(f"FilterResolver: LLM call failed for {column}='{guessed}' — {exc}")
        return {**base, "match": guessed, "confidence": 0.0, "clarify": True,
                "question": f"I couldn't verify the value '{guessed}' for '{column}'."}


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — routes to hierarchy or simple resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_one(
    llm,
    question: str,
    column: str,
    guessed: str,
    cached_values: Dict[str, List[str]],
) -> dict:
    """
    Dispatch to the correct resolver based on the column type.

    Hierarchy columns (category, product, or geographic hierarchy)
        -> _resolve_hierarchy_entity
        Uses 4-phase intent-first, level-aware scoring restricted to the
        column's own hierarchy axis (no cross-contamination between axes).

    All other columns -> _resolve_simple
        Single-level LLM fuzzy matcher using cached or freshly-fetched values.

    ``cached_values`` is the full dict returned by _get_all_cached_values().
    """
    # ── Entity expansion (abbreviations, misspellings, CPG nicknames) ────────
    # Expands the SQL value to its canonical CPG entity name BEFORE fuzzy
    # scoring so that e.g. "SPK" → "SOUR PATCH KIDS" rather than scoring ~0
    # against all DB values.
    #
    # IMPORTANT: Skip expansion if the value already exists verbatim in ANY
    # cached DB column.  LLM expansion of an already-valid DB value can corrupt
    # it — e.g. 'NAB RITZ CRK' (exact brand) would be "helpfully" expanded to
    # 'NAB RITZ CRACKERFULS' (a subbrand), causing the wrong column to be chosen.
    # An exact DB hit means the value is already canonical; scoring handles the rest.
    guessed_norm = guessed.upper().strip()
    _already_in_db = any(
        any(v.upper().strip() == guessed_norm for v in vals)
        for vals in cached_values.values()
        if vals
    )
    if _already_in_db:
        expanded_guessed, col_hint = guessed, None
        logger.debug(
            f"FilterResolver: '{guessed}' already exists verbatim in DB cache "
            "— skipping entity expansion to preserve exact match"
        )
    else:
        expanded_guessed, col_hint = _llm_expand_entity(llm, guessed, question)

    if expanded_guessed != guessed:
        # If the expansion supplies a preferred column hint, trust it over the
        # LLM-generated column name (which is often wrong for abbreviations).
        if col_hint and col_hint in _ALL_HIERARCHY_COLUMNS:
            logger.info(
                f"FilterResolver: using column hint '{col_hint}' from "
                f"abbreviation expansion (overrides SQL column '{column}')"
            )
            column = col_hint
        guessed = expanded_guessed

    # Normalise hallucinated column names first
    real_column = _suggest_column(column) if column not in _KNOWN_DB_COLUMNS else column
    if real_column and real_column != column:
        logger.info(f"FilterResolver: column '{column}' mapped to '{real_column}'")

    effective_column = real_column or column

    # ── Exact-match hierarchy redirect ────────────────────────────────────────
    # When the SQL uses a non-hierarchy column (e.g. product_name, upc) but the
    # filter value is an exact match in a cached hierarchy column (e.g. brand),
    # redirect to that hierarchy column so the hierarchy resolver runs and
    # returns the correct column name.
    #
    # Example: product_name = 'NAB RITZ CRK'
    #   → exact hit in cached_values['brand'] → redirect effective_column to 'brand'
    #   → hierarchy resolver returns brand = 'NAB RITZ CRK'  (perfect-match guard)
    #
    # Priority order matches coarsest→finest in _PRODUCT_HIERARCHY so that
    # brand beats subbrand when the same string exists at multiple levels.
    if effective_column not in _ALL_HIERARCHY_COLUMNS and _already_in_db:
        _redirect_col: Optional[str] = None
        _redirect_priority = len(_PRODUCT_HIERARCHY) + len(_CATEGORY_HIERARCHY) + len(_GEO_HIERARCHY)
        for hier_list in (_CATEGORY_HIERARCHY, _PRODUCT_HIERARCHY, _GEO_HIERARCHY):
            for hier_col in hier_list:
                vals = cached_values.get(hier_col, [])
                if any(v.upper().strip() == guessed_norm for v in vals):
                    idx = hier_list.index(hier_col)
                    if idx < _redirect_priority:
                        _redirect_priority = idx
                        _redirect_col = hier_col
                        _redirect_hierarchy = hier_list
        if _redirect_col:
            logger.info(
                f"FilterResolver: exact-match redirect — '{effective_column}'='{guessed}' "
                f"found verbatim in hierarchy column '{_redirect_col}'; "
                f"routing to hierarchy resolver on '{_redirect_col}'"
            )
            effective_column = _redirect_col

    if effective_column in _ALL_HIERARCHY_COLUMNS:
        hierarchy = _COLUMN_TO_HIERARCHY[effective_column]
        # Only pass values that belong to *this* hierarchy axis
        hierarchy_values = {col: cached_values.get(col, []) for col in hierarchy}

        # If every level in the hierarchy has zero cached values the column was
        # either excluded from _CACHED_COLUMNS (dbrx schema mismatch for sibling
        # columns) or the warmup hasn't finished yet.  In that case, fetch the
        # specific column's values on-demand and fall back to the simple resolver
        # — we can still correct the value even without cross-level comparison.
        total_hierarchy_vals = sum(len(v) for v in hierarchy_values.values())
        if total_hierarchy_vals == 0:
            logger.debug(
                f"FilterResolver: hierarchy cache empty for '{effective_column}' axis "
                f"({hierarchy}) — fetching on-demand from DB."
            )
            live_vals = _fetch_distinct(effective_column)
            if live_vals:
                logger.info(
                    f"FilterResolver: on-demand fetch got {len(live_vals)} values "
                    f"for '{effective_column}' — using simple resolver."
                )
                return _resolve_simple(llm, effective_column, guessed, live_vals)
            # Truly no data for this column — nothing to correct
            logger.warning(
                f"FilterResolver: no values found for '{effective_column}' "
                "(column may not exist in DB) — skipping correction."
            )
            return {
                "match": guessed, "matched_column": effective_column,
                "confidence": 1.0, "clarify": False, "question": None,
                "interpretation": None,
            }

        return _resolve_hierarchy_entity(
            llm, question, effective_column, guessed, hierarchy_values, hierarchy
        )
    else:
        vals = cached_values.get(effective_column) or _fetch_distinct(effective_column)
        return _resolve_simple(llm, effective_column, guessed, vals)


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

class FilterResolverAgent:
    """
    Resolves SQL filter literals against actual DB values.

    For product-hierarchy columns (mega_category, manufacturer, category,
    sub_category, brand, subbrand, ppg) uses 4-phase intent-first resolution:
      1. Classify intended granularity from the question's linguistic signals.
      2. Score candidates across ALL hierarchy levels (level-alignment dominant).
      3. Apply contextual override when top hit is finer than the intended level.
      4. Default to coarser level when ambiguous — safer for analytics contexts.

    For all other columns (market, customer, division, etc.) uses the original
    single-level LLM fuzzy matcher.
    """

    def __init__(self):
        # The filter resolver's LLM tasks (intent classification and fuzzy
        # value matching from a list) are well within the capability of the
        # fast model.  Using "fast" here saves ≈1-2 s per filter without any
        # accuracy loss — the heavy lifting is done by the algorithmic
        # hierarchy-scoring logic, not by the LLM.
        self.llm = create_llm("fast")

    def resolve(self, state: AgentState) -> dict:
        sql      = state.get("sql_query", "")
        question = state.get("question", "")
        if not sql:
            return {}

        # Fix bare-integer quarter_nielsen filters BEFORE any other step.
        # These are numeric literals and therefore invisible to the string-
        # filter walker; we must catch them here with a direct regex pass.
        sql = _fix_quarter_integer(sql)

        logger.info("FilterResolver: scanning SQL for filter values to verify")

        filters = _extract_string_filters(sql)
        if not filters:
            # Even if no string filters remain, return the (possibly fixed) SQL.
            if sql != state.get("sql_query", ""):
                return {"sql_query": sql}
            logger.info("FilterResolver: no string filter literals found — skipping")
            return {}

        logger.info(
            f"FilterResolver: {len(filters)} filter(s) to verify: "
            + ", ".join(f"{c}='{v}'" for c, v in filters)
        )

        # Load all cached distinct values in ONE call — no DB hit when cache exists.
        # If the cache file is absent this will fetch everything from the DB once
        # and persist the result, so subsequent requests are always fast.
        cached_values = _get_all_cached_values()
        logger.debug("FilterResolver: value cache loaded")

        # Resolve all filters in parallel
        value_corrections:  Dict[Tuple[str, str], str]             = {}
        column_corrections: Dict[Tuple[str, str], Tuple[str, str]] = {}
        clarifications:  List[str] = []
        interpretations: List[str] = []
        filter_log:      List[Dict] = []

        # ── Business-rule pre-pass (no LLM, runs before the parallel loop) ───
        # Any filter whose value matches a hard-coded rule is overridden
        # immediately; the rest fall through to normal LLM-based resolution.
        q_lower = question.lower()
        filters_for_llm: List[Tuple[str, str]] = []

        for col, val in filters:
            matched_rule: Optional[Dict] = None
            val_lower = val.lower().strip()

            for rule in _BUSINESS_RULE_OVERRIDES:
                # Skip if question doesn't match
                if not any(kw in q_lower for kw in rule["question_keywords"]):
                    continue
                # Skip if value doesn't match (already canonical → leave alone)
                if val.strip() == rule["force_value"]:
                    continue
                if not any(kw in val_lower for kw in rule["value_keywords"]):
                    continue
                matched_rule = rule
                break

            if matched_rule:
                fc = matched_rule["force_column"]
                fv = matched_rule["force_value"]
                logger.info(
                    f"FilterResolver: BUSINESS RULE — "
                    f"'{col}'='{val}' forced to '{fc}'='{fv}'"
                )
                column_corrections[(col, val)] = (fc, fv)
                filter_log.append({
                    "column":         col,
                    "sql_value":      val,
                    "db_match":       fv,
                    "matched_column": fc,
                    "confidence":     1.0,
                    "action":         "business_rule",
                })
            else:
                filters_for_llm.append((col, val))

        # Replace filters list: only unresolved filters go to the LLM loop
        filters = filters_for_llm

        # All remaining filters (not handled by business rules) are resolved
        # in parallel via LLM — one thread per filter.
        if filters:
            with ThreadPoolExecutor(max_workers=len(filters)) as pool:
                futures = {
                    pool.submit(
                        _resolve_one,
                        self.llm,
                        question,
                        col,
                        val,
                        cached_values,
                    ): (col, val)
                    for col, val in filters
                }
                for future in as_completed(futures):
                    col, val = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.error(f"FilterResolver: resolution failed for {col}='{val}': {exc}")
                        result = {
                            "match": val, "matched_column": col,
                            "confidence": 0.0, "clarify": False,
                            "question": None, "interpretation": None,
                        }

                    match          = result.get("match") or val
                    matched_column = result.get("matched_column") or col
                    confidence     = float(result.get("confidence", 0.0))
                    clarify        = result.get("clarify", False)
                    question_text  = result.get("question")
                    interpretation = result.get("interpretation")

                    log_entry = {
                        "column":         col,
                        "sql_value":      val,
                        "db_match":       match,
                        "matched_column": matched_column,
                        "confidence":     confidence,
                    }

                    if interpretation:
                        interpretations.append(interpretation)

                    if clarify or confidence < _AUTO_CORRECT_THRESHOLD:
                        logger.warning(
                            f"FilterResolver: '{col}'='{val}' — "
                            f"low confidence ({confidence:.2f}), needs clarification"
                        )
                        if question_text:
                            clarifications.append(question_text)
                        log_entry["action"] = "clarify"

                    elif matched_column != col:
                        # Cross-column correction: hierarchy level changed
                        logger.info(
                            f"FilterResolver: '{col}'='{val}' -> "
                            f"'{matched_column}'='{match}' "
                            f"(conf={confidence:.2f}) — cross-column correction"
                        )
                        column_corrections[(col, val)] = (matched_column, match)
                        log_entry["action"] = "column_corrected"

                    elif match != val:
                        logger.info(
                            f"FilterResolver: '{col}' '{val}' -> '{match}' "
                            f"(conf={confidence:.2f}) — value corrected"
                        )
                        value_corrections[(col, val)] = match
                        log_entry["action"] = "corrected"

                    else:
                        log_entry["action"] = "unchanged"

                    filter_log.append(log_entry)

        # Apply corrections to SQL
        corrected_sql = sql
        if value_corrections or column_corrections:
            corrected_sql = _rewrite_sql(sql, value_corrections, column_corrections)

        # Summary log
        n_val      = sum(1 for e in filter_log if e["action"] == "corrected")
        n_col      = sum(1 for e in filter_log if e["action"] == "column_corrected")
        n_rule     = sum(1 for e in filter_log if e["action"] == "business_rule")
        n_same     = sum(1 for e in filter_log if e["action"] == "unchanged")
        n_clarify  = sum(1 for e in filter_log if e["action"] == "clarify")
        logger.info(
            f"FilterResolver done: {n_rule} business-rule, "
            f"{n_val} value-corrected, "
            f"{n_col} column+value-corrected, "
            f"{n_same} unchanged, {n_clarify} need clarification"
        )

        out: dict = {"filter_log": filter_log}

        if value_corrections or column_corrections:
            out["sql_query"] = corrected_sql

        if interpretations:
            out["filter_interpretations"] = interpretations

        if clarifications:
            intro = "Before I run the query, I need a quick clarification:\n\n"
            if interpretations:
                intro = (
                    "Note: " + " | ".join(interpretations) + "\n\n" + intro
                )
            intro += "\n".join(f"• {q}" for q in clarifications)
            out["direct_response"]     = intro
            out["needs_clarification"] = True
            out["pending_filter_clarification"] = [
                {
                    "column":         e["column"],
                    "matched_column": e.get("matched_column"),
                    "sql_value":      e["sql_value"],
                    "db_match":       e.get("db_match"),
                }
                for e in filter_log if e.get("action") == "clarify"
            ]

        return out


# ── LangGraph node ────────────────────────────────────────────────────────────

def filter_resolver_node(state: AgentState) -> dict:
    """LangGraph node wrapper for FilterResolverAgent."""
    return _filter_resolver_agent.resolve(state)


# ── Module-level singletons ───────────────────────────────────────────────────
# FilterResolverAgent is stateless after __init__ (self.llm is immutable).
# The singleton is shared safely across threads — each resolve() call is
# completely independent.
_filter_resolver_agent = FilterResolverAgent()


# ── Eagerly start warmup at import time ───────────────────────────────────────
# The daemon thread runs in the background while the rest of the application
# initialises.  The JSON cache file is read (or built from the DB) once, then
# served from memory for the lifetime of the process.
initialize_value_cache()

