"""
Schema Linker Agent: Identifies relevant columns from the Nielsen POS schema
and builds rich schema context (with CTE SQL conventions) for SQL generation.
"""

import os
import sys
from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from core.state import AgentState
from config import settings

# ── Import the definitive Nielsen schema ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nielsen_schema import schema as NIELSEN_SCHEMA           # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers (computed once at import time)
# ─────────────────────────────────────────────────────────────────────────────

TABLE_NAME: str = NIELSEN_SCHEMA["table"]    # "nielsen_pos"
_COLUMNS: dict  = NIELSEN_SCHEMA["columns"]  # nested dict keyed by group


def _build_full_schema_text() -> str:
    """
    Render the full Nielsen schema as a structured text block that can be
    dropped verbatim into an LLM prompt.
    """
    lines = [f"TABLE: {TABLE_NAME}", "=" * 70]
    for group, cols in _COLUMNS.items():
        lines.append(f"\n── {group.upper().replace('_', ' ')} ──")
        for col_name, description in cols.items():
            lines.append(f"  • {col_name}:{description}")
    return "\n".join(lines)


def _flat_column_list() -> List[str]:
    """Return a flat list of every column name in the schema."""
    cols: List[str] = []
    for group_cols in _COLUMNS.values():
        cols.extend(group_cols.keys())
    return cols


FULL_SCHEMA_TEXT: str  = _build_full_schema_text()
ALL_COLUMNS: List[str] = _flat_column_list()


# ─────────────────────────────────────────────────────────────────────────────
# CTE convention block (appended to every schema context handed to generator)
# ─────────────────────────────────────────────────────────────────────────────

CTE_CONVENTIONS = f"""
══════════════════════════════════════════════════════════════════════════
SQL GENERATION RULES  –  MANDATORY FOR EVERY QUERY
══════════════════════════════════════════════════════════════════════════
1. ALWAYS write SQL using CTEs (WITH … AS (…)).  NEVER write flat queries.
2. Give each CTE a descriptive snake_case name that reflects its purpose
   (e.g., entity_sales, total_market_sales, period1_sales, ytd_current).
3. The ONLY table is:  {TABLE_NAME}
4. Every query MUST include exactly ONE geographic filter (priority order):
     a. customer  is mentioned → AND customer = '…'          (omit total / market)
     b. division  is mentioned → AND division = '…'          (omit total / market)
     c. market    is mentioned → AND market   = '…'          (omit total)
     d. (default)              → AND total    = 'Total US xAOC + Conv'
5. Use NULLIF(denominator, 0) in every division operation.
6. Round all percentages / ratios with ROUND(…, 2).
7. Alias every computed column with a meaningful snake_case name.
8. TDP average  →  SUM(tdp) / COUNT(DISTINCT period_date)
9. Velocity     →  SUM(sales_units) / SUM(tdp)
10. Display must NEVER be aggregated with SUM – use weighted average only.

──────────────────────────────────────────────────────────────────────────
CTE SKELETON  (adapt structure and names to the question)
──────────────────────────────────────────────────────────────────────────
WITH <entity_cte> AS (
    SELECT SUM(sales_dollar) AS entity_sales
    FROM {TABLE_NAME}
    WHERE <product_filter>
      AND <temporal_filter>               -- year_month / period_date / year_nielsen
      AND total = 'Total US xAOC + Conv'  -- or customer / division / market filter
),
<market_cte> AS (
    SELECT SUM(sales_dollar) AS total_sales
    FROM {TABLE_NAME}
    WHERE <broader_category_filter>
      AND <temporal_filter>
      AND total = 'Total US xAOC + Conv'
)
SELECT
    ROUND(
        (<entity_cte>.entity_sales / NULLIF(<market_cte>.total_sales, 0)) * 100.0,
        2
    ) AS market_share_pct
FROM <entity_cte>, <market_cte>;
══════════════════════════════════════════════════════════════════════════
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class SchemaLinkerAgent:
    """
    Schema-linking agent tailored to the Nielsen POS single-table model.

    Responsibilities
    ────────────────
    • Identify which columns from ``nielsen_pos`` are relevant to the question.
    • Surface column semantics (types, business rules) from nielsen_schema.py.
    • Inject CTE SQL conventions into every schema context it produces.
    """

    def __init__(self):
        self.llm = ChatGroq(
            model=settings.groq_model_fast,
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )

        # ── Column-selection prompt ──────────────────────────────────────────
        self.column_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a Nielsen POS data expert. The database contains exactly ONE table:

  Table: {TABLE_NAME}

FULL SCHEMA:
{FULL_SCHEMA_TEXT}

══════════════════════════════════════════════════════════════════════════
COLUMN SELECTION RULES
══════════════════════════════════════════════════════════════════════════
Return ONLY the column names strictly needed, as a comma-separated list.
No explanations. No extra text.

GEOGRAPHIC PRIORITY (pick exactly ONE group):
  • customer  named → include: customer
  • division  named → include: division
  • market    named → include: market
  • default         → include: total

TEMPORAL RULES:
  • weekly / specific week    → period_date
  • monthly / multi-month     → year_month
  • quarterly                 → quarter_nielsen  (+ year_nielsen)
  • annual / YTD              → year_nielsen     (+ year_month for YTD)

METRIC RULES:
  • "sales" / "revenue"       → sales_dollar
  • "units"                   → sales_units
  • "TDP" / "distribution"    → tdp
  • "velocity"                → sales_units, tdp
  • "display"                 → display
  • market share              → sales_dollar  (both numerator & denominator)
  • promotional question      → relevant sales_dollar_with_* / sales_units_with_*

Always include: one temporal column + one geographic column + metric column(s).

EXAMPLE:
  Question: "What was OREO brand dollar market share in Q3 2024?"
  Response: year_month, quarter_nielsen, year_nielsen, total, brand, category, sales_dollar"""),
            ("user", """Question: {question}

Plan: {plan}

All available columns: {all_columns}

Comma-separated column names only:"""),
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def select_columns(self, question: str, plan: str) -> List[str]:
        """
        Ask the LLM which columns are needed; validate against the real schema.

        Returns:
            List of valid column names from NIELSEN_SCHEMA.
        """
        try:
            chain = self.column_selection_prompt | self.llm
            response = chain.invoke({
                "question": question,
                "plan": plan,
                "all_columns": ", ".join(ALL_COLUMNS),
            })
            selected = [c.strip() for c in response.content.split(",")]
            # Keep only names that actually exist in the schema
            selected = [c for c in selected if c in ALL_COLUMNS]
            logger.info(f"Schema linker selected {len(selected)} columns: {selected}")
            return selected or ["year_month", "total", "sales_dollar"]
        except Exception as e:
            logger.error(f"Column selection error: {e}")
            return ["year_month", "total", "sales_dollar"]

    def _build_targeted_schema(self, columns: List[str]) -> str:
        """
        Build a concise schema text block containing ONLY the selected columns.
        """
        lines = [f"TABLE: {TABLE_NAME}", "=" * 70]
        for group, cols in _COLUMNS.items():
            relevant = {col: desc for col, desc in cols.items() if col in columns}
            if relevant:
                lines.append(f"\n── {group.upper().replace('_', ' ')} ──")
                for col_name, desc in relevant.items():
                    lines.append(f"  • {col_name}:{desc}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_schema(self, state: AgentState) -> dict:
        """
        Build the schema context for downstream SQL generation.

        State keys written
        ──────────────────
        relevant_tables : ["nielsen_pos"]
        schema_context  : targeted column schema + mandatory CTE conventions
        schema_metadata : {column → {group, description}} for selected columns
        """
        logger.info("SCHEMA LINKER: Selecting relevant Nielsen columns")

        question: str = state["question"]
        plan: str     = state.get("plan", "")

        try:
            selected_cols = self.select_columns(question, plan)

            # Targeted schema for the selected columns + obligatory CTE rules
            schema_context = self._build_targeted_schema(selected_cols) + CTE_CONVENTIONS

            # Column-level metadata dict for state
            schema_metadata: Dict[str, dict] = {}
            for group, cols in _COLUMNS.items():
                for col_name, desc in cols.items():
                    if col_name in selected_cols:
                        schema_metadata[col_name] = {
                            "group": group,
                            "description": desc.strip(),
                        }

            logger.info(
                f"Schema context ready — table='{TABLE_NAME}', "
                f"columns selected={len(selected_cols)}"
            )

            return {
                "relevant_tables": [TABLE_NAME],
                "schema_context": schema_context,
                "schema_metadata": schema_metadata,
            }

        except Exception as e:
            logger.error(f"Schema retrieval error: {e}")
            # Fallback: full schema so generation can still proceed
            return {
                "relevant_tables": [TABLE_NAME],
                "schema_context": FULL_SCHEMA_TEXT + CTE_CONVENTIONS,
                "schema_metadata": {},
                "error": f"Schema retrieval failed: {str(e)}",
                "should_retry": False,
            }


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def schema_linker_node(state: AgentState) -> dict:
    """LangGraph node wrapper for SchemaLinkerAgent."""
    agent = SchemaLinkerAgent()
    return agent.retrieve_schema(state)


def context_builder_node(state: AgentState) -> dict:
    """
    Parallel context builder: runs schema linking + few-shot retrieval simultaneously.

    Replaces the former sequential retrieve_few_shot → schema_retriever two-node
    chain. Schema and few-shot examples are fetched ONCE per question and reused
    on every retry (validation or execution), so there is no redundant I/O.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tools import few_shot_retriever as _few_shot_retriever

    question = state["question"]

    with ThreadPoolExecutor(max_workers=2) as ex:
        schema_future = ex.submit(SchemaLinkerAgent().retrieve_schema, state)
        few_shot_future = (
            ex.submit(_few_shot_retriever.retrieve, question)
            if settings.enable_dynamic_few_shot else None
        )
        schema_result = schema_future.result()
        examples = few_shot_future.result() if few_shot_future else []

    logger.info(
        f"context_builder: schema ready + "
        f"few_shot({len(examples)} example(s)) retrieved in parallel"
    )
    return {**schema_result, "few_shot_examples": examples}
