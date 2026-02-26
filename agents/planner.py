"""Planner Agent: Breaks down complex questions into logical steps."""

import re
import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from core.state import AgentState
from core.llm_factory import create_llm
from config import settings

# Provider-aware table reference for prompt context
_TABLE_REF: str = (
    settings.dbx_full_table
    if settings.llm_provider.lower() == "dbrx"
    else "nielsen_pos"
)

# ── Nielsen schema category metadata ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nielsen_schema import CATEGORIES_DESCRIPTION, ALL_CATEGORIES   # noqa: E402

# Build a compact summary of available categories for the planner prompt.
_CATEGORIES_BLOCK = "\n".join(
    f"  • {cat}: {desc}"
    for cat, desc in CATEGORIES_DESCRIPTION.items()
)


class PlannerAgent:
    """Decomposes natural language questions into structured logical plans
    and identifies the minimal set of schema categories required."""

    def __init__(self):
        self.llm = create_llm("fast")

        # System prompt for logical planning + category selection
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data architect specializing in Nielsen POS SQL query planning.
The database has a SINGLE table called `{table_ref}`.

Your task: Decompose the user's question into clear, numbered logical steps AND identify
the minimal set of schema column categories needed to answer it.

═══════════════════════════════════════════════════════════
STEP 0 — DOMAIN KEYWORD DISAMBIGUATION  (resolve FIRST)
═══════════════════════════════════════════════════════════
Some phrases in a question describe the ANALYTICAL DOMAIN (which metric to use),
NOT a geographic filter.  Resolve these BEFORE applying any geographic logic.

  Analytical domain phrases → metric column to use (NOT a filter column):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ "Distribution & Availability"  → metric = tdp                       │
  │ "Distribution"                 → metric = tdp                       │
  │ "Availability"                 → metric = tdp                       │
  │ "TDP" / "Total Distribution"   → metric = tdp                       │
  │ "Sales" / "Revenue"            → metric = sales_dollar              │
  │ "Volume" / "Units"             → metric = sales_units               │
  │ "Velocity" / "Rate of Sale"    → metric = sales_units / tdp ratio   │
  │ "Display" / "Merchandising"    → metric = display                   │
  │ "Promotional"/"Promo" + dollar → metric = promotional_dollar cols   │
  │ "Promotional"/"Promo" + units  → metric = promotional_units cols    │
  │ "Promotional"/"Promo" + TDP    → metric = promotional_tdp cols      │
  └──────────────────────────────────────────────────────────────────────┘

  RULE: If a phrase matches an analytical domain above, it sets the METRIC.
        Do NOT interpret it as a customer, division, or market filter.

═══════════════════════════════════════════════════════════
STEP 1 — GEOGRAPHIC FILTER IDENTIFICATION
═══════════════════════════════════════════════════════════
Classify the geographic scope by the TYPE of entity the user mentions,
not by exact naming patterns.  Users speak naturally — short names are fine;
the downstream filter resolver will map them to the exact DB value.

  ENTITY TYPE → COLUMN  (pick the first that applies):

  customer  — the user names a RETAILER, STORE CHAIN, or RETAIL CHANNEL
              e.g. "Walmart", "Target", "Kroger", "Amazon", "Club",
                   "Dollar General", "Mass", "Convenience", "Drug",
                   "Grocery", "eComm", any named retail account
              → filter column = customer

  division  — the user names a GEOGRAPHIC DIVISION or SALES REGION
              e.g. "South Atlantic", "Mountain", "Pacific", "New England",
                   "Great Plains", "Southeast", "Northeast", "Midwest"
              → filter column = division

  market    — the user names a SPECIFIC METRO AREA, CITY, or DMA
              e.g. "New York", "Los Angeles", "Chicago", "Dallas"
              → filter column = market

  (default) — no retailer, division, or market is mentioned
              → filter column = total   (use value 'Total US xAOC + Conv')

  Priority order: customer > division > market > total (default)
  Write the filter column name in the plan so the SQL generator uses it.
  Do NOT invent the exact DB string — the filter resolver handles that.

═══════════════════════════════════════════════════════════
AVAILABLE SCHEMA CATEGORIES  (choose only what is needed)
═══════════════════════════════════════════════════════════
{categories_block}

Selection rules:
- `temporal`  and `geographic` are needed for virtually every query (dates + geo filters).
- `category_hierarchy` only if the query references mega_category, category, or sub_category.
- `product_hierarchy` only if the query references brand, manufacturer, subbrand, ppg, product, upc, or pack.
- `base_metrics` for any generic sales / units / TDP / display KPI.
- `promotional_dollar` only if the query references promo dollar splits.
- `promotional_units`  only if the query references promo unit splits.
- `promotional_tdp`    only if the query references promo TDP splits.

═══════════════════════════════════════════════════════════
PLANNING GUIDELINES
═══════════════════════════════════════════════════════════
1. Apply STEP 0 first: resolve domain keywords → metric columns.
2. Apply STEP 1 next: identify the geographic filter (or default to total).
3. Identify the core intent (ranking, period comparison, growth analysis, share).
4. Resolve follow-up references — if the question says "same product" or "that brand", use the
   conversation history to identify what it refers to and name it explicitly in the plan.
5. Break down into atomic logical steps: filters → aggregations → calculations → comparisons.
6. Define metrics and formulas explicitly (e.g., tdp_growth = (current_tdp - prior_tdp) / prior_tdp * 100).
7. YTD (Year-to-Date) comparisons MUST follow this pattern:
   a. YTD <year> = all rows WHERE period_date <= '<latest Saturday in <year>'
      AND period_date >= '<year>-01-01'  (or equivalent year_month range).
   b. For YTD vs prior-year comparison, align both windows to the SAME day-of-year:
      YTD current: period_date BETWEEN '<year>-01-01' AND <max_period_date in current year>
      YTD prior:   period_date BETWEEN '<prior_year>-01-01'
                                   AND ADD_MONTHS(<max_period_date_current_year>, -12)
      (SQLite equivalent: date(<max_period_date>, '-1 year'))
   c. NEVER compare raw annual totals when the question asks for YTD.
8. quarter_nielsen is always a STRING with Q prefix: 'Q1', 'Q2', 'Q3', 'Q4'.
   NEVER use bare integers (1, 2, 3, 4) when filtering quarter_nielsen.

═══════════════════════════════════════════════════════════
OUTPUT FORMAT  (follow this exactly)
═══════════════════════════════════════════════════════════
PLAN:
1. <first step>
2. <second step>
...

SCHEMA_CATEGORIES: <comma-separated list of needed categories from the list above>

Do NOT write SQL. Output only the PLAN block and the SCHEMA_CATEGORIES line.

Conversation history (for resolving follow-up questions):
{conversation_history}"""),
            ("user", "Question: {question}")
        ])

        self.chain = self.prompt | self.llm

    def plan(self, state: AgentState) -> dict:
        """Generate a logical plan and select relevant schema categories."""
        logger.info("PLANNER: Decomposing question into logical steps")
        question = state["question"]

        # Build conversation history text for follow-up resolution
        history = state.get("conversation_history") or []
        if history:
            history_text = "\n\n".join(
                f"Q: {h['question']}\nA: {h['nl_response']}"
                for h in history[-4:]
            )
        else:
            history_text = "None — this is the first question"

        try:
            response = self.chain.invoke({
                "question": question,
                "conversation_history": history_text,
                "categories_block": _CATEGORIES_BLOCK,
                "table_ref": _TABLE_REF,
            })
            raw = response.content

            # ── Parse PLAN block ─────────────────────────────────────────────
            plan_match = re.search(r"PLAN:\s*(.*?)(?=SCHEMA_CATEGORIES:|$)", raw, re.DOTALL | re.IGNORECASE)
            plan = plan_match.group(1).strip() if plan_match else raw.strip()

            steps = re.findall(r"^\d+\..*$", plan, re.MULTILINE)
            logger.info(f"Generated plan with {len(steps)} steps")

            # ── Parse SCHEMA_CATEGORIES line ─────────────────────────────────
            cats_match = re.search(r"SCHEMA_CATEGORIES:\s*(.+)", raw, re.IGNORECASE)
            if cats_match:
                raw_cats = [c.strip().lower() for c in cats_match.group(1).split(",")]
                # Keep only valid category names
                selected_cats = [c for c in raw_cats if c in ALL_CATEGORIES]
            else:
                # Fallback: use all categories if parsing fails
                selected_cats = ALL_CATEGORIES

            # Always ensure temporal + geographic are included (they supply filters)
            for mandatory in ("temporal", "geographic"):
                if mandatory not in selected_cats:
                    selected_cats.insert(0, mandatory)

            logger.info(f"Schema categories selected by planner: {selected_cats}")

            return {
                "plan": plan,
                "plan_steps": steps,
                "relevant_schema_categories": selected_cats,
                "iterations": 0,
                "should_retry": True,
            }
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return {"error": f"Planning failed: {str(e)}", "should_retry": False}


# ── Module-level singleton ───────────────────────────────────────────────────
# PlannerAgent is stateless beyond __init__; self.llm and self.chain are
# immutable after construction and the cached LLM is shared safely.
_planner_agent = PlannerAgent()


# Node function for LangGraph
def planner_node(state: AgentState) -> dict:
    """LangGraph node wrapper for PlannerAgent."""
    return _planner_agent.plan(state)
