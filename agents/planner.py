"""Planner Agent: Breaks down complex questions into logical steps."""

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from core.state import AgentState
from core.llm_factory import get_llm
from config import settings


class PlannerAgent:
    """Decomposes natural language questions into structured logical plans."""
    
    def __init__(self):
        self.llm = get_llm("reasoning")
        
        # System prompt for logical planning
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data architect specializing in Nielsen POS SQL query planning.
The database has a SINGLE table called `nielsen_pos`.

Your task: Decompose the user's question into clear, numbered logical steps.

Guidelines:
1. CLASSIFY the query intent — state it explicitly on the FIRST line as:
   INTENT: <one of MARKET_SHARE | PERIOD_COMPARISON | PERFORMANCE_TRACKING | GROWTH_ANALYSIS | RANKING | OTHER>

2. Resolve follow-up references — if the question says "same product" or "that brand", use the
   conversation history to identify what it refers to and name it explicitly in the plan.

3. Break down into atomic logical steps: filters → aggregations → calculations → comparisons.

4. Define metrics and formulas explicitly:
   - Market share  → NUMERATOR: <entity filter>  /  DENOMINATOR: <broader category/market filter>
   - Period compare → period1 metric  vs  period2 metric  + % change
   - Growth        → (current - prior) / prior * 100
   - TDP average   → SUM(tdp) / COUNT(DISTINCT period_date)
   - Velocity      → SUM(sales_units) / SUM(tdp)

5. Always specify the geographic filter priority:
   customer named? → filter by customer; division named? → filter by division;
   market named? → filter by market; otherwise → total = 'Total US xAOC + Conv'

6. For MARKET_SHARE queries, explicitly state:
   NUMERATOR FILTER: <exact entity column = value>
   DENOMINATOR FILTER: <broader column IN (value) representing the total market>

7. Identify entity hierarchy column (use HIGHEST-priority match):
   mega_category → manufacturer → category → sub_category → brand → subbrand → ppg

8. Output: A clear, numbered plan only. Do NOT write SQL.

Conversation history (for resolving follow-up questions):
{conversation_history}"""),
            ("user", "Question: {question}")
        ])
        
        self.chain = self.prompt | self.llm
    
    def plan(self, state: AgentState) -> dict:
        """Generate a logical plan for the question."""
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
            })
            plan = response.content
            
            # Extract numbered steps from the plan
            import re
            steps = re.findall(r'^\d+\..*$', plan, re.MULTILINE)
            logger.info(f"Generated plan with {len(steps)} steps")
            
            return {
                "plan": plan,
                "plan_steps": steps,
                "iterations": 0,
                "should_retry": True
            }
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return {"error": f"Planning failed: {str(e)}", "should_retry": False}


# Node function for LangGraph
def planner_node(state: AgentState) -> dict:
    """LangGraph node wrapper for PlannerAgent."""
    agent = PlannerAgent()
    return agent.plan(state)
