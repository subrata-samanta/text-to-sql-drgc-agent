"""
Critic Agent (Refiner): Validates, executes, and corrects SQL queries.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from core.state import AgentState
from core.database import db_manager
from config import settings


class CriticAgent:
    """
    Validates, executes, and corrects SQL queries.
    Implements closed-loop error correction with execution feedback.
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.groq_model_reasoning,
            temperature=settings.groq_temperature,
            groq_api_key=settings.groq_api_key
        )
        
        # Prompt for error correction
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL debugging expert. A query failed and you must fix it.

Your Task:
1. Analyze the error message carefully
2. Review the schema to understand what went wrong
3. Identify the specific issue (wrong column, incorrect join, syntax error, etc.)
4. Generate a CORRECTED SQL query

Common Error Patterns:
- Column does not exist → Check schema for correct column names
- Table does not exist → Verify table name spelling
- Syntax error → Check SQL dialect requirements
- Ambiguous column → Add table aliases
- Join error → Verify foreign key relationships

CRITICAL — THIS IS SQLite (NOT MySQL / SQL Server):
- "no such function: YEAR"    → replace YEAR(col) with: year_nielsen column OR (year_month / 100)
- "no such function: MONTH"   → replace MONTH(col) with: year_month % 100
- "no such function: QUARTER" → use quarter_nielsen column directly
- "no such function: NOW"     → use date('now')
- "no such function: GETDATE"  → use date('now')
- "no such function: ISNULL"  → use COALESCE(a, b)
- "no such function: NVL"     → use COALESCE(a, b)
- "near FROM: syntax error"   → remove trailing comma before FROM / GROUP BY / ORDER BY
- TOP N not supported         → use LIMIT N
- year_month is a YYYYMM INTEGER (e.g. 202301); do NOT wrap it in YEAR() or strftime()

IMPORTANT: Return ONLY the fixed SQL query (no explanations, no markdown)

SCHEMA:
{schema_context}

ORIGINAL QUESTION:
{question}

FAILED SQL:
{sql_query}

ERROR MESSAGE:
{error}

LOGICAL PLAN (reference):
{plan}

Generate the CORRECTED SQL:"""),
            ("user", "Fix the query:")
        ])
    
    def execute_and_validate(self, state: AgentState) -> dict:
        """
        Execute SQL query and handle results/errors.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with results or error information
        """
        logger.info("CRITIC: Executing and validating SQL query")
        
        sql_query = state.get("sql_query")
        if not sql_query:
            return {
                "error": "No SQL query to execute",
                "should_retry": False
            }
        
        try:
            # Execute the query
            result, error, exec_time = db_manager.execute_query(
                sql_query,
                timeout=settings.query_timeout_seconds
            )
            
            if error:
                # Query failed - prepare for reflection
                logger.warning(f"Query execution failed: {error}")
                error_type = self._classify_error(error)
                
                return {
                    "error": error,
                    "error_type": error_type,
                    "query_result": None,
                    "execution_time_ms": exec_time,
                    "should_retry": True
                }
            else:
                # Query succeeded
                logger.info(f"Query executed successfully in {exec_time:.2f}ms")
                result_preview = self._format_result_preview(result)
                
                return {
                    "query_result": result,
                    "result_preview": result_preview,
                    "execution_time_ms": exec_time,
                    "error": None,
                    "should_retry": False
                }
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "error": str(e),
                "error_type": "runtime",
                "should_retry": True
            }
    
    def reflect_and_fix(self, state: AgentState) -> dict:
        """
        Analyze error and generate corrected SQL.
        
        Args:
            state: Current agent state with error information
            
        Returns:
            Updated state with corrected SQL
        """
        logger.info("CRITIC: Reflecting on error and fixing SQL")
        
        iterations = state.get("iterations", 0)
        
        # Check if we've exceeded max iterations
        if iterations >= settings.max_iterations:
            logger.error(f"Max iterations ({settings.max_iterations}) reached")
            return {
                "should_retry": False,
                "error": f"Failed to generate valid SQL after {settings.max_iterations} attempts"
            }
        
        question = state["question"]
        plan = state.get("plan", "")
        schema_context = state.get("schema_context", "")
        sql_query = state.get("sql_query", "")
        error = state.get("error", "")
        
        try:
            chain = self.reflection_prompt | self.llm
            
            response = chain.invoke({
                "question": question,
                "plan": plan,
                "schema_context": schema_context,
                "sql_query": sql_query,
                "error": error
            })
            
            # Clean the fixed SQL
            from agents.generator import SQLGeneratorAgent
            generator = SQLGeneratorAgent()
            fixed_sql = generator._clean_sql(response.content)
            
            logger.info(f"Generated corrected SQL (iteration {iterations + 1})")
            logger.debug(f"Fixed SQL: {fixed_sql}")
            
            return {
                "sql_query": fixed_sql,
                "iterations": iterations + 1,
                "should_retry": True
            }
            
        except Exception as e:
            logger.error(f"Reflection error: {e}")
            return {
                "error": f"Failed to correct SQL: {str(e)}",
                "should_retry": False
            }
    
    def _classify_error(self, error_msg: str) -> str:
        """
        Classify error type for better handling.
        
        Args:
            error_msg: Error message from database
            
        Returns:
            Error category
        """
        error_lower = error_msg.lower()
        
        if "column" in error_lower and ("does not exist" in error_lower or "not found" in error_lower):
            return "column_not_found"
        elif "table" in error_lower and ("does not exist" in error_lower or "not found" in error_lower):
            return "table_not_found"
        elif "syntax" in error_lower:
            return "syntax_error"
        elif "ambiguous" in error_lower:
            return "ambiguous_column"
        elif "timeout" in error_lower:
            return "timeout"
        else:
            return "runtime_error"
    
    def _format_result_preview(self, result, max_rows: int = 5) -> str:
        """
        Format query result for display.
        
        Args:
            result: Query result (list of rows or message)
            max_rows: Maximum rows to include in preview
            
        Returns:
            Formatted string preview
        """
        if isinstance(result, str):
            return result
        
        if not result:
            return "Query returned no results"
        
        try:
            # Handle list of dicts (preferred) or SQLAlchemy Row objects
            first = result[0]
            if isinstance(first, dict):
                rows = result[:max_rows]
            elif hasattr(first, '_mapping'):
                rows = [dict(row._mapping) for row in result[:max_rows]]
            elif hasattr(first, '_asdict'):
                rows = [row._asdict() for row in result[:max_rows]]
            else:
                return str(result[:max_rows])

            preview = f"Returned {len(result)} row(s). Preview:\n"
            for i, row in enumerate(rows, 1):
                preview += f"Row {i}: {row}\n"

            if len(result) > max_rows:
                preview += f"... ({len(result) - max_rows} more rows)"

            return preview
                
        except Exception as e:
            logger.warning(f"Could not format result: {e}")
            return str(result)[:500]  # Truncate to 500 chars


# Node functions for LangGraph
def executor_node(state: AgentState) -> dict:
    """LangGraph node wrapper for execution."""
    agent = CriticAgent()
    return agent.execute_and_validate(state)


def reflector_node(state: AgentState) -> dict:
    """LangGraph node wrapper for reflection/correction."""
    agent = CriticAgent()
    return agent.reflect_and_fix(state)
