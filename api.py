"""FastAPI REST API for the Text-to-SQL agent."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from loguru import logger

from graph import run_agent_async
from tools import seed_examples, semantic_cache, few_shot_retriever
from core.database import db_manager
from config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Text-to-SQL Agent API",
    description="REST API for the state-of-the-art Text-to-SQL multi-agent system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    use_cache: bool = Field(default=True, description="Whether to use semantic cache")
    max_iterations: int = Field(default=3, description="Max correction attempts")


class QueryResponse(BaseModel):
    success: bool
    sql_query: Optional[str] = None
    result_preview: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    iterations: int = 0
    cache_hit: bool = False
    plan: Optional[str] = None
    relevant_tables: Optional[List[str]] = None


class ExampleRequest(BaseModel):
    question: str
    sql: str
    explanation: Optional[str] = None
    complexity: str = Field(default="medium", pattern="^(simple|medium|complex)$")


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    total_tables: int
    cache_enabled: bool
    few_shot_enabled: bool


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and system health."""
    try:
        tables = db_manager.get_all_table_names()
        return HealthResponse(
            status="healthy",
            database_connected=True,
            total_tables=len(tables),
            cache_enabled=settings.enable_semantic_cache,
            few_shot_enabled=settings.enable_dynamic_few_shot
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database_connected=False,
            total_tables=0,
            cache_enabled=settings.enable_semantic_cache,
            few_shot_enabled=settings.enable_dynamic_few_shot
        )


# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Execute a natural language query against the database."""
    logger.info(f"Query: {request.question}")
    
    try:
        result = await run_agent_async(request.question)
        
        return QueryResponse(
            success=result.get("error") is None,
            sql_query=result.get("sql_query"),
            result_preview=result.get("result_preview"),
            error=result.get("error"),
            execution_time_ms=result.get("execution_time_ms"),
            total_latency_ms=result.get("total_latency_ms"),
            iterations=result.get("iterations", 0),
            cache_hit=result.get("cache_hit", False),
            plan=result.get("plan"),
            relevant_tables=result.get("relevant_tables")
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Database schema endpoints
@app.get("/schema/tables")
async def get_tables():
    """Get list of all tables in the database."""
    try:
        tables = db_manager.get_all_table_names()
        return {"tables": tables, "count": len(tables)}
    except Exception as e:
        logger.error(f"Error fetching tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema/table/{table_name}")
async def get_table_schema(table_name: str):
    """Get detailed schema for a specific table."""
    try:
        metadata = db_manager.get_table_metadata(table_name)
        schema = db_manager.get_schema_for_tables([table_name])
        
        return {
            "table_name": table_name,
            "metadata": metadata,
            "schema": schema
        }
    except Exception as e:
        logger.error(f"Error fetching schema for {table_name}: {e}")
        raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")


# Few-shot example endpoints
@app.post("/examples")
async def add_example(request: ExampleRequest):
    """Add a new few-shot example to the vector store."""
    try:
        few_shot_retriever.add_example(
            question=request.question,
            sql=request.sql,
            explanation=request.explanation,
            complexity=request.complexity
        )
        return {"message": "Example added successfully"}
    except Exception as e:
        logger.error(f"Error adding example: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/examples/seed")
async def seed_default_examples(background_tasks: BackgroundTasks):
    """Seed the vector store with default examples."""
    try:
        background_tasks.add_task(seed_examples)
        return {"message": "Seeding examples in background"}
    except Exception as e:
        logger.error(f"Error seeding examples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples/search")
async def search_examples(query: str, k: int = 3):
    """Search for similar examples."""
    try:
        examples = few_shot_retriever.retrieve(query, k=k)
        return {"examples": examples, "count": len(examples)}
    except Exception as e:
        logger.error(f"Error searching examples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache management endpoints
@app.delete("/cache")
async def clear_cache():
    """Clear the semantic cache."""
    try:
        semantic_cache.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Text-to-SQL Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting Text-to-SQL Agent API")
    logger.info(f"Database: {settings.database_uri}")
    logger.info(f"Semantic Cache: {'Enabled' if settings.enable_semantic_cache else 'Disabled'}")
    logger.info(f"Few-Shot Learning: {'Enabled' if settings.enable_dynamic_few_shot else 'Disabled'}")
    # Pre-build the filter-resolver value cache so the first query is never delayed.
    from agents.filter_resolver import initialize_value_cache
    initialize_value_cache()


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Text-to-SQL Agent API")
    db_manager.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
