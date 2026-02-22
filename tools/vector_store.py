"""
Vector store for dynamic few-shot example retrieval.
Examples are sourced from nielsen_few_shots.yaml.

Embedding backend is chosen by the current llm_provider:
  "groq"  → HuggingFace local sentence-transformers
  "dbrx"  → Databricks Foundation Model embedding endpoint

A separate ChromaDB collection is kept per provider so that switching
providers doesn't corrupt the existing index.
"""

from typing import List, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from config import settings
from core.llm_factory import get_embedding_function, get_llm
import os
import yaml

# ── All categories that exist in nielsen_few_shots.yaml ──────────────────────
KNOWN_CATEGORIES: List[str] = [
    "MARKET_SHARE",
    "PERFORMANCE_TRACKING",
    "PERIOD_COMPARISON",
    "GROWTH_DECLINE_ANALYSIS",
    "SALES_ANALYSIS",
    "TREND_ANALYSIS",
    "TOP_PERFORMING",
    "WEEKLY_VELOCITY",
    "NEW_PRODUCT_IDENTIFICATION",
    "COMPETITIVE_ANALYSIS",
    "PERCENTAGE_CONTRIBUTION",
    "ROLLBACK_YTD",
    "DISCOUNT_DEPTH",
    "AVERAGE_METRICS",
]

_CATEGORY_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Nielsen retail analytics query classifier.

Given a user question, identify which of the following query categories apply.
A question may belong to ONE or MULTIPLE categories.

Available categories and when to pick them:
- MARKET_SHARE         : ratio of one entity's sales to a broader market/category total
- PERFORMANCE_TRACKING : how a product/brand/category has performed on a metric over time
- PERIOD_COMPARISON    : comparing a metric between two specific time periods (months, quarters, years)
- GROWTH_DECLINE_ANALYSIS : percentage growth or decline, YoY / QoQ / MoM rates
- SALES_ANALYSIS       : total or aggregated sales (dollars or units) without comparison
- TREND_ANALYSIS       : evolution or trend of a metric over multiple time points
- TOP_PERFORMING       : ranking, top-N or bottom-N entities by a metric
- WEEKLY_VELOCITY      : sales units divided by TDP (rate of sale per distribution point)
- NEW_PRODUCT_IDENTIFICATION : products launched/new in a given period vs prior periods
- COMPETITIVE_ANALYSIS : comparing one manufacturer's performance against competitors
- PERCENTAGE_CONTRIBUTION : contribution of a subset to a total (e.g. top 50% of revenue)
- ROLLBACK_YTD         : year-to-date sales anchored to the latest available period vs same period last year
- DISCOUNT_DEPTH       : average selling price vs base (non-promoted) price gap
- AVERAGE_METRICS      : average weekly/monthly sales, average TDP, or other averages

Rules:
- Return ONLY a comma-separated list of matching category names from the list above.
- No explanations, no extra text, no markdown.
- If multiple categories apply, list all of them (e.g. MARKET_SHARE,PERIOD_COMPARISON).
- Always return at least one category."""),
    ("user", "Question: {question}"),
])


class FewShotRetriever:
    """
    Manages a vector store of SQL examples for dynamic few-shot learning.
    Provider-aware: embedding model and ChromaDB collection are selected
    based on settings.llm_provider at initialisation time.
    """

    def __init__(self):
        self.enabled = settings.enable_dynamic_few_shot

        if not self.enabled:
            logger.info("Dynamic few-shot learning disabled")
            return

        # Provider-aware embeddings
        self.embeddings = get_embedding_function()

        # Use a separate collection per provider to avoid dimension mismatches
        provider = settings.llm_provider
        collection_name = f"{settings.chroma_collection_name}_{provider}"
        persist_directory = settings.vector_store_path
        os.makedirs(persist_directory, exist_ok=True)

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        self._collection_name = collection_name

        logger.info(
            f"Few-shot retriever initialised | provider={provider} | "
            f"collection={collection_name}"
        )

        # Auto-seed from YAML when the collection is empty
        if self.vectorstore._collection.count() == 0:
            logger.info("Vector store is empty – seeding from nielsen_few_shots.yaml")
            self._seed_from_yaml()

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------

    def _load_yaml_examples(self, yaml_path: str) -> List[Dict]:
        """
        Parse nielsen_few_shots.yaml and return a flat list of example dicts.

        Each entry in the YAML has this shape (under a category key):
            - query: <natural language question>
              sql:   <SQL string>
              explanation: <optional string>

        Returns:
            List of dicts with keys: question, sql, explanation, category
        """
        examples = []
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.warning(f"Unexpected YAML structure in {yaml_path}")
                return examples

            for category, entries in data.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    # Skip malformed entries or nested category keys (YAML quirk)
                    if not isinstance(entry, dict):
                        continue
                    query = entry.get("query") or entry.get("question")
                    sql = entry.get("sql")
                    if not query or not sql:
                        continue
                    examples.append({
                        "question": query.strip(),
                        "sql": sql.strip() if isinstance(sql, str) else sql,
                        "explanation": (entry.get("explanation") or "").strip(),
                        "category": category,
                        # Capture market-share-specific numerator/denominator logic
                        "share_numerator": (entry.get("share_numerator") or "").strip(),
                        "share_denominator": (entry.get("share_denominator") or "").strip(),
                    })

        except FileNotFoundError:
            logger.error(f"Few-shots YAML not found: {yaml_path}")
        except yaml.YAMLError as exc:
            logger.error(f"Failed to parse YAML {yaml_path}: {exc}")

        return examples

    def _seed_from_yaml(self, yaml_path: str = None):
        """Load examples from the YAML file and add them to the vector store."""
        if yaml_path is None:
            # Resolve relative to this file → project root
            yaml_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "nielsen_few_shots.yaml"
            )

        examples = self._load_yaml_examples(yaml_path)
        if not examples:
            logger.warning("No examples loaded from YAML – vector store left empty")
            return

        try:
            docs = []
            for ex in examples:
                # Build a richer page_content so the embedding captures business
                # logic signals (numerator/denominator) for better similarity search
                content_parts = [ex["question"]]
                if ex.get("share_numerator"):
                    content_parts.append(f"NUMERATOR: {ex['share_numerator']}")
                if ex.get("share_denominator"):
                    content_parts.append(f"DENOMINATOR: {ex['share_denominator']}")
                if ex.get("explanation"):
                    content_parts.append(ex["explanation"])
                page_content = "\n".join(content_parts)

                docs.append(Document(
                    page_content=page_content,
                    metadata={
                        "sql": ex["sql"],
                        "explanation": ex["explanation"],
                        "category": ex["category"],
                        "share_numerator": ex.get("share_numerator", ""),
                        "share_denominator": ex.get("share_denominator", ""),
                        "schema_context": "",
                        "complexity": "domain-specific",
                    }
                ))
            self.vectorstore.add_documents(docs)
            logger.info(f"Seeded {len(docs)} Nielsen examples into vector store")
        except Exception as e:
            logger.error(f"Error seeding from YAML: {e}")
    
    def add_example(self, question: str, sql: str, explanation: str = None, 
                    schema_context: str = None, complexity: str = "medium"):
        """
        Add a new SQL example to the vector store.
        
        Args:
            question: Natural language question
            sql: Corresponding SQL query
            explanation: Optional explanation
            schema_context: Optional schema info
            complexity: Difficulty level (simple, medium, complex)
        """
        if not self.enabled:
            return
        
        try:
            doc = Document(
                page_content=question,
                metadata={
                    "sql": sql,
                    "explanation": explanation or "",
                    "schema_context": schema_context or "",
                    "complexity": complexity
                }
            )
            
            self.vectorstore.add_documents([doc])
            logger.info(f"Added example: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding example: {e}")
    
    def add_examples_batch(self, examples: List[Dict]):
        """
        Add multiple examples at once.
        
        Args:
            examples: List of example dicts with 'question', 'sql', etc.
        """
        if not self.enabled:
            return
        
        try:
            docs = []
            for ex in examples:
                doc = Document(
                    page_content=ex["question"],
                    metadata={
                        "sql": ex.get("sql", ""),
                        "explanation": ex.get("explanation", ""),
                        "schema_context": ex.get("schema_context", ""),
                        "complexity": ex.get("complexity", "medium")
                    }
                )
                docs.append(doc)
            
            self.vectorstore.add_documents(docs)
            logger.info(f"Added {len(docs)} examples to vector store")
            
        except Exception as e:
            logger.error(f"Error adding batch examples: {e}")
    
    def _classify_categories(self, question: str) -> List[str]:
        """
        Use a fast LLM to classify the question into one or more KNOWN_CATEGORIES.
        Falls back to all categories if classification fails.
        """
        try:
            llm = get_llm("fast")
            chain = _CATEGORY_CLASSIFIER_PROMPT | llm
            response = chain.invoke({"question": question})
            raw = response.content.strip()
            # Parse comma-separated list and validate against known categories
            candidates = [c.strip().upper() for c in raw.split(",")]
            matched = [c for c in candidates if c in KNOWN_CATEGORIES]
            if not matched:
                logger.warning(
                    f"Category classifier returned unknown categories '{raw}' "
                    f"— falling back to full search"
                )
                return []
            logger.info(f"Question classified into categories: {matched}")
            return matched
        except Exception as e:
            logger.error(f"Category classification failed: {e} — falling back to full search")
            return []

    def retrieve(self, question: str, k: int = None) -> List[Dict]:
        """
        Smart retrieval: classify the question into categories first, then
        retrieve the top ``per_category`` examples from EACH matched category.

        This ensures business-logic from every relevant category is represented
        while keeping the total prompt size bounded.

        Falls back to a flat similarity search if classification fails.
        """
        if not self.enabled:
            return []

        per_category = settings.few_shot_per_category_count  # default 2

        categories = self._classify_categories(question)

        if categories:
            return self._retrieve_per_category(question, categories, per_category)
        else:
            # Fallback: plain similarity search across all examples
            return self._retrieve_flat(question, k or settings.few_shot_examples_count)

    def _retrieve_per_category(
        self, question: str, categories: List[str], per_category: int
    ) -> List[Dict]:
        """
        For each category, run a similarity search filtered to that category
        and take the top ``per_category`` results. Deduplicate by question text.
        """
        seen_questions: set = set()
        all_examples: List[Dict] = []

        for cat in categories:
            try:
                results = self.vectorstore.similarity_search(
                    question,
                    k=per_category,
                    filter={"category": cat},
                )
                for doc in results:
                    q = doc.page_content.split("\n")[0]
                    if q in seen_questions:
                        continue
                    seen_questions.add(q)
                    all_examples.append({
                        "question":        q,
                        "sql":             doc.metadata.get("sql", ""),
                        "explanation":     doc.metadata.get("explanation", ""),
                        "schema_context":  doc.metadata.get("schema_context", ""),
                        "category":        doc.metadata.get("category", cat),
                        "share_numerator": doc.metadata.get("share_numerator", ""),
                        "share_denominator": doc.metadata.get("share_denominator", ""),
                        "complexity":      doc.metadata.get("complexity", "domain-specific"),
                    })
                logger.info(
                    f"  [{cat}] retrieved {len(results)} example(s) "
                    f"(top {per_category} per category)"
                )
            except Exception as e:
                logger.error(f"Error retrieving examples for category '{cat}': {e}")

        logger.info(
            f"Smart few-shot: {len(categories)} category(ies) × top {per_category}"
            f" = {len(all_examples)} unique example(s) total"
        )
        return all_examples

    def _retrieve_flat(self, question: str, k: int) -> List[Dict]:
        """Plain similarity search across all examples (fallback path)."""
        try:
            results = self.vectorstore.similarity_search(question, k=k)
            examples = []
            for doc in results:
                examples.append({
                    "question":        doc.page_content.split("\n")[0],
                    "sql":             doc.metadata.get("sql", ""),
                    "explanation":     doc.metadata.get("explanation", ""),
                    "schema_context":  doc.metadata.get("schema_context", ""),
                    "category":        doc.metadata.get("category", ""),
                    "share_numerator": doc.metadata.get("share_numerator", ""),
                    "share_denominator": doc.metadata.get("share_denominator", ""),
                    "complexity":      doc.metadata.get("complexity", "domain-specific"),
                })
            logger.info(f"Flat fallback: retrieved {len(examples)} example(s)")
            return examples
        except Exception as e:
            logger.error(f"Error in flat retrieval: {e}")
            return []

    
    def clear(self):
        """Clear all examples from vector store."""
        if self.enabled:
            # Delete and recreate collection
            self.vectorstore.delete_collection()
            logger.info("Vector store cleared")


# Global retriever instance
few_shot_retriever = FewShotRetriever()


def seed_examples(yaml_path: str = None):
    """
    Seed (or re-seed) the vector store from nielsen_few_shots.yaml.

    Clears the existing collection first so that re-runs stay idempotent.
    Pass ``yaml_path`` to override the default location (project root).
    """
    if not settings.enable_dynamic_few_shot:
        return

    # Wipe existing data so we don't accumulate duplicates
    few_shot_retriever.clear()

    # Re-initialise the collection after clear()
    provider = settings.llm_provider
    collection_name = f"{settings.chroma_collection_name}_{provider}"
    few_shot_retriever.vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=few_shot_retriever.embeddings,
        persist_directory=settings.vector_store_path,
    )

    few_shot_retriever._seed_from_yaml(yaml_path=yaml_path)
    logger.info("seed_examples() completed – vector store ready")

