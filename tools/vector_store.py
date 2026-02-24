"""
Vector store for dynamic few-shot example retrieval.
Examples are sourced from nielsen_few_shots.yaml.
"""

from typing import List, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger
from config import settings
import os
import yaml


def _build_embeddings() -> Embeddings:
    """
    Return the appropriate embedding model for the active provider.

    groq  → HuggingFace all-MiniLM-L6-v2  (local, no API cost)
    dbrx  → Databricks embedding model via OpenAI-compatible endpoint.

            DBRX_EMBEDDING_ENDPOINT_URL should be the full invocations URL, e.g.:
              https://<workspace>.gcp.databricks.com/serving-endpoints
                /databricks-gte-large-en/invocations

            The base_url (everything up to and including /serving-endpoints)
            and model name are derived automatically from that URL.
    """
    if settings.llm_provider.lower() == "dbrx":
        from langchain_openai import OpenAIEmbeddings

        endpoint_url = settings.dbrx_embedding_endpoint_url.strip()
        if not endpoint_url:
            raise ValueError(
                "DBRX_EMBEDDING_ENDPOINT_URL is not set in .env. "
                "Set it to the full Databricks invocations URL, e.g.: "
                "https://<workspace>/serving-endpoints/databricks-gte-large-en/invocations"
            )

        # Derive base_url and model from the full invocations URL.
        # URL shape: {workspace}/serving-endpoints/{model}/invocations
        stripped = endpoint_url.rstrip("/")
        if stripped.endswith("/invocations"):
            stripped = stripped[: -len("/invocations")]
        # stripped is now: {workspace}/serving-endpoints/{model}
        base_url, _, model = stripped.rpartition("/")
        # base_url: https://<workspace>/serving-endpoints

        logger.info(
            f"Embeddings: Databricks '{model}' "
            f"via {base_url}"
        )
        return OpenAIEmbeddings(
            model=model,
            api_key=settings.dbrx_api_key,
            base_url=base_url,
            # Databricks endpoints only accept strings, not token-ID arrays.
            # Disabling length checking prevents the OpenAI client from
            # chunking long texts into token arrays before sending.
            check_embedding_ctx_length=False,
        )

    # Default: local HuggingFace
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info(f"Embeddings: HuggingFace '{settings.embedding_model}' (local)")
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def _collection_name() -> str:
    """
    Return a provider-scoped ChromaDB collection name.

    HuggingFace and Databricks embeddings have different vector dimensions
    (384 vs 1024+), so they MUST live in separate collections or ChromaDB
    will raise a dimension-mismatch error.
    """
    suffix = "_dbrx" if settings.llm_provider.lower() == "dbrx" else "_hf"
    return settings.chroma_collection_name + suffix


class FewShotRetriever:
    """
    Manages a vector store of SQL examples for dynamic few-shot learning.
    """
    
    def __init__(self):
        self.enabled = settings.enable_dynamic_few_shot
        
        if not self.enabled:
            logger.info("Dynamic few-shot learning disabled")
            return
        
        # Provider-aware embedding model
        self.embeddings = _build_embeddings()

        # Provider-scoped collection (prevents dimension-mismatch between groq/dbrx)
        self._collection_name = _collection_name()
        
        # Initialize vector store
        persist_directory = settings.vector_store_path
        os.makedirs(persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            collection_name=self._collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        logger.info(
            f"Few-shot retriever initialised — collection='{self._collection_name}', "
            f"provider='{settings.llm_provider}'"
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
                docs.append(Document(
                    page_content=ex["question"],
                    metadata={
                        "sql": ex["sql"],
                        "explanation": ex["explanation"],
                        "category": ex["category"],
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
    
    def retrieve(self, question: str, k: int = None) -> List[Dict]:
        """
        Retrieve most relevant examples for a question.
        
        Args:
            question: User's question
            k: Number of examples to retrieve
            
        Returns:
            List of example dicts
        """
        if not self.enabled:
            return []
        
        k = k or settings.few_shot_examples_count
        
        try:
            # Similarity search
            results = self.vectorstore.similarity_search(question, k=k)
            
            examples = []
            for doc in results:
                examples.append({
                    "question": doc.page_content,
                    "sql": doc.metadata.get("sql", ""),
                    "explanation": doc.metadata.get("explanation", ""),
                    "schema_context": doc.metadata.get("schema_context", ""),
                    "category": doc.metadata.get("category", ""),
                    "complexity": doc.metadata.get("complexity", "domain-specific")
                })
            
            logger.info(f"Retrieved {len(examples)} similar examples")
            return examples
            
        except Exception as e:
            logger.error(f"Error retrieving examples: {e}")
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
    few_shot_retriever.vectorstore = Chroma(
        collection_name=few_shot_retriever._collection_name,
        embedding_function=few_shot_retriever.embeddings,
        persist_directory=settings.vector_store_path,
    )

    few_shot_retriever._seed_from_yaml(yaml_path=yaml_path)
    logger.info("seed_examples() completed – vector store ready")
