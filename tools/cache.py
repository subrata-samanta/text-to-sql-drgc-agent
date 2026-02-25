"""
Semantic caching layer for Text-to-SQL queries.
Reduces latency by caching query results based on semantic similarity.
"""

from typing import Optional, Tuple, Any
import hashlib
import json
from diskcache import Cache
from loguru import logger
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from config import settings


class SemanticCache:
    """
    Semantic cache that stores query results indexed by question embeddings.
    Uses cosine similarity to match semantically similar questions.

    The backend (disk cache + embedding model) is initialised lazily on first
    use so that keeping the cache *disabled* by default has zero startup cost.
    Enable at runtime via the UI toggle, --cache CLI flag, or ENABLE_SEMANTIC_CACHE=true.
    """

    def __init__(self):
        self.threshold = settings.cache_similarity_threshold
        self._cache: Optional[Cache] = None       # lazy
        self._embeddings = None                    # lazy
        if settings.enable_semantic_cache:
            self._init_backend()
        else:
            logger.info("Semantic cache disabled (off by default; enable via UI or --cache)")

    # ------------------------------------------------------------------
    def _init_backend(self) -> None:
        """Initialise the disk cache + embedding model (idempotent)."""
        if self._cache is not None:
            return  # already up
        try:
            self._cache = Cache("./cache/semantic_cache")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model
            )
            logger.info(f"Semantic cache backend ready (threshold: {self.threshold})")
        except Exception as exc:
            logger.error(f"Semantic cache init failed: {exc}")
            self._cache = None
            self._embeddings = None
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding vector for text."""
        try:
            embedding = self._embeddings.embed_query(text)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get(self, question: str) -> Optional[dict]:
        """
        Retrieve cached result for a question.
        Checks settings.enable_semantic_cache at call-time so a runtime
        toggle takes effect immediately without restarting.
        """
        if not settings.enable_semantic_cache:
            return None
        if self._cache is None:
            self._init_backend()
        if self._cache is None:          # init failed
            return None

        try:
            query_embedding = self._compute_embedding(question)
            if query_embedding is None:
                return None

            best_match = None
            best_similarity = 0.0

            for key in self._cache:
                if key.startswith("embedding_"):
                    cached_data = self._cache[key]
                    cached_embedding = np.array(cached_data["embedding"])
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    if similarity > best_similarity and similarity >= self.threshold:
                        best_similarity = similarity
                        best_match = cached_data

            if best_match:
                logger.info(f"Cache HIT (similarity: {best_similarity:.3f})")
                return best_match["result"]
            else:
                logger.info("Cache MISS")
                return None

        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    def set(self, question: str, result: dict):
        """
        Store query result in cache.
        Checks settings.enable_semantic_cache at call-time.
        """
        if not settings.enable_semantic_cache:
            return
        if self._cache is None:
            self._init_backend()
        if self._cache is None:          # init failed
            return

        try:
            embedding = self._compute_embedding(question)
            if embedding is None:
                return

            key = f"embedding_{hashlib.md5(question.encode()).hexdigest()}"
            cache_data = {
                "question":  question,
                "embedding": embedding.tolist(),
                "result":    result,
            }
            self._cache[key] = cache_data
            logger.info("Cached result for question")

        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def clear(self):
        """Clear all cached items."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Cache cleared")


# Global cache instance
semantic_cache = SemanticCache()
