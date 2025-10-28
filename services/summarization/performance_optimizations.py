import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)


class EmbeddingModelCache:
    """
    Singleton cache for embedding models to avoid repeated loading.
    Thread-safe implementation.
    """
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_or_create_embeddings(self, model_name: str, cache_folder: str, **kwargs):
        """
        Get cached embedding model or create new one using embedding_service.
        
        Args:
            model_name: Name of the embedding model
            cache_folder: Cache folder for the model
            **kwargs: Additional model arguments
            
        Returns:
            Embedding model instance
        """
        cache_key = f"{model_name}_{cache_folder}"
        
        if cache_key not in self._models:
            with self._lock:
                # Double-check locking pattern
                if cache_key not in self._models:
                    from services.rag.embedding_service import HuggingFaceStrategy
                    
                    logger.info(f"Loading embedding model: {model_name} (first time)")
                    
                    # Use HuggingFaceStrategy's create_embedding method
                    strategy = HuggingFaceStrategy(
                        model=model_name,
                        cache_folder=cache_folder,
                        **kwargs
                    )
                    
                    self._models[cache_key] = strategy.embedding_instance
                    logger.info(f"✅ Embedding model cached: {cache_key}")
        else:
            logger.debug(f"♻️ Using cached embedding model: {cache_key}")
        
        return self._models[cache_key]
    
    def clear_cache(self):
        """Clear all cached models (useful for memory management)"""
        with self._lock:
            self._models.clear()
            logger.info("Embedding model cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached models"""
        return len(self._models)


class MemoryOptimizer:
    """Utilities for memory optimization"""
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache if available"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.debug("GPU cache cleared")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
        return False
    
    @staticmethod
    def estimate_memory_usage(obj) -> int:
        """Estimate memory usage of an object in bytes"""
        import sys
        return sys.getsizeof(obj)

# Global instances
_embedding_cache = EmbeddingModelCache()

def get_embedding_cache() -> EmbeddingModelCache:
    """Get the global embedding cache instance"""
    return _embedding_cache