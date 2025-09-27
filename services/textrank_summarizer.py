"""
Pure TextRank Summarization Strategy (No LLM)
Efficient content extraction for use with downstream LLM processing
"""

from typing import Dict, List, Optional, Any
import logging
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

from services.token_manager import TokenManager, create_token_manager

logger = logging.getLogger(__name__)

class HybridSummarizerStrategy:
    """
    Pure TextRank strategy for intelligent content extraction
    
    Designed to work with downstream LLM processing (like summarizer_agent)
    
    Benefits:
    - 100% free operation (no LLM calls)
    - Intelligent content selection using TextRank
    - Token-aware chunking and optimization
    - Perfect for multi-stage processing pipelines
    """
    
    def __init__(self, 
                 embedding_model: str = "Alibaba-NLP/gte-multilingual-base",
                 cache_folder: str = "./model",
                 token_manager: Optional[TokenManager] = None,
                 cost_optimization: bool = True):
        """
        Initialize pure TextRank summarizer
        
        Args:
            embedding_model: Model for TextRank embeddings
            cache_folder: Cache folder for embeddings
            token_manager: Token manager instance
            cost_optimization: Enable content optimization (always True for TextRank)
        """
        # TextRank components
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            cache_folder=cache_folder,
            model_kwargs={"trust_remote_code": True,
                          "device": "auto"}
        )
        
        # Token management (no LLM needed)
        self.token_manager = token_manager or create_token_manager("gemini-2.5-flash-lite")  # For token calculations only
        
        # Configuration
        self.embedding_model = embedding_model
        self.cost_optimization = True  # Always optimize since we're not using LLM
        
        # Performance tracking
        self.textrank_selections = 0
        
        logger.info(f"Pure TextRank Summarizer initialized:")
        logger.info(f"  - Embedding Model: {embedding_model}")
        logger.info(f"  - Mode: Pure TextRank (No LLM)")
        logger.info(f"  - Designed for: Downstream LLM processing")
        
    @property
    def strategy_name(self) -> str:
        return "pure_textrank"
    
    def generate_content(self, text: str, title: str, **kwargs) -> str:
        """
        Generate intelligent content extraction using pure TextRank
        
        Process:
        1. Use TextRank to select optimal chunks (no LLM calls)
        2. Combine selected chunks optimally
        3. Return structured content for downstream LLM processing
        
        Args:
            text: Input text to process
            title: Section title for context
            **kwargs: Additional parameters
            
        Returns:
            Intelligently selected and optimized content
        """
        logger.info(f"Generating TextRank content extraction for: {title}")
        
        if not text or not text.strip():
            return f"Nội dung {title}: (không có nội dung)."
        
        try:
            # PHASE 1: TextRank-based intelligent chunk selection
            selected_chunks = self._textrank_chunk_selection(text, title)
            
            if not selected_chunks:
                return f"Nội dung {title}: (không thể xử lý nội dung)."
            
            # PHASE 2: Optimize content for downstream processing
            optimized_content = self._optimize_content_structure(selected_chunks, title)
            
            # PHASE 3: Format for downstream LLM
            final_content = self._format_for_downstream_llm(optimized_content, title)
            
            logger.info(f"Content extracted for '{title}': "
                       f"{len(selected_chunks)} chunks selected")
            
            return final_content
            
        except Exception as e:
            logger.error(f"Error in TextRank content extraction for {title}: {e}")
            return f"Nội dung {title}: Lỗi khi xử lý ({str(e)})."
    
    def _textrank_chunk_selection(self, text: str, title: str) -> List[str]:
        """
        Use TextRank to select most important chunks efficiently
        
        Args:
            text: Input text
            title: Section title
            
        Returns:
            Selected chunks based on TextRank scoring
        """
        logger.debug("Performing TextRank chunk selection")
        
        # Use adaptive chunking from token_manager (MUCH BETTER!)
        chunks = self.token_manager.adaptive_chunking(text)
        
        if not chunks:
            return []
        
        if len(chunks) == 1:
            return chunks
        
        try:
            # Calculate optimal number of chunks based on token budget
            optimal_k = self.token_manager.calculate_optimal_top_k(chunks, title)
            
            # Get embeddings for chunks
            chunk_embeddings = self.embeddings.embed_documents(chunks)
            
            # Create similarity matrix
            similarity_matrix = cosine_similarity(chunk_embeddings)
            
            # Apply TextRank algorithm
            graph = nx.from_numpy_array(similarity_matrix)
            pagerank_scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
            
            # Add title relevance boost
            title_boosted_scores = self._boost_title_relevance(
                chunks, pagerank_scores, title
            )
            
            # Select top chunks maintaining document order
            ranked_chunks = sorted(
                [(score, i, chunk) for i, (chunk, score) in 
                 enumerate(zip(chunks, title_boosted_scores.values()))],
                reverse=True
            )
            
            # Select top-k chunks and reorder by original position
            selected_indices = sorted([idx for _, idx, _ in ranked_chunks[:optimal_k]])
            selected_chunks = [chunks[i] for i in selected_indices]
            
            self.textrank_selections += 1
            
            logger.debug(f"TextRank selected {len(selected_chunks)}/{len(chunks)} chunks")
            
            return selected_chunks
            
        except Exception as e:
            logger.warning(f"TextRank selection failed: {e}, using simple selection")
            # Fallback to simple selection
            return chunks[:3]
    
    def _boost_title_relevance(self, chunks: List[str], pagerank_scores: Dict, title: str) -> Dict:
        """
        Boost chunks that are semantically relevant to the title
        
        Args:
            chunks: Text chunks
            pagerank_scores: Original PageRank scores
            title: Section title
            
        Returns:
            Boosted scores dictionary
        """
        title_words = set(title.lower().split())
        boosted_scores = {}
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            
            # Calculate title relevance
            overlap_ratio = len(title_words.intersection(chunk_words)) / len(title_words) if title_words else 0
            
            # Boost factor (1.0 = no boost, up to 1.5 = significant boost)
            boost_factor = 1.0 + (overlap_ratio * 0.5)
            
            # Apply boost to PageRank score
            boosted_scores[i] = pagerank_scores[i] * boost_factor
        
        return boosted_scores
    
    def _optimize_content_structure(self, chunks: List[str], title: str) -> str:
        """
        Optimize selected chunks for downstream processing
        
        Args:
            chunks: Selected chunks from TextRank
            title: Section title
            
        Returns:
            Well-structured content for downstream LLM processing
        """
        if not chunks:
            return ""
        
        # Combine chunks with clear separators
        combined_content = "\n\n".join(chunks)
        
    
        combined_content = self.token_manager._truncate_text_to_tokens(
                combined_content
        )
        
        return combined_content
    
    def _format_for_downstream_llm(self, content: str, title: str) -> str:
        """
        Format content optimally for downstream LLM processing
        
        Args:
            content: Optimized content
            title: Section title
            
        Returns:
            Formatted content ready for LLM processing
        """
        if not content or not content.strip():
            return f"Nội dung {title}: (không có nội dung)."
        
        # Clean and structure the content
        content = content.strip()
        
        # Ensure the content is properly formatted with title context
        if not content.startswith(title):
            formatted_content = f"Nội dung phần '{title}':\n\n{content}"
        else:
            formatted_content = content
        
        return formatted_content
    

# Factory function
def create_hybrid_summarizer(
    embedding_model: str = "Alibaba-NLP/gte-multilingual-base",
    **kwargs  # Accept but ignore LLM-related kwargs for backward compatibility
) -> HybridSummarizerStrategy:
    """Factory function to create pure TextRank summarizer"""
    return HybridSummarizerStrategy(
        embedding_model=embedding_model,
        cost_optimization=True
    )