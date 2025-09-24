"""
Advanced Token Management for LLM-based TOC Generation
Provides intelligent token counting, context window management, and adaptive chunking
"""

import tiktoken
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TokenBudget:
    """Token budget configuration for different LLM models"""
    max_context_window: int
    max_output_tokens: int
    system_prompt_tokens: int
    safety_margin: int = 100
    
    @property
    def available_input_tokens(self) -> int:
        """Calculate available tokens for input content"""
        return self.max_context_window - self.max_output_tokens - self.system_prompt_tokens - self.safety_margin
    
    @property
    def target_utilization_ratio(self) -> float:
        """Target utilization ratio to maximize LLM performance"""
        return 0.85  # Use 85% of available input tokens for optimal performance

    @property
    def optimal_input_tokens(self) -> int:
        """Optimal number of input tokens for best performance"""
        return int(self.available_input_tokens * self.target_utilization_ratio)

# Predefined token budgets for popular models
MODEL_TOKEN_BUDGETS = {
    "gpt-oss-20b": TokenBudget(
        max_context_window=128000,
        max_output_tokens=16000,
        system_prompt_tokens=200
    ),
   "gemini-2.5-flash-lite": TokenBudget(
        max_context_window=8192,
        max_output_tokens=1000,
        system_prompt_tokens=200
    ),
}

class TokenManager:
    """Advanced token management for LLM optimization"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", encoding_name: str = "cl100k_base"):
        """
        Initialize token manager with specific model configuration
        
        Args:
            model_name: LLM model name for token budget
            encoding_name: Tokenizer encoding (cl100k_base for GPT models)
        """
        self.model_name = model_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.available_tokens = None
        
        # Get token budget for the model
        if model_name in MODEL_TOKEN_BUDGETS:
            self.token_budget = MODEL_TOKEN_BUDGETS[model_name]
        else:
            logger.warning(f"Unknown model {model_name}, using default gemini-2.5-flash-lite budget")
            self.token_budget = MODEL_TOKEN_BUDGETS["gemini-2.5-flash-lite"]
        
        logger.info(f"TokenManager initialized for {model_name}")
        logger.info(f"Available input tokens: {self.token_budget.available_input_tokens}")
        logger.info(f"Optimal input tokens: {self.token_budget.optimal_input_tokens}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer"""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def estimate_vietnamese_tokens(self, text: str) -> int:
        """
        Estimate tokens for Vietnamese text with better accuracy
        Vietnamese typically uses 1.5-2 tokens per word
        """
        if not text:
            return 0
        
        # Basic token counting
        base_tokens = self.count_tokens(text)
        
        # Vietnamese adjustment factor
        vietnamese_factor = 1.2  # Vietnamese often uses slightly more tokens
        
        return int(base_tokens * vietnamese_factor)
    
    def calculate_optimal_top_k(self, chunks: List[str], title: str = "", 
                               available_tokens: int = None) -> int:
        """
        Calculate optimal top-k value to maximize token utilization
        
        Args:
            chunks: List of text chunks to select from
            title: Title for token calculation
            available_tokens: Override available tokens (optional)
            
        Returns:
            Optimal top-k value that maximizes token usage without exceeding limits
        """
        if not chunks:
            return 0
        
        # Calculate tokens for title and formatting overhead
        title_tokens = self.count_tokens(title)
        formatting_overhead = 50  # Overhead for formatting, separators, etc.
        
        # Available tokens for content - use override if provided
        if available_tokens is not None:
            self.available_tokens = available_tokens
        else:
            self.available_tokens = self.token_budget.optimal_input_tokens - title_tokens - formatting_overhead
        
        # Ensure available_tokens is positive
        if self.available_tokens <= 0:
            logger.warning(f"No available tokens after title and overhead. Title tokens: {title_tokens}")
            return 0
        
        # Sort chunks by token count (descending) to prioritize longer, more informative chunks
        chunk_tokens = [(chunk, self.count_tokens(chunk)) for chunk in chunks]
        chunk_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Find optimal top-k
        selected_tokens = 0
        optimal_k = 0
        
        for i, (chunk, tokens) in enumerate(chunk_tokens):
            if selected_tokens <= self.available_tokens:
                selected_tokens += tokens
                optimal_k = i + 1
            else:
                break
        
        # Ensure we select at least 1 chunk if possible
        optimal_k = max(1, optimal_k) if chunks else 0
        
        logger.info(f"Optimal top-k: {optimal_k} (using {selected_tokens}/{self.available_tokens} tokens)")
        
        return optimal_k
    
    def adaptive_chunking(self, text: str, max_chunk_tokens: int = None) -> List[str]:
        """
        Create adaptive chunks based on token limits and semantic boundaries
        
        Args:
            text: Input text to chunk
            max_chunk_tokens: Maximum tokens per chunk (defaults to 1/4 of optimal input)
            
        Returns:
            List of optimally sized chunks
        """
        if not text:
            return []
        
        if max_chunk_tokens is None:
            max_chunk_tokens = self.token_budget.optimal_input_tokens // 4
        
        # Split by sentences first (Vietnamese sentence patterns)
        sentences = self._split_vietnamese_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds limit, save current chunk
            if current_tokens + sentence_tokens > max_chunk_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Created {len(chunks)} adaptive chunks (max {max_chunk_tokens} tokens each)")
        
        return chunks
    
    def _split_vietnamese_sentences(self, text: str) -> List[str]:
        """Split Vietnamese text into sentences using appropriate delimiters"""
        import re
        
        # Vietnamese sentence delimiters
        sentence_delimiters = r'[.!?]\s+|[。！？]\s*'
        sentences = re.split(sentence_delimiters, text)
        
        # Filter out empty sentences and clean whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    def _truncate_text_to_tokens(self, text: str) -> str:
        """Truncate text to specified token count while preserving sentence boundaries"""
        if not text:
            return ""
        
        # Ensure available_tokens is set, fallback to optimal_input_tokens
        if self.available_tokens is None:
            self.available_tokens = self.token_budget.optimal_input_tokens
            
        if estimate_tokens(text, self.model_name) <= self.available_tokens:
            return text
        sentences = self._split_vietnamese_sentences(text)

        truncated = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens <= self.available_tokens:
                truncated += " " + sentence if truncated else sentence
                current_tokens += sentence_tokens
            else:
                break
        
        return truncated.strip()
def create_token_manager(model_name: str = "gemini-2.5-flash-lite") -> TokenManager:
    """Factory function to create token manager"""
    return TokenManager(model_name=model_name)

def estimate_tokens(text: str, model_name: str = "gemini-2.5-flash-lite") -> int:
    """Quick utility to estimate tokens for text"""
    manager = create_token_manager(model_name)
    return manager.count_tokens(text)

