"""
Memory System Package
====================
Provides conversation memory and context management for the AI teacher agent.

Features:
- LangChain integration for runtime memory
- MongoDB persistence for conversation history
- Automatic summarization every 10 messages
- Session management
"""

from .memory_service import HybridMemoryService

__all__ = ['HybridMemoryService']
