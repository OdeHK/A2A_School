"""
Hybrid Memory Service
====================
Combines LangChain's ConversationBufferMemory (runtime) with MongoDB persistence.

Features:
- Fast in-memory conversation tracking
- Persistent storage in MongoDB
- Auto-summarization every 10 messages
- Session-based history retrieval
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


class HybridMemoryService:
    """
    Hybrid approach combining LangChain runtime memory with MongoDB persistence.
    
    Architecture:
    1. Runtime: ConversationBufferMemory (fast access, last N messages)
    2. Persistence: MongoDB (full conversation history)
    3. Auto-summarization: Every 10 messages using ConversationSummaryMemory
    
    Usage:
        memory = HybridMemoryService(mongo_collection, llm, session_id="user123")
        memory.add_user_message("What is photosynthesis?")
        memory.add_ai_message("Photosynthesis is the process...")
        context = memory.get_context()  # For RAG queries
    """
    
    def __init__(
        self,
        mongo_collection: Collection,
        llm,  # LLM for summarization
        session_id: str,
        buffer_size: int = 10,
        summarize_threshold: int = 10
    ):
        """
        Initialize Hybrid Memory Service.
        
        Args:
            mongo_collection: MongoDB collection for conversation history
            llm: Language model for summarization (from llm_service)
            session_id: Unique session identifier (user_id or session_uuid)
            buffer_size: Number of recent messages to keep in buffer
            summarize_threshold: Create summary after this many messages
        """
        self.mongo_collection = mongo_collection
        self.llm = llm
        self.session_id = session_id
        self.buffer_size = buffer_size
        self.summarize_threshold = summarize_threshold
        
        # LangChain runtime memory (fast access)
        self.buffer_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Summary memory for condensing long conversations
        self.summary_memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="conversation_summary",
            return_messages=True
        )
        
        # Message counter for auto-summarization
        self.message_count = 0
        self.last_summary = None
        
        # Load existing conversation from MongoDB
        self._load_from_mongodb()
        
        logger.info(f"✅ HybridMemoryService initialized for session: {session_id}")
    
    def _load_from_mongodb(self):
        """Load conversation history from MongoDB into runtime memory."""
        try:
            # Find all messages for this session, sorted by timestamp
            messages = list(self.mongo_collection.find(
                {"session_id": self.session_id}
            ).sort("timestamp", 1))
            
            if not messages:
                logger.info(f"No existing conversation found for session: {self.session_id}")
                return
            
            # Load last N messages into buffer
            recent_messages = messages[-self.buffer_size:]
            for msg in recent_messages:
                if msg["role"] == "user":
                    self.buffer_memory.chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    self.buffer_memory.chat_memory.add_ai_message(msg["content"])
            
            self.message_count = len(messages)
            
            # Load summary if it exists
            summary_doc = self.mongo_collection.find_one({
                "session_id": self.session_id,
                "type": "summary"
            })
            if summary_doc:
                self.last_summary = summary_doc["content"]
            
            logger.info(f"📚 Loaded {len(messages)} messages from MongoDB ({len(recent_messages)} in buffer)")
            
        except Exception as e:
            logger.error(f"❌ Error loading conversation from MongoDB: {e}")
    
    def add_user_message(self, message: str):
        """
        Add a user message to memory.
        
        Args:
            message: User's message text
        """
        # Add to runtime memory
        self.buffer_memory.chat_memory.add_user_message(message)
        
        # Persist to MongoDB
        self._save_to_mongodb(role="user", content=message)
        
        self.message_count += 1
        logger.debug(f"👤 User message added (count: {self.message_count})")
        
        # Check if summarization needed
        self._check_summarization()
    
    def add_ai_message(self, message: str):
        """
        Add an AI response to memory.
        
        Args:
            message: AI's response text
        """
        # Add to runtime memory
        self.buffer_memory.chat_memory.add_ai_message(message)
        
        # Persist to MongoDB
        self._save_to_mongodb(role="assistant", content=message)
        
        self.message_count += 1
        logger.debug(f"🤖 AI message added (count: {self.message_count})")
        
        # Check if summarization needed
        self._check_summarization()
    
    def _save_to_mongodb(self, role: str, content: str):
        """Save a message to MongoDB."""
        try:
            document = {
                "session_id": self.session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow(),
                "type": "message"
            }
            self.mongo_collection.insert_one(document)
            logger.debug(f"💾 Message saved to MongoDB")
        except Exception as e:
            logger.error(f"❌ Error saving to MongoDB: {e}")
    
    def _check_summarization(self):
        """Check if conversation should be summarized."""
        if self.message_count % self.summarize_threshold == 0:
            logger.info(f"📝 Summarization threshold reached ({self.message_count} messages)")
            self._create_summary()
    
    def _create_summary(self):
        """Create a summary of the conversation so far."""
        try:
            # Get all messages from buffer
            messages = self.buffer_memory.chat_memory.messages
            
            if len(messages) < 2:
                return
            
            # Use LangChain's ConversationSummaryMemory
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    self.summary_memory.chat_memory.add_user_message(msg.content)
                elif isinstance(msg, AIMessage):
                    self.summary_memory.chat_memory.add_ai_message(msg.content)
            
            # Get the summary
            summary = self.summary_memory.load_memory_variables({})
            summary_text = str(summary.get("conversation_summary", ""))
            
            self.last_summary = summary_text
            
            # Save summary to MongoDB
            try:
                summary_doc = {
                    "session_id": self.session_id,
                    "type": "summary",
                    "content": summary_text,
                    "message_count": self.message_count,
                    "timestamp": datetime.utcnow()
                }
                self.mongo_collection.replace_one(
                    {"session_id": self.session_id, "type": "summary"},
                    summary_doc,
                    upsert=True
                )
                logger.info(f"✅ Summary created and saved ({len(summary_text)} chars)")
            except Exception as e:
                logger.error(f"❌ Error saving summary: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error creating summary: {e}")
    
    def get_context(self, include_summary: bool = True) -> str:
        """
        Get conversation context for RAG queries.
        
        Args:
            include_summary: Include conversation summary if available
            
        Returns:
            Formatted conversation context
        """
        context_parts = []
        
        # Add summary if available
        if include_summary and self.last_summary:
            context_parts.append(f"**Conversation Summary:**\n{self.last_summary}\n")
        
        # Add recent messages from buffer
        messages = self.buffer_memory.chat_memory.messages
        if messages:
            context_parts.append("**Recent Conversation:**")
            for msg in messages[-5:]:  # Last 5 messages
                if isinstance(msg, HumanMessage):
                    context_parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_parts.append(f"AI: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_full_history(self) -> List[Dict]:
        """
        Get full conversation history from MongoDB.
        
        Returns:
            List of message dictionaries
        """
        try:
            messages = list(self.mongo_collection.find(
                {"session_id": self.session_id, "type": "message"}
            ).sort("timestamp", 1))
            return messages
        except Exception as e:
            logger.error(f"❌ Error retrieving full history: {e}")
            return []
    
    def clear_session(self):
        """Clear all messages for this session."""
        try:
            # Clear runtime memory
            self.buffer_memory.clear()
            self.summary_memory.clear()
            
            # Clear MongoDB
            self.mongo_collection.delete_many({"session_id": self.session_id})
            
            self.message_count = 0
            self.last_summary = None
            
            logger.info(f"🗑️ Session {self.session_id} cleared")
        except Exception as e:
            logger.error(f"❌ Error clearing session: {e}")
    
    def get_statistics(self) -> Dict:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        buffer_count = len(self.buffer_memory.chat_memory.messages)
        
        return {
            "session_id": self.session_id,
            "total_messages": self.message_count,
            "buffer_messages": buffer_count,
            "has_summary": self.last_summary is not None,
            "summary_length": len(self.last_summary) if self.last_summary else 0
        }
